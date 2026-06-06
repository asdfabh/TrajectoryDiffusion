import contextlib
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_ddp_fut import (
    build_distributed_loader,
    cleanup_ddp,
    is_main_process,
    reduce_tensor,
    setup_ddp,
)
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_fut_refiner import (
    compute_refiner_loss,
    get_refiner_checkpoint_dir,
    load_frozen_fut_model,
    write_csv_row,
)
from method_diffusion.utils.fut_utils import TrajectoryMetrics, select_closest_prediction


REFINER_STAT_KEYS = (
    "loss",
    "xy_loss",
    "fde_loss",
    "delta_norm",
    "gate",
)


def load_frozen_fut_model_for_rank(args, device, rank):
    if is_main_process(rank):
        return load_frozen_fut_model(args, device)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_frozen_fut_model(args, device)


def reduce_metrics(metrics, device):
    for attr in (
        "total_coord_se",
        "total_de",
        "total_theta_abs_deg",
        "total_v_abs",
        "total_counts",
    ):
        value = getattr(metrics, attr).to(device=device, dtype=torch.float64)
        reduce_tensor(value)
        setattr(metrics, attr, value.cpu())
    return metrics.summary()


def train_epoch(args, fut_model, refiner, dataloader, optimizer, device, epoch, rank):
    fut_model.eval()
    refiner.train()
    totals = {key: 0.0 for key in REFINER_STAT_KEYS}
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"E6 DDP Ep{epoch} Train",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )

    for batch_idx, batch in enumerate(pbar, start=1):
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            args.feature_dim,
            device=device,
        )
        with torch.no_grad():
            all_preds = fut_model.forwardEvalMulti(
                hist,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                device,
                K=fut_model.fut_k,
            )

        refined_all, aux = refiner(hist, all_preds, fut_model.fut_dt)
        loss, logs = compute_refiner_loss(
            refined_all,
            fut,
            op_mask,
            aux,
            args.fut_refiner_fde_weight,
            args.fut_refiner_delta_weight,
            args.fut_refiner_gate_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += float(loss.item())
        for key, value in logs.items():
            totals[key] += float(value.item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.5f}",
                    "avg": f"{totals['loss'] / max(num_batches, 1):.5f}",
                    "gate": f"{logs['gate'].item():.3f}",
                }
            )

    stats = torch.tensor(
        [totals[key] for key in REFINER_STAT_KEYS] + [float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    reduce_tensor(stats)
    denom = max(int(stats[len(REFINER_STAT_KEYS)].item()), 1)
    return {
        key: float(stats[idx].item()) / denom
        for idx, key in enumerate(REFINER_STAT_KEYS)
    }


@torch.no_grad()
def evaluate(args, fut_model, refiner, dataloader, device, epoch, rank):
    fut_model.eval()
    refiner.eval()
    baseline_metrics = TrajectoryMetrics(fut_model.T)
    refined_metrics = TrajectoryMetrics(fut_model.T)
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"E6 DDP Ep{epoch} Val",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )

    for batch_idx, batch in enumerate(pbar, start=1):
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            args.feature_dim,
            device=device,
        )
        all_preds = fut_model.forwardEvalMulti(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            device,
            K=fut_model.fut_k,
        )
        pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        refined_all, _ = refiner(hist, all_preds, fut_model.fut_dt)
        refined_pred, _, _ = select_closest_prediction(refined_all, fut, op_mask)

        baseline_metrics.update(pred_fut, fut, op_mask)
        refined_metrics.update(refined_pred, fut, op_mask)

        if is_main_process(rank):
            summary = refined_metrics.summary()
            last_idx = refined_pred.size(1) - 1
            pbar.set_postfix(
                {
                    "rmse": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
                    "ade": f"{summary['ade_per_step_m'][last_idx]:.4f}",
                    "fde": f"{summary['fde_per_step_m'][last_idx]:.4f}",
                }
            )

    return reduce_metrics(baseline_metrics, device), reduce_metrics(refined_metrics, device)


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    dataset_name = str(args.dataset).strip().lower()
    checkpoint_dir = get_refiner_checkpoint_dir(dataset_name)
    csv_path = checkpoint_dir / "train_log.csv"

    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DDP RefinerTrain] Dataset: {dataset_name}")
        print(f"[DDP RefinerTrain] World size: {world_size}")
        print("[DDP RefinerTrain] Refiner: TABR-temporal-basis")
        print(f"[DDP RefinerTrain] Checkpoint dir: {checkpoint_dir}")

    train_path = str(get_split_path(args, dataset_name, "Train"))
    val_path = str(get_split_path(args, dataset_name, "Val"))
    if is_main_process(rank):
        print(f"[DDP RefinerTrain] Train path: {train_path}")
        print(f"[DDP RefinerTrain] Val path: {val_path}")
    train_dataset = build_trajectory_dataset(
        train_path,
        dataset_name,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    val_dataset = build_trajectory_dataset(
        val_path,
        dataset_name,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    train_loader = build_distributed_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler,
        drop_last=True,
    )
    val_loader = build_distributed_loader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=val_sampler,
        drop_last=False,
    )

    fut_model = load_frozen_fut_model_for_rank(args, device, rank)
    refiner = build_trajectory_refiner(args).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.fut_refiner_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.num_epochs), 1))

    if dist.is_initialized():
        if device.type == "cuda":
            refiner = DDP(refiner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            refiner = DDP(refiner, find_unused_parameters=False)

    best_rmse = float("inf")
    for epoch in range(1, int(args.num_epochs) + 1):
        train_sampler.set_epoch(epoch)
        train_stats = train_epoch(args, fut_model, refiner, train_loader, optimizer, device, epoch, rank)
        baseline_summary, refined_summary = evaluate(args, fut_model, refiner, val_loader, device, epoch, rank)
        last_idx = len(refined_summary["rmse_per_step_m"]) - 1
        selection_score = float(refined_summary["rmse_per_step_m"][last_idx].item())
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process(rank):
            write_csv_row(csv_path, epoch, train_stats, baseline_summary, refined_summary, current_lr)
            print(
                f"Epoch {epoch}/{args.num_epochs} | "
                f"loss={train_stats['loss']:.6f} | "
                f"gate={train_stats['gate']:.4f} | "
                f"baseline_rmse={baseline_summary['rmse_per_step_m'][last_idx].item():.4f} | "
                f"refined_rmse={refined_summary['rmse_per_step_m'][last_idx].item():.4f} | "
                f"refined_ade={refined_summary['ade_per_step_m'][last_idx].item():.4f} | "
                f"refined_fde={refined_summary['fde_per_step_m'][last_idx].item():.4f}"
            )

        scheduler.step()
        is_best = selection_score < best_rmse
        if is_best:
            best_rmse = selection_score

        if is_main_process(rank):
            refiner_state = refiner.module.state_dict() if hasattr(refiner, "module") else refiner.state_dict()
            state = {
                "epoch": epoch,
                "model_state_dict": refiner_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_rmse_m": best_rmse,
                "refiner_type": "temporal_basis",
                "args": vars(args),
            }
            if epoch % int(args.save_interval) == 0:
                torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
            if is_best:
                torch.save(state, checkpoint_dir / "best.pth")

        if dist.is_initialized():
            dist.barrier()

    cleanup_ddp()


if __name__ == "__main__":
    main()
