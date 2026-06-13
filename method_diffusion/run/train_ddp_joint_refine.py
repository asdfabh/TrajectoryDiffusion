import contextlib
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_ddp_joint import cleanup_ddp, is_main_process, reduce_tensor, setup_ddp
from method_diffusion.run.train_joint import normalize_dataset_name
from method_diffusion.run.train_joint_refine import (
    build_joint_proposals,
    get_joint_refiner_checkpoint_dir,
    load_frozen_hist_model,
    load_frozen_joint_fut_model,
    write_csv_row,
)
from method_diffusion.run.train_refine import compute_refiner_loss
from method_diffusion.utils.fut_utils import TrajectoryMetrics, select_closest_prediction

LOSS_KEYS = ["loss", "xy_loss", "fde_loss", "delta_norm", "gate"]


def build_distributed_loader(dataset, batch_size, num_workers, sampler, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        sampler=sampler,
        drop_last=drop_last,
    )


def load_frozen_hist_model_for_rank(args, dataset_name, device, rank):
    if is_main_process(rank):
        return load_frozen_hist_model(args, dataset_name, device)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_frozen_hist_model(args, dataset_name, device)


def load_frozen_joint_fut_model_for_rank(args, dataset_name, device, rank):
    if is_main_process(rank):
        return load_frozen_joint_fut_model(args, dataset_name, device)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_frozen_joint_fut_model(args, dataset_name, device)


def reduce_trajectory_metrics(metrics, device):
    if not dist.is_initialized():
        return metrics

    for name in [
        "total_coord_se",
        "total_de",
        "total_theta_abs_deg",
        "total_v_abs",
        "total_counts",
    ]:
        value = getattr(metrics, name).to(device=device, dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        setattr(metrics, name, value.cpu())
    return metrics


def train_epoch(args, hist_model, fut_model, refiner, dataloader, optimizer, device, epoch, dataset_name, enable_latent_bridge, rank):
    refiner.train()
    totals = {key: 0.0 for key in LOSS_KEYS}
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"DDP JointRefine Ep{epoch} Train",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        pred_hist, all_preds, fut, op_mask = build_joint_proposals(
            args,
            hist_model,
            fut_model,
            batch,
            device,
            dataset_name,
            enable_latent_bridge,
        )
        refined_all, aux = refiner(pred_hist, all_preds, fut_model.fut_dt)
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
            pbar.set_postfix({"loss": f"{loss.item():.5f}", "gate": f"{logs['gate'].item():.3f}"})

    stats = torch.tensor([totals[key] for key in LOSS_KEYS] + [float(num_batches)], device=device, dtype=torch.float64)
    stats = reduce_tensor(stats)
    denom = max(int(stats[-1].item()), 1)
    return {key: float(stats[idx].item()) / denom for idx, key in enumerate(LOSS_KEYS)}


@torch.no_grad()
def evaluate(args, hist_model, fut_model, refiner, dataloader, device, epoch, dataset_name, enable_latent_bridge, rank):
    refiner.eval()
    baseline_metrics = TrajectoryMetrics(fut_model.T)
    refined_metrics = TrajectoryMetrics(fut_model.T)
    refiner_model = refiner.module if hasattr(refiner, "module") else refiner
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"DDP JointRefine Ep{epoch} Val",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        pred_hist, all_preds, fut, op_mask = build_joint_proposals(
            args,
            hist_model,
            fut_model,
            batch,
            device,
            dataset_name,
            enable_latent_bridge,
        )
        pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        refined_all, _ = refiner_model(pred_hist, all_preds, fut_model.fut_dt)
        refined_pred, _, _ = select_closest_prediction(refined_all, fut, op_mask)
        baseline_metrics.update(pred_fut, fut, op_mask)
        refined_metrics.update(refined_pred, fut, op_mask)

        if is_main_process(rank):
            summary = refined_metrics.summary()
            last_idx = refined_pred.size(1) - 1
            pbar.set_postfix({
                "rmse": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
                "ade": f"{summary['ade_per_step_m'][last_idx]:.4f}",
                "fde": f"{summary['fde_per_step_m'][last_idx]:.4f}",
            })

    baseline_metrics = reduce_trajectory_metrics(baseline_metrics, device)
    refined_metrics = reduce_trajectory_metrics(refined_metrics, device)
    return baseline_metrics.summary(), refined_metrics.summary()


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    checkpoint_dir = get_joint_refiner_checkpoint_dir(dataset_name)
    csv_path = checkpoint_dir / "train_log.csv"
    enable_latent_bridge = int(args.enable_past_fut_latent_bridge) > 0

    try:
        if is_main_process(rank):
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DDP JointRefineTrain] Dataset: {dataset_name}")
            print(f"[DDP JointRefineTrain] Device: {device}")
            print(f"[DDP JointRefineTrain] World size: {world_size}")
            print(f"[DDP JointRefineTrain] Checkpoint dir: {checkpoint_dir}")
            print(f"[DDP JointRefineTrain] latent_bridge={int(enable_latent_bridge)}")

        train_path = str(get_split_path(args, dataset_name, "Train"))
        val_path = str(get_split_path(args, dataset_name, "Val"))
        if is_main_process(rank):
            print(f"[DDP JointRefineTrain] Train path: {train_path}")
            print(f"[DDP JointRefineTrain] Val path: {val_path}")

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
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        train_loader = build_distributed_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, drop_last=True)
        val_loader = build_distributed_loader(val_dataset, args.batch_size, args.num_workers, val_sampler, drop_last=False)

        hist_model = load_frozen_hist_model_for_rank(args, dataset_name, device, rank)
        fut_model = load_frozen_joint_fut_model_for_rank(args, dataset_name, device, rank)
        refiner = build_trajectory_refiner(args).to(device)
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.fut_refiner_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.num_epochs), 1))

        if dist.is_initialized():
            if device.type == "cuda":
                refiner = DDP(refiner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            else:
                refiner = DDP(refiner, find_unused_parameters=False)

        if is_main_process(rank):
            print("[DDP JointRefineTrain] Refiner: TABR-temporal-basis")

        best_rmse = float("inf")
        for epoch in range(1, int(args.num_epochs) + 1):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

            train_stats = train_epoch(
                args,
                hist_model,
                fut_model,
                refiner,
                train_loader,
                optimizer,
                device,
                epoch,
                dataset_name,
                enable_latent_bridge,
                rank,
            )
            baseline_summary, refined_summary = evaluate(
                args,
                hist_model,
                fut_model,
                refiner,
                val_loader,
                device,
                epoch,
                dataset_name,
                enable_latent_bridge,
                rank,
            )
            scheduler.step()

            last_idx = len(refined_summary["rmse_per_step_m"]) - 1
            selection_score = float(refined_summary["rmse_per_step_m"][last_idx].item())
            is_best = selection_score < best_rmse
            if is_best:
                best_rmse = selection_score

            if is_main_process(rank):
                current_lr = optimizer.param_groups[0]["lr"]
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

                refiner_state = refiner.module.state_dict() if hasattr(refiner, "module") else refiner.state_dict()
                state = {
                    "epoch": epoch,
                    "model_state_dict": refiner_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_rmse_m": best_rmse,
                    "refiner_type": "temporal_basis",
                    "resume_hist": args.resume_hist,
                    "resume_fut": args.resume_fut,
                    "enable_past_fut_latent_bridge": int(enable_latent_bridge),
                    "args": vars(args),
                }
                if epoch % int(args.save_interval) == 0:
                    torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
                if is_best:
                    torch.save(state, checkpoint_dir / "best.pth")

            if dist.is_initialized():
                dist.barrier()
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
