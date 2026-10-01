import contextlib
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.run.train_fut import (
    FUT_CHECKPOINT_DIR,
    LOSS_STAT_KEYS,
    init_csv_log,
    load_checkpoint,
    print_eval_summary,
    prepare_input_data,
    write_csv_log,
    write_tensorboard_log,
)
from method_diffusion.utils.fut_utils import (
    DistributedEvalSampler,
    TrajectoryMetrics, reduce_trajectory_metrics, print_trajectory_metrics,
    validation_metric_values, write_horizon_metrics,
)


def setup_ddp():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 0, 1, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return rank, local_rank, world_size, device


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def load_checkpoint_for_rank(resume_arg, checkpoint_dir, model, optimizer, scheduler, device, rank):
    if is_main_process(rank):
        return load_checkpoint(resume_arg, checkpoint_dir, model, optimizer, scheduler, device)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_checkpoint(resume_arg, checkpoint_dir, model, optimizer, scheduler, device)


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


def reduce_tensor(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank):
    model.train()
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", dynamic_ncols=True, disable=not is_main_process(rank))

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_logs = model(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key in LOSS_STAT_KEYS:
            totals[key] += float(loss_logs[key].item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "avg_loss": f"{(totals['loss'] / num_batches):.6f}",
                }
            )

    stats = torch.tensor(
        [totals[key] for key in LOSS_STAT_KEYS] + [float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    denom = max(int(stats[len(LOSS_STAT_KEYS)].item()), 1)

    stats_dict = {
        key: float(stats[idx].item()) / denom
        for idx, key in enumerate(LOSS_STAT_KEYS)
    }
    stats_dict["fut_k"] = int(getattr(model.module if hasattr(model, "module") else model, "fut_k", 0))
    return stats_dict


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, feature_dim, rank, return_summary=False):
    was_training = model.training
    fut_model = model.module if hasattr(model, "module") else model
    model.eval()
    metrics = TrajectoryMetrics(fut_model.T, num_candidates=fut_model.fut_k)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True, disable=not is_main_process(rank))
    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device, K=fut_model.fut_k)
        metrics.update(all_preds, fut, op_mask)
        if is_main_process(rank):
            summary = metrics.summary()
            pbar.set_postfix({
                "RMSE_full_selection": f"{summary['rmse_full_m']:.4f}",
                "RMSE_final_horizon": f"{summary['rmse_per_step_m'][-1]:.4f}",
                "minADE_final_horizon": f"{summary['min_ade_per_step_m'][-1]:.4f}",
                "minFDE_final_horizon": f"{summary['min_fde_per_step_m'][-1]:.4f}",
            })
    metrics = reduce_trajectory_metrics(metrics, device)
    summary = metrics.summary()
    if is_main_process(rank):
        print_trajectory_metrics(summary, f"Val epoch {epoch}", fut_model.fut_dt)
    model.train(was_training)
    return summary if return_summary else validation_metric_values(summary)


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    dataset_name = str(args.dataset).lower()
    checkpoint_dir = FUT_CHECKPOINT_DIR / dataset_name

    writer = None
    log_csv_path = None
    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir = checkpoint_dir / "log"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        log_csv_path = tensorboard_log_dir / "train_log.csv"
        init_csv_log(log_csv_path)
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    train_path = str(get_split_path(args, dataset_name, "Train"))
    val_path = str(get_split_path(args, dataset_name, "Val"))
    if is_main_process(rank):
        print(f"[DDP FutTrain] Dataset: {dataset_name}")
        print(f"[DDP FutTrain] Train path: {train_path}")
        print(f"[DDP FutTrain] Val path: {val_path}")

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
    val_sampler = DistributedEvalSampler(val_dataset, num_replicas=world_size, rank=rank)

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

    model = DiffusionFut(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_rmse = load_checkpoint_for_rank(args.resume_fut, checkpoint_dir, model, optimizer, scheduler, device, rank)

    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, rank)
        eval_summary = evaluate(
            model, val_loader, device, epoch + 1, args.feature_dim, rank, return_summary=True)
        eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps = validation_metric_values(eval_summary)
        selection_score = float(eval_rmse)
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process(rank):
            write_horizon_metrics(checkpoint_dir / "log" / "val_metrics_per_second.csv", epoch + 1, eval_summary, model.module.fut_dt if hasattr(model, "module") else model.fut_dt, writer=writer)
            write_csv_log(log_csv_path, epoch + 1, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, current_lr)
            write_tensorboard_log(writer, epoch + 1, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, current_lr)
            print_eval_summary(epoch + 1, args.num_epochs, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps)

        scheduler.step()
        is_best = selection_score < best_rmse
        if is_best:
            best_rmse = selection_score

        if is_main_process(rank):
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_stats["loss"],
                "eval_rmse_full_m": eval_rmse,
                "eval_min_ade_final_horizon_m": eval_ade,
                "eval_min_fde_final_horizon_m": eval_fde,
                "eval_rmse_final_horizon_m": eval_rmse_5s,
                "eval_theta_deg": eval_theta_deg,
                "eval_v_mps": eval_v_mps,
                "selection_score": selection_score,
                "selection_metric": "rmse_full_m",
                "eval_metrics": eval_summary,
                "best_score": best_rmse,
                "best_rmse_m": best_rmse,
            }

            if (epoch + 1) % args.save_interval == 0:
                torch.save(state, checkpoint_dir / f"epoch_{epoch + 1}.pth")
            if is_best:
                torch.save(state, checkpoint_dir / "best.pth")

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
