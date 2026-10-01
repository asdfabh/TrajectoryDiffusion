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
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_joint import (
    JOINT_FUT_CHECKPOINT_DIR,
    build_hist_outputs,
    get_joint_report_horizon,
    hist_checkpoint_dirs_for_dataset,
    init_csv_log,
    load_fut_checkpoint,
    load_hist_checkpoint,
    normalize_dataset_name,
    write_csv_log,
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


def reduce_tensor(tensor):
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


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


def load_hist_checkpoint_for_rank(model, resume_hist, checkpoint_dirs, device, rank, trainable=False, dataset_name=None):
    if is_main_process(rank):
        return load_hist_checkpoint(
            model,
            resume_hist,
            checkpoint_dirs,
            device,
            trainable=trainable,
            dataset_name=dataset_name,
        )
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_hist_checkpoint(
                model,
                resume_hist,
                checkpoint_dirs,
                device,
                trainable=trainable,
                dataset_name=dataset_name,
            )


def load_fut_checkpoint_for_rank(args, model, optimizer, scheduler, device, rank):
    if is_main_process(rank):
        return load_fut_checkpoint(args, model, optimizer, scheduler, device)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_fut_checkpoint(args, model, optimizer, scheduler, device)


def train_epoch(
    model_fut,
    model_hist,
    dataloader,
    optimizer,
    device,
    epoch,
    feature_dim,
    rank,
    mask_ratio,
    enable_latent_bridge,
):
    model_fut.train()
    model_hist.eval()

    total_loss = 0.0
    total_fut_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", ncols=140, disable=not is_main_process(rank))

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        pred_hist, past_latent_tokens = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            mask_ratio=mask_ratio,
            device=device,
            return_tokens=enable_latent_bridge,
        )
        fut_loss, _ = model_fut(pred_hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, past_latent_tokens=past_latent_tokens)
        loss = fut_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fut.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_fut_loss += float(fut_loss.item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "fut": f"{fut_loss.item():.6f}",
                "avg": f"{(total_loss / num_batches):.6f}",
            })

    stats = torch.tensor(
        [
            total_loss,
            total_fut_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    denom = max(int(stats[2].item()), 1)

    return {
        "loss": float(stats[0].item()) / denom,
        "loss_fut": float(stats[1].item()) / denom,
    }


@torch.no_grad()
def evaluate(
    model_fut,
    model_hist,
    dataloader,
    device,
    epoch,
    feature_dim,
    rank,
    mask_ratio,
    horizon_idx,
    horizon_label,
    enable_latent_bridge,
    return_summary=False,
):
    was_fut_training = model_fut.training
    was_hist_training = model_hist.training
    fut_model = model_fut.module if hasattr(model_fut, "module") else model_fut
    model_fut.eval()
    model_hist.eval()
    metrics = TrajectoryMetrics(fut_model.T, num_candidates=fut_model.fut_k)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True, disable=not is_main_process(rank))
    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        pred_hist, past_latent_tokens = build_hist_outputs(
            model_hist=model_hist, hist=hist, mask_ratio=mask_ratio,
            device=device, return_tokens=enable_latent_bridge,
        )
        all_preds = fut_model.forwardEvalMulti(
            pred_hist, hist_nbrs, mask, temporal_mask, device, K=fut_model.fut_k,
            past_latent_tokens=past_latent_tokens,
        )
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
    model_fut.train(was_fut_training)
    model_hist.train(was_hist_training)
    if return_summary:
        return summary
    full_rmse, ade, fde, horizon_rmse, theta, velocity = validation_metric_values(summary, horizon_idx)
    return full_rmse, horizon_rmse, ade, fde, theta, velocity


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    dataset_name = normalize_dataset_name(args.dataset)
    checkpoint_dir = JOINT_FUT_CHECKPOINT_DIR / dataset_name
    args.checkpoint_dir = str(checkpoint_dir)

    writer = None
    log_csv_path = None
    if is_main_process(rank):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir = checkpoint_dir / "log"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        log_csv_path = tensorboard_log_dir / "train_log_bestofk.csv"
        if not log_csv_path.exists() or args.resume_fut in ("none", "", None):
            init_csv_log(log_csv_path)
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))
    train_path = str(get_split_path(args, dataset_name, "Train"))
    val_path = str(get_split_path(args, dataset_name, "Val"))
    if is_main_process(rank):
        print(f"[DDP JointTrain] Dataset: {dataset_name}")
        print(f"[DDP JointTrain] Train path: {train_path}")
        print(f"[DDP JointTrain] Val path: {val_path}")

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
    val_sampler = DistributedEvalSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = build_distributed_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, drop_last=True)
    val_loader = build_distributed_loader(val_dataset, args.batch_size, args.num_workers, val_sampler, drop_last=False)

    model_hist = DiffusionPast(args).to(device)
    load_hist_checkpoint_for_rank(
        model_hist,
        args.resume_hist,
        hist_checkpoint_dirs_for_dataset(dataset_name),
        device,
        rank,
        trainable=False,
        dataset_name=dataset_name,
    )

    model_fut = DiffusionFut(args).to(device)
    fut_lr = float(args.learning_rate)
    optimizer = torch.optim.AdamW(model_fut.parameters(), lr=fut_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_rmse = load_fut_checkpoint_for_rank(args, model_fut, optimizer, scheduler, device, rank)

    if dist.is_initialized():
        if device.type == "cuda":
            model_fut = DDP(model_fut, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model_fut = DDP(model_fut, find_unused_parameters=False)

    mask_ratio = max(0.0, min(1.0, float(args.mask_prob)))
    report_horizon_s, report_horizon_idx, report_horizon_label = get_joint_report_horizon(dataset_name)
    enable_latent_bridge = int(args.enable_past_fut_latent_bridge) > 0

    if is_main_process(rank):
        print(
            f"[DDP JointTrain] hist=frozen | latent_bridge={int(enable_latent_bridge)} | "
            f"lr_fut={fut_lr:.2e}"
        )

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        train_stats = train_epoch(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
            rank=rank,
            mask_ratio=mask_ratio,
            enable_latent_bridge=enable_latent_bridge,
        )
        eval_summary = evaluate(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=val_loader,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
            rank=rank,
            mask_ratio=mask_ratio,
            horizon_idx=report_horizon_idx,
            horizon_label=report_horizon_label,
            enable_latent_bridge=enable_latent_bridge,
            return_summary=True,
        )
        eval_rmse, eval_ade, eval_fde, eval_horizon_rmse, eval_theta_deg, eval_v_mps = validation_metric_values(eval_summary, report_horizon_idx)

        if is_main_process(rank):
            current_lr_fut = float(optimizer.param_groups[0]["lr"])
            write_horizon_metrics(checkpoint_dir / "log" / "val_metrics_per_second.csv", epoch + 1, eval_summary, model_fut.module.fut_dt if hasattr(model_fut, "module") else model_fut.fut_dt, writer=writer)
            write_csv_log(
                log_csv_path,
                epoch + 1,
                train_stats,
                eval_rmse,
                report_horizon_s,
                eval_horizon_rmse,
                eval_ade,
                eval_fde,
                eval_theta_deg,
                eval_v_mps,
                current_lr_fut,
            )
            writer.add_scalar("Loss/Train", train_stats["loss"], epoch + 1)
            writer.add_scalar("Loss/TrainFut", train_stats["loss_fut"], epoch + 1)
            writer.add_scalar("Eval/RMSE_full_selection_m", eval_rmse, epoch + 1)
            writer.add_scalar(f"Eval/RMSE_{report_horizon_label}", eval_horizon_rmse, epoch + 1)
            writer.add_scalar("Eval/minADE_final_horizon_m", eval_ade, epoch + 1)
            writer.add_scalar("Eval/minFDE_final_horizon_m", eval_fde, epoch + 1)
            writer.add_scalar("Eval/Theta_deg", eval_theta_deg, epoch + 1)
            writer.add_scalar("Eval/V_mps", eval_v_mps, epoch + 1)
            writer.add_scalar("LR", current_lr_fut, epoch + 1)
            print(
                f"Epoch {epoch + 1}/{args.num_epochs} | "
                f"train={train_stats['loss']:.6f} | "
                f"fut={train_stats['loss_fut']:.6f} | "
                f"rmse_full_selection_m={eval_rmse:.4f} | "
                f"rmse_{report_horizon_label}={eval_horizon_rmse:.4f} | "
                f"minADE_final_horizon_m={eval_ade:.4f} | "
                f"minFDE_final_horizon_m={eval_fde:.4f} | "
                f"theta_deg={eval_theta_deg:.4f} | "
                f"v_mps={eval_v_mps:.4f}"
            )

        scheduler.step()
        selection_score = float(eval_rmse)
        is_best = selection_score < best_rmse
        if is_best:
            best_rmse = selection_score

        if is_main_process(rank):
            fut_state = {
                "epoch": epoch + 1,
                "model_state_dict": model_fut.module.state_dict() if hasattr(model_fut, "module") else model_fut.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_stats["loss"],
                "loss_fut": train_stats["loss_fut"],
                "eval_rmse_full_m": eval_rmse,
                "eval_horizon_s": report_horizon_s,
                "eval_horizon_rmse_m": eval_horizon_rmse,
                "eval_min_ade_final_horizon_m": eval_ade,
                "eval_min_fde_final_horizon_m": eval_fde,
                "eval_theta_deg": eval_theta_deg,
                "eval_v_mps": eval_v_mps,
                "selection_score": selection_score,
                "selection_metric": "rmse_full_m",
                "eval_metrics": eval_summary,
                "best_score": best_rmse,
                "best_rmse_m": best_rmse,
                "resume_hist": args.resume_hist,
                "enable_past_fut_latent_bridge": int(enable_latent_bridge),
            }

            if (epoch + 1) % args.save_interval == 0:
                torch.save(fut_state, checkpoint_dir / f"epoch_{epoch + 1}.pth")

            if is_best:
                torch.save(fut_state, checkpoint_dir / "best.pth")

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
