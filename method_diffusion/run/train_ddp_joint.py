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
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_joint import (
    HIST_CHECKPOINT_DIR,
    JOINT_FUT_CHECKPOINT_DIR,
    JOINT_HIST_CHECKPOINT_DIR,
    METER_PER_FOOT,
    build_hist_outputs,
    init_csv_log,
    load_fut_checkpoint,
    load_hist_checkpoint,
    write_csv_log,
)
from method_diffusion.utils.fut_utils import compute_batch_kinematic_metrics, compute_batch_metric, select_closest_prediction


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


def load_hist_checkpoint_for_rank(model, resume_hist, checkpoint_dirs, device, rank):
    if is_main_process(rank):
        return load_hist_checkpoint(model, resume_hist, checkpoint_dirs, device)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_hist_checkpoint(model, resume_hist, checkpoint_dirs, device)


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
    random_mask_ratio,
    block_mask_start,
):
    model_fut.train()
    model_hist.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", ncols=140, disable=not is_main_process(rank))

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        pred_hist = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
            device=device,
        )
        loss, _ = model_fut(pred_hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fut.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "avg": f"{(total_loss / num_batches):.6f}",
            })

    stats = torch.tensor(
        [
            total_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    denom = max(int(stats[1].item()), 1)

    return {"loss": float(stats[0].item()) / denom}


@torch.no_grad()
def evaluate(model_fut, model_hist, dataloader, device, epoch, feature_dim, rank, mask_ratio, random_mask_ratio, block_mask_start):
    was_fut_training = model_fut.training
    was_hist_training = model_hist.training
    fut_model = model_fut.module if hasattr(model_fut, "module") else model_fut
    model_fut.eval()
    model_hist.eval()

    total_rmse = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_theta_deg = 0.0
    total_v_mps = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        if was_fut_training:
            model_fut.train()
        if was_hist_training:
            model_hist.train()
        return 0.0, 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Val",
        ncols=120,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        pred_hist = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
            device=device,
        )
        if int(fut_model.fut_k) > 1:
            all_preds = fut_model.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, fut, device, K=fut_model.fut_k)
            pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        else:
            all_preds = fut_model.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, fut, device, K=1)
            pred_fut = all_preds.squeeze(1)
        eval_rmse, eval_ade, eval_fde = compute_batch_metric(pred_fut, fut, op_mask)
        eval_theta_deg, eval_v_mps = compute_batch_kinematic_metrics(pred_fut, fut, op_mask, meter_per_unit=METER_PER_FOOT)
        eval_rmse = float(eval_rmse.item()) * METER_PER_FOOT
        eval_ade = float(eval_ade.item()) * METER_PER_FOOT
        eval_fde = float(eval_fde.item()) * METER_PER_FOOT
        eval_theta_deg = float(eval_theta_deg.item())
        eval_v_mps = float(eval_v_mps.item())

        total_rmse += eval_rmse
        total_ade += eval_ade
        total_fde += eval_fde
        total_theta_deg += eval_theta_deg
        total_v_mps += eval_v_mps
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({
                "avg_rmse_m": f"{(total_rmse / num_batches):.4f}",
                "avg_ade_m": f"{(total_ade / num_batches):.4f}",
                "avg_fde_m": f"{(total_fde / num_batches):.4f}",
                "avg_theta_deg": f"{(total_theta_deg / num_batches):.4f}",
                "avg_v_mps": f"{(total_v_mps / num_batches):.4f}",
            })

    stats = torch.tensor(
        [total_rmse, total_ade, total_fde, total_theta_deg, total_v_mps, float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    if was_fut_training:
        model_fut.train()
    if was_hist_training:
        model_hist.train()
    denom = max(int(stats[5].item()), 1)
    return (
        float(stats[0].item()) / denom,
        float(stats[1].item()) / denom,
        float(stats[2].item()) / denom,
        float(stats[3].item()) / denom,
        float(stats[4].item()) / denom,
    )


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(JOINT_FUT_CHECKPOINT_DIR)

    writer = None
    log_csv_path = None
    if is_main_process(rank):
        JOINT_FUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir = JOINT_FUT_CHECKPOINT_DIR / "log"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        log_csv_path = tensorboard_log_dir / "train_log.csv"
        if not log_csv_path.exists() or args.resume_fut in ("none", "", None):
            init_csv_log(log_csv_path)
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")
    if is_main_process(rank):
        print(f"[DDP JointTrain] Dataset: {dataset_name}")
        print(f"[DDP JointTrain] Train path: {train_path}")
        print(f"[DDP JointTrain] Val path: {val_path}")

    train_dataset = NgsimDataset(
        train_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    val_dataset = NgsimDataset(
        val_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = build_distributed_loader(train_dataset, args.batch_size, args.num_workers, train_sampler, drop_last=True)
    val_loader = build_distributed_loader(val_dataset, args.batch_size, args.num_workers, val_sampler, drop_last=False)

    model_hist = DiffusionPast(args).to(device)
    load_hist_checkpoint_for_rank(
        model_hist,
        args.resume_hist,
        [HIST_CHECKPOINT_DIR, JOINT_HIST_CHECKPOINT_DIR],
        device,
        rank,
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
    random_mask_ratio = max(0.0, min(1.0, float(args.random_mask_ratio)))
    block_mask_start = int(args.block_mask_start) > 0

    if is_main_process(rank):
        print(
            f"[DDP JointTrain] hist_frozen=1 | lr_fut={fut_lr:.2e}"
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
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )
        eval_rmse, eval_ade, eval_fde, eval_theta_deg, eval_v_mps = evaluate(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=val_loader,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
            rank=rank,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )

        if is_main_process(rank):
            current_lr_fut = float(optimizer.param_groups[0]["lr"])
            write_csv_log(log_csv_path, epoch + 1, train_stats, eval_rmse, eval_ade, eval_fde, eval_theta_deg, eval_v_mps, current_lr_fut)
            writer.add_scalar("Loss/Train", train_stats["loss"], epoch + 1)
            writer.add_scalar("Eval/RMSE_m", eval_rmse, epoch + 1)
            writer.add_scalar("Eval/ADE_m", eval_ade, epoch + 1)
            writer.add_scalar("Eval/FDE_m", eval_fde, epoch + 1)
            writer.add_scalar("Eval/Theta_deg", eval_theta_deg, epoch + 1)
            writer.add_scalar("Eval/V_mps", eval_v_mps, epoch + 1)
            writer.add_scalar("LR", current_lr_fut, epoch + 1)
            print(
                f"Epoch {epoch + 1}/{args.num_epochs} | "
                f"train={train_stats['loss']:.6f} | "
                f"rmse_m={eval_rmse:.4f} | "
                f"ade_m={eval_ade:.4f} | "
                f"fde_m={eval_fde:.4f} | "
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
                "eval_rmse_m": eval_rmse,
                "eval_ade_m": eval_ade,
                "eval_fde_m": eval_fde,
                "eval_theta_deg": eval_theta_deg,
                "eval_v_mps": eval_v_mps,
                "selection_score": selection_score,
                "best_score": best_rmse,
                "best_rmse_m": best_rmse,
                "resume_hist": args.resume_hist,
            }

            if (epoch + 1) % args.save_interval == 0:
                torch.save(fut_state, JOINT_FUT_CHECKPOINT_DIR / f"epoch_{epoch + 1}.pth")

            if is_best:
                torch.save(fut_state, JOINT_FUT_CHECKPOINT_DIR / "best.pth")

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
