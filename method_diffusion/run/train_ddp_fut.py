import contextlib
import math
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
from method_diffusion.run.train_fut import (
    FUT_CHECKPOINT_DIR,
    build_zero_loss_stats,
    compute_selection_score,
    init_csv_log,
    load_checkpoint,
    print_eval_summary,
    prepare_input_data,
    write_csv_log,
    write_tensorboard_log,
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


def load_checkpoint_for_rank(args, model, optimizer, scheduler, device, rank):
    if is_main_process(rank):
        return load_checkpoint(args, model, optimizer, scheduler, device)

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_checkpoint(args, model, optimizer, scheduler, device)


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
    total_loss = 0.0
    total_vel_loss = 0.0
    total_pos_loss = 0.0
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        ncols=120,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_parts = model(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_components=True,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_vel_loss += float(loss_parts["loss_vel"].item())
        total_pos_loss += float(loss_parts["loss_pos"].item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.6f}",
                    "avg_loss": f"{(total_loss / num_batches):.6f}",
                    "vel_loss": f"{(total_vel_loss / num_batches):.6f}",
                    "pos_loss": f"{(total_pos_loss / num_batches):.6f}",
                }
            )

    stats = torch.tensor(
        [total_loss, total_vel_loss, total_pos_loss, float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    denom = max(int(stats[3].item()), 1)

    return {
        "loss": float(stats[0].item()) / denom,
        "loss_vel": float(stats[1].item()) / denom,
        "loss_pos": float(stats[2].item()) / denom,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, feature_dim, eval_ratio, rank):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    fut_model = model.module if hasattr(model, "module") else model
    fut_model.eval()

    total_loss = 0.0
    total_vel_loss = 0.0
    total_pos_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        fut_model.train()
        return build_zero_loss_stats(), 0.0, 0.0

    total_batches = len(dataloader)
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        target_batches = total_batches
    else:
        target_batches = max(1, int(math.ceil(total_batches * float(eval_ratio))))

    pbar = tqdm(
        dataloader,
        total=target_batches,
        desc=f"Ep{epoch} Val",
        ncols=120,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        if num_batches >= target_batches:
            break

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        val_loss, val_parts = fut_model.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_components=True,
        )
        _, eval_ade, eval_fde = fut_model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        total_loss += float(val_loss.item())
        total_vel_loss += float(val_parts["loss_vel"].item())
        total_pos_loss += float(val_parts["loss_pos"].item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix(
                {
                    "val_loss": f"{(total_loss / num_batches):.6f}",
                    "val_vel": f"{(total_vel_loss / num_batches):.6f}",
                    "val_pos": f"{(total_pos_loss / num_batches):.6f}",
                    "avg_ade_ft": f"{(total_ade / num_batches):.4f}",
                    "avg_fde_ft": f"{(total_fde / num_batches):.4f}",
                }
            )

    stats = torch.tensor(
        [total_loss, total_vel_loss, total_pos_loss, total_ade, total_fde, float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    fut_model.train()

    denom = max(int(stats[5].item()), 1)
    val_stats = {
        "loss": float(stats[0].item()) / denom,
        "loss_vel": float(stats[1].item()) / denom,
        "loss_pos": float(stats[2].item()) / denom,
    }
    return val_stats, float(stats[3].item()) / denom, float(stats[4].item()) / denom


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR)

    writer = None
    log_csv_path = None
    if is_main_process(rank):
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir = Path(args.checkpoint_dir) / "log"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        log_csv_path = tensorboard_log_dir / "train_log.csv"
        init_csv_log(log_csv_path)
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    data_root = Path(args.data_root)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")

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

    model = DiffusionFut(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_score = load_checkpoint_for_rank(args, model, optimizer, scheduler, device, rank)

    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)

    eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, rank)
        val_stats, eval_ade, eval_fde = evaluate(model, val_loader, device, epoch + 1, args.feature_dim, eval_ratio, rank)
        selection_score = compute_selection_score(eval_ade, eval_fde)
        current_lr = optimizer.param_groups[0]["lr"]

        if is_main_process(rank):
            write_csv_log(log_csv_path, epoch + 1, train_stats, val_stats, eval_ade, eval_fde, selection_score, current_lr)
            write_tensorboard_log(writer, epoch + 1, train_stats, val_stats, eval_ade, eval_fde, selection_score, current_lr)
            print_eval_summary(epoch + 1, args.num_epochs, train_stats, val_stats, eval_ade, eval_fde, selection_score)

        scheduler.step()
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score

        if is_main_process(rank):
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_stats["loss"],
                "eval_ade": eval_ade,
                "eval_fde": eval_fde,
                "selection_score": selection_score,
                "best_score": best_score,
            }

            if (epoch + 1) % args.save_interval == 0:
                torch.save(state, Path(args.checkpoint_dir) / f"epoch_{epoch + 1}.pth")
            if is_best:
                torch.save(state, Path(args.checkpoint_dir) / "best.pth")

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
