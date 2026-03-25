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
from method_diffusion.run.train_joint import (
    HIST_CHECKPOINT_DIR,
    JOINT_FUT_CHECKPOINT_DIR,
    JOINT_HIST_CHECKPOINT_DIR,
    build_hist_outputs,
    init_csv_log,
    load_fut_checkpoint,
    load_hist_checkpoint,
    prepare_input_data,
    write_csv_log,
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


def load_hist_checkpoint_for_rank(model, resume_hist, checkpoint_dirs, device, freeze_hist, rank):
    if is_main_process(rank):
        return load_hist_checkpoint(model, resume_hist, checkpoint_dirs, device, freeze_hist)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            return load_hist_checkpoint(model, resume_hist, checkpoint_dirs, device, freeze_hist)


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
    freeze_hist,
    hist_loss_weight,
    detach_hist_for_fut,
):
    model_fut.train()
    if freeze_hist:
        model_hist.eval()
    else:
        model_hist.train()

    total_loss = 0.0
    total_hist_loss = 0.0
    total_hist_loss_weighted = 0.0
    total_fut_loss = 0.0
    total_vel_loss = 0.0
    total_pos_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        ncols=140,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        loss_hist, hist_for_fut = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
            device=device,
            freeze_hist=freeze_hist,
            detach_hist_for_fut=detach_hist_for_fut,
        )
        loss_fut, fut_parts = model_fut(
            hist_for_fut,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_components=True,
        )

        loss_hist_weighted = hist_loss_weight * loss_hist
        loss = loss_fut + loss_hist_weighted

        optimizer.zero_grad()
        loss.backward()

        params = list(model_fut.parameters())
        if not freeze_hist:
            params += list(model_hist.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_hist_loss += float(loss_hist.item())
        total_hist_loss_weighted += float(loss_hist_weighted.item())
        total_fut_loss += float(loss_fut.item())
        total_vel_loss += float(fut_parts["loss_vel"].item())
        total_pos_loss += float(fut_parts["loss_pos"].item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "avg": f"{(total_loss / num_batches):.6f}",
                "hist": f"{(total_hist_loss / num_batches):.6f}",
                "fut": f"{(total_fut_loss / num_batches):.6f}",
                "vel": f"{(total_vel_loss / num_batches):.6f}",
                "pos": f"{(total_pos_loss / num_batches):.6f}",
            })

    stats = torch.tensor(
        [
            total_loss,
            total_hist_loss,
            total_hist_loss_weighted,
            total_fut_loss,
            total_vel_loss,
            total_pos_loss,
            float(num_batches),
        ],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    denom = max(int(stats[6].item()), 1)

    return {
        "loss": float(stats[0].item()) / denom,
        "loss_hist": float(stats[1].item()) / denom,
        "loss_hist_weighted": float(stats[2].item()) / denom,
        "loss_fut": float(stats[3].item()) / denom,
        "loss_vel": float(stats[4].item()) / denom,
        "loss_pos": float(stats[5].item()) / denom,
    }


@torch.no_grad()
def evaluate(model_fut, model_hist, dataloader, device, epoch, feature_dim, eval_ratio, rank, mask_ratio, random_mask_ratio, block_mask_start):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    was_fut_training = model_fut.training
    was_hist_training = model_hist.training
    fut_model = model_fut.module if hasattr(model_fut, "module") else model_fut
    model_fut.eval()
    model_hist.eval()

    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        if was_fut_training:
            model_fut.train()
        if was_hist_training:
            model_hist.train()
        return 0.0, 0.0

    total_batches = len(dataloader)
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        target_batches = total_batches
    else:
        target_batches = max(1, int(total_batches * float(eval_ratio) + 0.999999))

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

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        _, hist_for_fut = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
            device=device,
            freeze_hist=True,
            detach_hist_for_fut=True,
        )
        _, eval_ade, eval_fde = fut_model.forwardEval(
            hist_for_fut,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
        )

        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix({
                "avg_ade_ft": f"{(total_ade / num_batches):.4f}",
                "avg_fde_ft": f"{(total_fde / num_batches):.4f}",
            })

    stats = torch.tensor(
        [total_ade, total_fde, float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    if was_fut_training:
        model_fut.train()
    if was_hist_training:
        model_hist.train()
    denom = max(int(stats[2].item()), 1)
    return float(stats[0].item()) / denom, float(stats[1].item()) / denom


def main():
    rank, local_rank, world_size, device = setup_ddp()
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(JOINT_FUT_CHECKPOINT_DIR)

    writer = None
    log_csv_path = None
    if is_main_process(rank):
        JOINT_FUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        JOINT_HIST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        tensorboard_log_dir = JOINT_FUT_CHECKPOINT_DIR / "log"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        log_csv_path = tensorboard_log_dir / "train_log.csv"
        if not log_csv_path.exists() or args.resume_fut in ("none", "", None):
            init_csv_log(log_csv_path)
        writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    freeze_hist = int(args.joint_freeze_hist) > 0
    detach_hist_for_fut = int(args.joint_detach_hist_for_fut) > 0
    hist_loss_weight = 0.0 if freeze_hist else max(0.0, float(args.joint_hist_loss_weight))
    hist_lr_scale = max(0.0, float(args.joint_hist_lr_scale))

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
        freeze_hist,
        rank,
    )
    if dist.is_initialized() and not freeze_hist:
        if device.type == "cuda":
            model_hist = DDP(model_hist, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model_hist = DDP(model_hist, find_unused_parameters=False)

    model_fut = DiffusionFut(args).to(device)
    fut_lr = float(args.learning_rate)
    hist_lr = 0.0 if freeze_hist else fut_lr * hist_lr_scale
    param_groups = [{"params": model_fut.parameters(), "lr": fut_lr}]
    if not freeze_hist:
        param_groups.append({"params": model_hist.parameters(), "lr": hist_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_ade = load_fut_checkpoint_for_rank(args, model_fut, optimizer, scheduler, device, rank)

    if dist.is_initialized():
        if device.type == "cuda":
            model_fut = DDP(model_fut, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model_fut = DDP(model_fut, find_unused_parameters=False)

    eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))
    mask_ratio = max(0.0, min(1.0, float(args.mask_prob)))
    random_mask_ratio = max(0.0, min(1.0, float(args.random_mask_ratio)))
    block_mask_start = int(args.block_mask_start) > 0

    if is_main_process(rank):
        print(
            f"[DDP JointTrain] freeze_hist={freeze_hist} | "
            f"detach_hist_for_fut={detach_hist_for_fut} | "
            f"hist_loss_weight={hist_loss_weight:.4f} | "
            f"lr_fut={fut_lr:.2e} | lr_hist={hist_lr:.2e}"
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
            freeze_hist=freeze_hist,
            hist_loss_weight=hist_loss_weight,
            detach_hist_for_fut=detach_hist_for_fut,
        )
        eval_ade, eval_fde = evaluate(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=val_loader,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
            eval_ratio=eval_ratio,
            rank=rank,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )

        if is_main_process(rank):
            write_csv_log(log_csv_path, epoch + 1, train_stats, eval_ade, eval_fde, fut_lr, hist_lr)
            writer.add_scalar("Loss/Train", train_stats["loss"], epoch + 1)
            writer.add_scalar("Loss/TrainHist", train_stats["loss_hist"], epoch + 1)
            writer.add_scalar("Loss/TrainHistWeighted", train_stats["loss_hist_weighted"], epoch + 1)
            writer.add_scalar("Loss/TrainFut", train_stats["loss_fut"], epoch + 1)
            writer.add_scalar("Loss/TrainVel", train_stats["loss_vel"], epoch + 1)
            writer.add_scalar("Loss/TrainPos", train_stats["loss_pos"], epoch + 1)
            writer.add_scalar("Eval/ADE_ft", eval_ade, epoch + 1)
            writer.add_scalar("Eval/FDE_ft", eval_fde, epoch + 1)
            print(
                f"Epoch {epoch + 1}/{args.num_epochs} | "
                f"train={train_stats['loss']:.6f} | "
                f"hist={train_stats['loss_hist']:.6f} | "
                f"fut={train_stats['loss_fut']:.6f} | "
                f"ade={eval_ade:.4f}ft | "
                f"fde={eval_fde:.4f}ft"
            )

        scheduler.step()
        is_best = eval_ade < best_ade
        if is_best:
            best_ade = eval_ade

        if is_main_process(rank):
            fut_state = {
                "epoch": epoch + 1,
                "model_state_dict": model_fut.module.state_dict() if hasattr(model_fut, "module") else model_fut.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": train_stats["loss"],
                "eval_ade": eval_ade,
                "eval_fde": eval_fde,
                "best_ade": best_ade,
                "joint_freeze_hist": int(freeze_hist),
                "joint_hist_loss_weight": hist_loss_weight,
                "joint_detach_hist_for_fut": int(detach_hist_for_fut),
                "joint_hist_lr_scale": hist_lr_scale,
                "resume_hist": args.resume_hist,
            }

            if (epoch + 1) % args.save_interval == 0:
                torch.save(fut_state, JOINT_FUT_CHECKPOINT_DIR / f"epoch_{epoch + 1}.pth")
                if not freeze_hist:
                    hist_state = model_hist.module.state_dict() if hasattr(model_hist, "module") else model_hist.state_dict()
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": hist_state,
                            "loss_hist": train_stats["loss_hist"],
                        },
                        JOINT_HIST_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pth",
                    )

            if is_best:
                torch.save(fut_state, JOINT_FUT_CHECKPOINT_DIR / "best.pth")
                if not freeze_hist:
                    hist_state = model_hist.module.state_dict() if hasattr(model_hist, "module") else model_hist.state_dict()
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": hist_state,
                            "loss_hist": train_stats["loss_hist"],
                        },
                        JOINT_HIST_CHECKPOINT_DIR / "checkpoint_best.pth",
                    )

        if dist.is_initialized():
            dist.barrier()

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == "__main__":
    main()
