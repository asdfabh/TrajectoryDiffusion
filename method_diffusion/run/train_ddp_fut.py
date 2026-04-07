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
from method_diffusion.run.train_fut import (
    FUT_CHECKPOINT_DIR,
    LOSS_STAT_KEYS,
    METER_PER_FOOT,
    build_zero_loss_stats,
    compute_selection_score,
    init_csv_log,
    load_checkpoint,
    print_eval_summary,
    prepare_input_data,
    write_csv_log,
    write_tensorboard_log,
)
from method_diffusion.utils.fut_utils import compute_batch_ade_fde, select_minade_prediction


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
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        ncols=120,
        disable=not is_main_process(rank),
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
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

        totals["loss"] += float(loss.item())
        for key in LOSS_STAT_KEYS:
            if key == "loss":
                continue
            totals[key] += float(loss_parts[key].item())
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
def evaluate(model, dataloader, device, epoch, feature_dim, rank):
    fut_model = model.module if hasattr(model, "module") else model
    fut_model.eval()

    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        fut_model.train()
        return build_zero_loss_stats(), 0.0, 0.0

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
        if int(fut_model.fut_k) > 1:
            all_preds = fut_model.forwardEvalMulti(
                hist,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                device,
                K=fut_model.fut_k,
            )
            pred_fut, _, _ = select_minade_prediction(all_preds, fut, op_mask)
        else:
            pred_fut = fut_model.forwardEval(
                hist,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                device,
            )
        eval_ade, eval_fde = compute_batch_ade_fde(pred_fut, fut, op_mask)
        eval_ade = float(eval_ade.item()) * METER_PER_FOOT
        eval_fde = float(eval_fde.item()) * METER_PER_FOOT

        totals["loss"] += float(val_loss.item())
        for key in LOSS_STAT_KEYS:
            if key == "loss":
                continue
            totals[key] += float(val_parts[key].item())
        total_ade += eval_ade
        total_fde += eval_fde
        num_batches += 1

        if is_main_process(rank):
            pbar.set_postfix(
                {
                    "val_loss": f"{(totals['loss'] / num_batches):.6f}",
                    "avg_ade_m": f"{(total_ade / num_batches):.4f}",
                    "avg_fde_m": f"{(total_fde / num_batches):.4f}",
                }
            )

    stats = torch.tensor(
        [totals[key] for key in LOSS_STAT_KEYS] + [total_ade, total_fde, float(num_batches)],
        device=device,
        dtype=torch.float64,
    )
    stats = reduce_tensor(stats)
    fut_model.train()

    offset = len(LOSS_STAT_KEYS)
    denom = max(int(stats[offset + 2].item()), 1)
    val_stats = {
        key: float(stats[idx].item()) / denom
        for idx, key in enumerate(LOSS_STAT_KEYS)
    }
    val_stats["fut_k"] = int(getattr(fut_model, "fut_k", 0))
    return val_stats, float(stats[offset].item()) / denom, float(stats[offset + 1].item()) / denom


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

    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")
    if is_main_process(rank):
        print(f"[DDP FutTrain] Dataset: {dataset_name}")
        print(f"[DDP FutTrain] Train path: {train_path}")
        print(f"[DDP FutTrain] Val path: {val_path}")

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

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)

        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, rank)
        val_stats, eval_ade, eval_fde = evaluate(model, val_loader, device, epoch + 1, args.feature_dim, rank)
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
                "eval_ade_m": eval_ade,
                "eval_fde_m": eval_fde,
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
