import sys
import os
import re
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.utils.mask_util import mixed_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "hist"


def resolve_resume_checkpoint(resume_arg, checkpoint_dir):
    if resume_arg in ("none", "", None):
        return None
    if resume_arg == "latest":
        ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        return ckpts[-1] if ckpts else None
    if resume_arg == "best":
        best_path = checkpoint_dir / "checkpoint_best.pth"
        return best_path if best_path.exists() else None
    if re.fullmatch(r"epoch\d+", str(resume_arg)):
        epoch_num = int(str(resume_arg).replace("epoch", ""))
        ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch_num}.pth"
        return ckpt_path if ckpt_path.exists() else None

    direct_path = Path(resume_arg)
    if direct_path.exists():
        return direct_path

    print(f"[HistModel] Unsupported resume_hist='{resume_arg}', expected 'best', 'latest' or 'epochN'.")
    return None


def load_checkpoint(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_loss = float("inf")
    ckpt_path = resolve_resume_checkpoint(args.resume_hist, Path(args.checkpoint_dir))

    if ckpt_path is None:
        return start_epoch, best_loss

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)

    try:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
    except Exception:
        pass

    start_epoch = int(state.get("epoch", 0))
    best_loss = float(state.get("best_loss", state.get("loss", best_loss)))
    print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")
    return start_epoch, best_loss


def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_xy_unknown",
        "train_xy_known",
        "train_va_unknown",
        "train_va_known",
        "val_loss",
        "val_masked_ade_ft",
        "val_masked_ade_m",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_log(csv_path, epoch, train_stats, eval_stats, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss_total"],
        "train_xy_unknown": train_stats["loss_xy_unknown"],
        "train_xy_known": train_stats["loss_xy_known"],
        "train_va_unknown": train_stats["loss_va_unknown"],
        "train_va_known": train_stats["loss_va_known"],
        "val_loss": eval_stats["loss"],
        "val_masked_ade_ft": eval_stats["masked_ade_ft"],
        "val_masked_ade_m": eval_stats["masked_ade_ft"] * 0.3048,
        "lr": lr,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def filter_valid_batch(batch):
    sample_valid = batch.get("sample_valid", None)
    if sample_valid is None:
        return batch

    sample_valid = sample_valid.bool()
    if sample_valid.numel() == 0 or bool(sample_valid.all()):
        return batch

    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == sample_valid.shape[0]:
            filtered[key] = value[sample_valid]
        else:
            filtered[key] = value
    return filtered


def prepare_input_data(batch, feature_dim, device="cuda"):
    hist = batch["hist"]
    va = batch["va"]
    lane = batch["lane"]
    cclass = batch["cclass"]

    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
    else:
        hist = hist.to(device)

    return hist


def build_masked_history(hist, mask_ratio, random_mask_ratio, block_mask_start):
    hist_mask = mixed_mask(
        hist,
        p=mask_ratio,
        random_ratio=random_mask_ratio,
        block_start=block_mask_start,
    )
    hist_masked_value = hist_mask * hist
    hist_masked = torch.cat([hist_masked_value, hist_mask], dim=-1)
    return hist_masked, hist_mask


def compute_masked_xy_ade(pred, target, hist_mask):
    masked = hist_mask[..., 0] < 0.5
    if not bool(masked.any()):
        return pred.new_tensor(0.0)
    xy_dist = torch.norm(pred[..., :2] - target[..., :2], dim=-1)
    return xy_dist[masked].mean()


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, mask_ratio, random_mask_ratio, block_mask_start):
    model.train()
    total_stats = {
        "loss_total": 0.0,
        "loss_xy_unknown": 0.0,
        "loss_xy_known": 0.0,
        "loss_va_unknown": 0.0,
        "loss_va_known": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", dynamic_ncols=True)
    for batch in pbar:
        batch = filter_valid_batch(batch)
        hist = prepare_input_data(batch, feature_dim, device=device)
        if hist.shape[0] == 0:
            continue

        hist_masked, _ = build_masked_history(
            hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )
        loss, pred, loss_parts = model.forward_train(hist, hist_masked, device, return_components=True)
        masked_ade = compute_masked_xy_ade(pred, hist, hist_masked[..., -1:])
        # _, _, _, _ = model.forward_train(hist, hist_masked, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key in total_stats:
            total_stats[key] += float(loss_parts[key].item())
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg_loss": f"{(total_stats['loss_total'] / num_batches):.6f}",
            "xy_u": f"{(total_stats['loss_xy_unknown'] / num_batches):.5f}",
            "va_u": f"{(total_stats['loss_va_unknown'] / num_batches):.5f}",
            "mask_ade": f"{masked_ade.item():.4f}",
        })

    denom = max(num_batches, 1)
    return {key: value / denom for key, value in total_stats.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, feature_dim, mask_ratio, random_mask_ratio, block_mask_start):
    model.eval()
    total_loss = 0.0
    total_masked_ade = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        model.train()
        return {"loss": 0.0, "masked_ade_ft": 0.0}

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True)
    for batch in pbar:
        batch = filter_valid_batch(batch)
        hist = prepare_input_data(batch, feature_dim, device=device)
        if hist.shape[0] == 0:
            continue

        hist_masked, hist_mask = build_masked_history(
            hist,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )
        loss, pred = model.forward_eval(hist, hist_masked, device)
        masked_ade = compute_masked_xy_ade(pred, hist, hist_mask)

        total_loss += float(loss.item())
        total_masked_ade += float(masked_ade.item())
        num_batches += 1
        pbar.set_postfix({
            "avg_loss": f"{(total_loss / num_batches):.6f}",
            "avg_mask_ade_ft": f"{(total_masked_ade / num_batches):.4f}",
        })

    model.train()
    if num_batches == 0:
        return {"loss": 0.0, "masked_ade_ft": 0.0}

    return {
        "loss": total_loss / num_batches,
        "masked_ade_ft": total_masked_ade / num_batches,
    }


def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(HIST_CHECKPOINT_DIR)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(args.checkpoint_dir) / "log"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = tensorboard_log_dir / "train_log.csv"
    if not log_csv_path.exists() or args.resume_hist in ("none", "", None):
        init_csv_log(log_csv_path)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")
    print(f"[HistTrain] Dataset: {dataset_name}")
    print(f"[HistTrain] Train path: {train_path}")
    print(f"[HistTrain] Val path: {val_path}")

    train_dataset = NgsimHistDataset(train_path, t_h=30, d_s=2)
    val_dataset = NgsimHistDataset(val_path, t_h=30, d_s=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = DiffusionPast(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    start_epoch, best_loss = load_checkpoint(args, model, optimizer, scheduler, device)
    mask_ratio = max(0.0, min(1.0, float(args.mask_prob)))
    random_mask_ratio = max(0.0, min(1.0, float(args.random_mask_ratio)))
    block_mask_start = int(args.block_mask_start) > 0

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch + 1,
            args.feature_dim,
            mask_ratio,
            random_mask_ratio,
            block_mask_start,
        )
        eval_stats = evaluate(
            model,
            val_loader,
            device,
            epoch + 1,
            args.feature_dim,
            mask_ratio,
            random_mask_ratio,
            block_mask_start,
        )
        current_lr = optimizer.param_groups[0]["lr"]

        write_csv_log(log_csv_path, epoch + 1, train_stats, eval_stats, current_lr)

        writer.add_scalar("Loss/Train", train_stats["loss_total"], epoch + 1)
        writer.add_scalar("Loss/TrainXYUnknown", train_stats["loss_xy_unknown"], epoch + 1)
        writer.add_scalar("Loss/TrainXYKnown", train_stats["loss_xy_known"], epoch + 1)
        writer.add_scalar("Loss/TrainVAUnknown", train_stats["loss_va_unknown"], epoch + 1)
        writer.add_scalar("Loss/TrainVAKnown", train_stats["loss_va_known"], epoch + 1)
        writer.add_scalar("Eval/Loss", eval_stats["loss"], epoch + 1)
        writer.add_scalar("Eval/MaskedADE_ft", eval_stats["masked_ade_ft"], epoch + 1)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train={train_stats['loss_total']:.6f} | "
            f"xy_u={train_stats['loss_xy_unknown']:.6f} | "
            f"xy_k={train_stats['loss_xy_known']:.6f} | "
            f"va_u={train_stats['loss_va_unknown']:.6f} | "
            f"va_k={train_stats['loss_va_known']:.6f} | "
            f"mask_ratio={mask_ratio:.2f} | "
            f"rand_ratio={random_mask_ratio:.2f} | "
            f"val_loss={eval_stats['loss']:.6f} | "
            f"mask_ade={eval_stats['masked_ade_ft']:.4f}ft"
        )

        scheduler.step()
        is_best = eval_stats["loss"] < best_loss
        if is_best:
            best_loss = eval_stats["loss"]

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_stats["loss_total"],
            "eval_loss": eval_stats["loss"],
            "eval_masked_ade": eval_stats["masked_ade_ft"],
            "best_loss": best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if is_best:
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

    writer.close()


if __name__ == "__main__":
    main()
