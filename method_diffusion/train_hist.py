import sys
import os
import re
import csv
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.utils.mask_util import random_mask, continuous_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HIST_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "hist"
TRAIN_BLOCK_MASK_RATIO = 0.3
VAL_MASK_TYPE = "random"


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
        "train_dxy",
        "train_v_cons",
        "train_a_cons",
        "val_loss",
        "val_ade_ft",
        "val_fde_ft",
        "val_ade_m",
        "val_fde_m",
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
        "train_dxy": train_stats["loss_dxy"],
        "train_v_cons": train_stats["loss_v_cons"],
        "train_a_cons": train_stats["loss_a_cons"],
        "val_loss": eval_stats["loss"],
        "val_ade_ft": eval_stats["ade_ft"],
        "val_fde_ft": eval_stats["fde_ft"],
        "val_ade_m": eval_stats["ade_ft"] * 0.3048,
        "val_fde_m": eval_stats["fde_ft"] * 0.3048,
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


def build_hist_mask(hist, mask_type, mask_prob):
    if mask_type == "random":
        return random_mask(hist, p=mask_prob)
    if mask_type == "block":
        return continuous_mask(hist, p=mask_prob)
    print(f"[HistModel] Unknown mask type '{mask_type}', fallback to random.")
    return random_mask(hist, p=mask_prob)


def build_masked_history(hist, mask_type, mask_prob):
    hist_mask = build_hist_mask(hist, mask_type=mask_type, mask_prob=mask_prob)
    hist_masked_value = hist_mask * hist
    hist_masked = torch.cat([hist_masked_value, hist_mask], dim=-1)
    return hist_masked, hist_mask


def sample_train_mask_type():
    return "block" if torch.rand(1).item() < TRAIN_BLOCK_MASK_RATIO else "random"


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, mask_prob):
    model.train()
    total_stats = {
        "loss_total": 0.0,
        "loss_xy_unknown": 0.0,
        "loss_xy_known": 0.0,
        "loss_va_unknown": 0.0,
        "loss_va_known": 0.0,
        "loss_dxy": 0.0,
        "loss_v_cons": 0.0,
        "loss_a_cons": 0.0,
    }
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", ncols=140)
    for batch in pbar:
        batch = filter_valid_batch(batch)
        hist = prepare_input_data(batch, feature_dim, device=device)
        if hist.shape[0] == 0:
            continue

        mask_type = sample_train_mask_type()
        hist_masked, _ = build_masked_history(hist, mask_type=mask_type, mask_prob=mask_prob)
        loss, _, ade, fde, loss_parts = model.forward_train(hist, hist_masked, device, return_components=True)
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
            "dxy": f"{(total_stats['loss_dxy'] / num_batches):.5f}",
            "ade": f"{ade.item():.4f}",
            "fde": f"{fde.item():.4f}",
        })

    denom = max(num_batches, 1)
    return {key: value / denom for key, value in total_stats.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, feature_dim, eval_ratio, mask_prob):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        model.train()
        return {"loss": 0.0, "ade_ft": 0.0, "fde_ft": 0.0}

    total_batches = len(dataloader)
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        target_batches = total_batches
    else:
        target_batches = max(1, int(math.ceil(total_batches * float(eval_ratio))))

    pbar = tqdm(dataloader, total=target_batches, desc=f"Ep{epoch} Val", ncols=140)
    for batch in pbar:
        if num_batches >= target_batches:
            break

        batch = filter_valid_batch(batch)
        hist = prepare_input_data(batch, feature_dim, device=device)
        if hist.shape[0] == 0:
            continue

        hist_masked, _ = build_masked_history(hist, mask_type=VAL_MASK_TYPE, mask_prob=mask_prob)
        loss, _, ade, fde = model.forward_eval(hist, hist_masked, device)

        total_loss += float(loss.item())
        total_ade += float(ade.item())
        total_fde += float(fde.item())
        num_batches += 1
        pbar.set_postfix({
            "avg_loss": f"{(total_loss / num_batches):.6f}",
            "avg_ade_ft": f"{(total_ade / num_batches):.4f}",
            "avg_fde_ft": f"{(total_fde / num_batches):.4f}",
        })

    model.train()
    if num_batches == 0:
        return {"loss": 0.0, "ade_ft": 0.0, "fde_ft": 0.0}

    return {
        "loss": total_loss / num_batches,
        "ade_ft": total_ade / num_batches,
        "fde_ft": total_fde / num_batches,
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

    data_root = Path(args.data_root)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")

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
    eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, args.mask_prob)
        eval_stats = evaluate(model, val_loader, device, epoch + 1, args.feature_dim, eval_ratio, args.mask_prob)
        current_lr = optimizer.param_groups[0]["lr"]

        write_csv_log(log_csv_path, epoch + 1, train_stats, eval_stats, current_lr)

        writer.add_scalar("Loss/Train", train_stats["loss_total"], epoch + 1)
        writer.add_scalar("Loss/TrainXYUnknown", train_stats["loss_xy_unknown"], epoch + 1)
        writer.add_scalar("Loss/TrainXYKnown", train_stats["loss_xy_known"], epoch + 1)
        writer.add_scalar("Loss/TrainVAUnknown", train_stats["loss_va_unknown"], epoch + 1)
        writer.add_scalar("Loss/TrainVAKnown", train_stats["loss_va_known"], epoch + 1)
        writer.add_scalar("Loss/TrainDXY", train_stats["loss_dxy"], epoch + 1)
        writer.add_scalar("Loss/TrainVConsistency", train_stats["loss_v_cons"], epoch + 1)
        writer.add_scalar("Loss/TrainAConsistency", train_stats["loss_a_cons"], epoch + 1)
        writer.add_scalar("Eval/Loss", eval_stats["loss"], epoch + 1)
        writer.add_scalar("Eval/ADE_ft", eval_stats["ade_ft"], epoch + 1)
        writer.add_scalar("Eval/FDE_ft", eval_stats["fde_ft"], epoch + 1)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train={train_stats['loss_total']:.6f} | "
            f"xy_u={train_stats['loss_xy_unknown']:.6f} | "
            f"xy_k={train_stats['loss_xy_known']:.6f} | "
            f"va_u={train_stats['loss_va_unknown']:.6f} | "
            f"va_k={train_stats['loss_va_known']:.6f} | "
            f"val_loss={eval_stats['loss']:.6f} | "
            f"ade={eval_stats['ade_ft']:.4f}ft | "
            f"fde={eval_stats['fde_ft']:.4f}ft"
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
            "eval_ade": eval_stats["ade_ft"],
            "eval_fde": eval_stats["fde_ft"],
            "best_loss": best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if is_best:
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

    writer.close()


if __name__ == "__main__":
    main()
