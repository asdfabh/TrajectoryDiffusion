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
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.utils.fut_utils import compute_batch_ade_fde, select_minade_prediction
from method_diffusion.utils.mask_util import mixed_mask

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "hist"
JOINT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "joint"
JOINT_FUT_CHECKPOINT_DIR = JOINT_CHECKPOINT_DIR / "fut"
JOINT_HIST_CHECKPOINT_DIR = JOINT_CHECKPOINT_DIR / "hist"
METER_PER_FOOT = 0.3048

def build_hist_masked(hist, mask_ratio, random_mask_ratio, block_mask_start):
    hist_mask = mixed_mask(
        hist,
        p=mask_ratio,
        random_ratio=random_mask_ratio,
        block_start=block_mask_start,
    ).to(hist.device)
    hist_masked_val = hist_mask * hist
    return torch.cat([hist_masked_val, hist_mask], dim=-1)


def resolve_fut_checkpoint(resume_arg, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg in ("none", "", None):
        return None
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if re.fullmatch(r"epoch_\d+", str(resume_arg)):
        return checkpoint_dir / f"{resume_arg}.pth"
    resume_path = Path(str(resume_arg))
    if resume_path.exists():
        return resume_path
    print(f"[JointFut] Unsupported resume_fut='{resume_arg}', expected 'best' or 'epoch_i'.")
    return None


def load_fut_checkpoint(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_ade = float("inf")
    ckpt_path = resolve_fut_checkpoint(args.resume_fut, Path(args.checkpoint_dir))
    if ckpt_path is None:
        return start_epoch, best_ade
    if not ckpt_path.exists():
        print(f"[JointFut] Checkpoint not found: {ckpt_path}")
        return start_epoch, best_ade

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
    except Exception:
        pass

    start_epoch = int(state.get("epoch", 0))
    best_ade = float(state.get("best_ade", state.get("best_loss", best_ade)))
    print(f"[JointFut] Resumed from {ckpt_path} @ epoch {start_epoch}")
    return start_epoch, best_ade


def resolve_hist_checkpoint(resume_arg, checkpoint_dirs):
    if resume_arg in ("none", "", None):
        resume_arg = "best"

    resume_path = Path(str(resume_arg))
    if resume_path.exists():
        return resume_path

    checkpoint_dirs = [Path(p) for p in checkpoint_dirs]
    if resume_arg == "best":
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / "checkpoint_best.pth"
            if candidate.exists():
                return candidate
    if resume_arg == "latest":
        for checkpoint_dir in checkpoint_dirs:
            ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            if ckpts:
                return ckpts[-1]
    if re.fullmatch(r"epoch\d+", str(resume_arg)):
        epoch_num = int(str(resume_arg).replace("epoch", ""))
        for checkpoint_dir in checkpoint_dirs:
            candidate = checkpoint_dir / f"checkpoint_epoch_{epoch_num}.pth"
            if candidate.exists():
                return candidate
    return None


def load_hist_checkpoint(model, resume_arg, checkpoint_dirs, device, freeze_hist):
    ckpt_path = resolve_hist_checkpoint(resume_arg, checkpoint_dirs)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"[JointHist] Checkpoint not found: resume_hist={resume_arg}, dirs={checkpoint_dirs}")

    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    cleaned_state = {}
    for key, value in state_dict.items():
        if key in ["pos_mean", "pos_std", "va_mean", "va_std"]:
            continue
        cleaned_state[key.replace("module.", "")] = value
    model.load_state_dict(cleaned_state, strict=False)

    if freeze_hist:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    else:
        model.train()
        for param in model.parameters():
            param.requires_grad = True

    print(f"[JointHist] Loaded checkpoint: {ckpt_path}")
    return model


def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_hist_loss",
        "train_hist_loss_weighted",
        "train_fut_loss",
        "val_ade_m",
        "val_fde_m",
        "lr_fut",
        "lr_hist",
        "fut_k",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_log(csv_path, epoch, train_stats, eval_ade, eval_fde, lr_fut, lr_hist):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_hist_loss": train_stats["loss_hist"],
        "train_hist_loss_weighted": train_stats["loss_hist_weighted"],
        "train_fut_loss": train_stats["loss_fut"],
        "val_ade_m": eval_ade,
        "val_fde_m": eval_fde,
        "lr_fut": lr_fut,
        "lr_hist": lr_hist,
        "fut_k": train_stats.get("fut_k", 0),
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def build_hist_outputs(model_hist, hist, mask_ratio, random_mask_ratio, block_mask_start, device, freeze_hist, detach_hist_for_fut):
    hist_model = model_hist.module if hasattr(model_hist, "module") else model_hist
    hist_masked = build_hist_masked(
        hist,
        mask_ratio=mask_ratio,
        random_mask_ratio=random_mask_ratio,
        block_mask_start=block_mask_start,
    )

    if freeze_hist:
        with torch.no_grad():
            _, pred_hist_eval = hist_model.forward_eval(hist, hist_masked, device)
        zero = hist.new_tensor(0.0)
        return zero, pred_hist_eval

    loss_hist, pred_hist_train, _ = model_hist(hist, hist_masked, device, return_components=True)
    if not detach_hist_for_fut:
        return loss_hist, pred_hist_train

    # detach_hist_for_fut=1 时，Fut 使用脱离计算图的 Hist 输出；
    # Hist 只接收自身重建损失，不接收 Fut 分支反传梯度。
    with torch.no_grad():
        _, pred_hist_eval = hist_model.forward_eval(hist, hist_masked, device)
    return loss_hist, pred_hist_eval.detach()


def train_epoch(
    model_fut,
    model_hist,
    dataloader,
    optimizer,
    device,
    epoch,
    feature_dim,
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
    total_x0_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", ncols=140)
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
        loss_fut, fut_parts = model_fut.forwardTrain(
            hist_for_fut,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
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
        total_x0_loss += float(fut_parts["loss_x0"].item())
        num_batches += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg": f"{(total_loss / num_batches):.6f}",
            "hist": f"{(total_hist_loss / num_batches):.6f}",
            "fut": f"{(total_fut_loss / num_batches):.6f}",
        })

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "loss_hist": total_hist_loss / denom,
        "loss_hist_weighted": total_hist_loss_weighted / denom,
        "loss_fut": total_fut_loss / denom,
        "loss_x0": total_x0_loss / denom,
        "fut_k": int(getattr(model_fut, "fut_k", 0)),
    }


@torch.no_grad()
def evaluate(model_fut, model_hist, dataloader, device, epoch, feature_dim, mask_ratio, random_mask_ratio, block_mask_start):
    model_fut.eval()
    model_hist.eval()
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        model_fut.train()
        if model_hist.training:
            model_hist.train()
        return 0.0, 0.0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", ncols=120)
    for batch in pbar:
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
        if int(model_fut.fut_k) > 1:
            all_preds = model_fut.forwardEvalMulti(
                hist_for_fut,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                device,
                K=model_fut.fut_k,
            )
            pred_fut, _, _ = select_minade_prediction(all_preds, fut, op_mask)
        else:
            pred_fut = model_fut.forwardEvalMulti(
                hist_for_fut,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                device,
                K=1,
            ).squeeze(1)
        eval_ade, eval_fde = compute_batch_ade_fde(pred_fut, fut, op_mask)
        eval_ade = float(eval_ade.item()) * METER_PER_FOOT
        eval_fde = float(eval_fde.item()) * METER_PER_FOOT

        total_ade += eval_ade
        total_fde += eval_fde
        num_batches += 1
        pbar.set_postfix({
            "avg_ade_m": f"{(total_ade / num_batches):.4f}",
            "avg_fde_m": f"{(total_fde / num_batches):.4f}",
        })

    model_fut.train()
    return (
        total_ade / max(num_batches, 1),
        total_fde / max(num_batches, 1),
    )


def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(JOINT_FUT_CHECKPOINT_DIR)
    JOINT_FUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    JOINT_HIST_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tensorboard_log_dir = Path(args.checkpoint_dir) / "log"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = tensorboard_log_dir / "train_log.csv"
    if not log_csv_path.exists() or args.resume_fut in ("none", "", None):
        init_csv_log(log_csv_path)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")
    print(f"[JointTrain] Dataset: {dataset_name}")
    print(f"[JointTrain] Train path: {train_path}")
    print(f"[JointTrain] Val path: {val_path}")

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

    freeze_hist = int(args.joint_freeze_hist) > 0
    detach_hist_for_fut = int(args.joint_detach_hist_for_fut) > 0
    hist_loss_weight = 0.0 if freeze_hist else max(0.0, float(args.joint_hist_loss_weight))
    hist_lr_scale = max(0.0, float(args.joint_hist_lr_scale))

    model_hist = DiffusionPast(args).to(device)
    load_hist_checkpoint(model_hist, args.resume_hist, [HIST_CHECKPOINT_DIR, JOINT_HIST_CHECKPOINT_DIR], device, freeze_hist=freeze_hist)

    model_fut = DiffusionFut(args).to(device)
    fut_lr = float(args.learning_rate)
    param_groups = [{"params": model_fut.parameters(), "lr": fut_lr}]
    hist_lr = 0.0
    if not freeze_hist:
        hist_lr = fut_lr * hist_lr_scale
        param_groups.append({"params": model_hist.parameters(), "lr": hist_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_ade = load_fut_checkpoint(args, model_fut, optimizer, scheduler, device)
    mask_ratio = max(0.0, min(1.0, float(args.mask_prob)))
    random_mask_ratio = max(0.0, min(1.0, float(args.random_mask_ratio)))
    block_mask_start = int(args.block_mask_start) > 0

    print(
        f"[JointTrain] freeze_hist={freeze_hist} | "
        f"detach_hist_for_fut={detach_hist_for_fut} | "
        f"hist_loss_weight={hist_loss_weight:.4f} | "
        f"lr_fut={fut_lr:.2e} | lr_hist={hist_lr:.2e}"
    )

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
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
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
        )

        write_csv_log(log_csv_path, epoch + 1, train_stats, eval_ade, eval_fde, fut_lr, hist_lr)
        writer.add_scalar("Loss/Train", train_stats["loss"], epoch + 1)
        writer.add_scalar("Loss/TrainHist", train_stats["loss_hist"], epoch + 1)
        writer.add_scalar("Loss/TrainHistWeighted", train_stats["loss_hist_weighted"], epoch + 1)
        writer.add_scalar("Loss/TrainFut", train_stats["loss_fut"], epoch + 1)
        writer.add_scalar("Eval/ADE_m", eval_ade, epoch + 1)
        writer.add_scalar("Eval/FDE_m", eval_fde, epoch + 1)
        writer.add_scalar("Config/FutK", train_stats.get("fut_k", 0), epoch + 1)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} | "
            f"train={train_stats['loss']:.6f} | "
            f"hist={train_stats['loss_hist']:.6f} | "
            f"fut={train_stats['loss_fut']:.6f} | "
            f"ade_m={eval_ade:.4f} | "
            f"fde_m={eval_fde:.4f} | "
            f"k={int(train_stats.get('fut_k', 0))}"
        )

        scheduler.step()
        is_best = eval_ade < best_ade
        if is_best:
            best_ade = eval_ade

        fut_state = {
            "epoch": epoch + 1,
            "model_state_dict": model_fut.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_stats["loss"],
            "eval_ade_m": eval_ade,
            "eval_fde_m": eval_fde,
            "best_ade": best_ade,
            "joint_freeze_hist": int(freeze_hist),
            "joint_hist_loss_weight": hist_loss_weight,
            "joint_detach_hist_for_fut": int(detach_hist_for_fut),
            "joint_hist_lr_scale": hist_lr_scale,
            "resume_hist": args.resume_hist,
        }
        if (epoch + 1) % args.save_interval == 0:
            torch.save(fut_state, Path(args.checkpoint_dir) / f"epoch_{epoch + 1}.pth")
            if not freeze_hist:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model_hist.state_dict(),
                        "loss_hist": train_stats["loss_hist"],
                    },
                    JOINT_HIST_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1}.pth",
                )
        if is_best:
            torch.save(fut_state, Path(args.checkpoint_dir) / "best.pth")
            if not freeze_hist:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model_hist.state_dict(),
                        "loss_hist": train_stats["loss_hist"],
                    },
                    JOINT_HIST_CHECKPOINT_DIR / "checkpoint_best.pth",
                )

    writer.close()


if __name__ == "__main__":
    main()
