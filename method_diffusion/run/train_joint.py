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
from method_diffusion.utils.fut_utils import compute_batch_kinematic_metrics, compute_batch_metric, select_closest_prediction
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
    best_rmse = float("inf")
    ckpt_path = resolve_fut_checkpoint(args.resume_fut, Path(args.checkpoint_dir))
    if ckpt_path is None:
        return start_epoch, best_rmse
    if not ckpt_path.exists():
        print(f"[JointFut] Checkpoint not found: {ckpt_path}")
        return start_epoch, best_rmse

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
    except Exception:
        pass

    start_epoch = int(state.get("epoch", 0))
    best_rmse = float(state.get("best_rmse_m", state.get("best_score", best_rmse)))
    print(f"[JointFut] Resumed from {ckpt_path} @ epoch {start_epoch}")
    return start_epoch, best_rmse


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


def load_hist_checkpoint(model, resume_arg, checkpoint_dirs, device):
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

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"[JointHist] Loaded checkpoint: {ckpt_path}")
    return model


def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_rmse_m",
        "val_ade_m",
        "val_fde_m",
        "val_theta_deg",
        "val_v_mps",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def write_csv_log(csv_path, epoch, train_stats, eval_rmse, eval_ade, eval_fde, eval_theta_deg, eval_v_mps, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "val_rmse_m": eval_rmse,
        "val_ade_m": eval_ade,
        "val_fde_m": eval_fde,
        "val_theta_deg": eval_theta_deg,
        "val_v_mps": eval_v_mps,
        "lr": lr,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def build_hist_outputs(model_hist, hist, mask_ratio, random_mask_ratio, block_mask_start, device):
    hist_model = model_hist.module if hasattr(model_hist, "module") else model_hist
    hist_masked = build_hist_masked(
        hist,
        mask_ratio=mask_ratio,
        random_mask_ratio=random_mask_ratio,
        block_mask_start=block_mask_start,
    )
    with torch.no_grad():
        _, pred_hist_eval = hist_model.forward_eval(hist, hist_masked, device)
    return pred_hist_eval


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
):
    model_fut.train()
    model_hist.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", ncols=140)
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
        loss, _ = model_fut.forwardTrain(pred_hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fut.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg": f"{(total_loss / num_batches):.6f}",
        })

    denom = max(num_batches, 1)
    return {"loss": total_loss / denom}


@torch.no_grad()
def evaluate(model_fut, model_hist, dataloader, device, epoch, feature_dim, mask_ratio, random_mask_ratio, block_mask_start):
    was_fut_training = model_fut.training
    was_hist_training = model_hist.training
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

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", ncols=120)
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
        if int(model_fut.fut_k) > 1:
            all_preds = model_fut.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, fut, device, K=model_fut.fut_k)
            pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        else:
            all_preds = model_fut.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, fut, device, K=1)
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
        pbar.set_postfix({
            "avg_rmse_m": f"{(total_rmse / num_batches):.4f}",
            "avg_ade_m": f"{(total_ade / num_batches):.4f}",
            "avg_fde_m": f"{(total_fde / num_batches):.4f}",
            "avg_theta_deg": f"{(total_theta_deg / num_batches):.4f}",
            "avg_v_mps": f"{(total_v_mps / num_batches):.4f}",
        })

    if was_fut_training:
        model_fut.train()
    if was_hist_training:
        model_hist.train()
    denom = max(num_batches, 1)
    return (
        total_rmse / denom,
        total_ade / denom,
        total_fde / denom,
        total_theta_deg / denom,
        total_v_mps / denom,
    )


def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(JOINT_FUT_CHECKPOINT_DIR)
    JOINT_FUT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
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

    model_hist = DiffusionPast(args).to(device)
    load_hist_checkpoint(model_hist, args.resume_hist, [HIST_CHECKPOINT_DIR, JOINT_HIST_CHECKPOINT_DIR], device)

    model_fut = DiffusionFut(args).to(device)
    fut_lr = float(args.learning_rate)
    optimizer = torch.optim.AdamW(model_fut.parameters(), lr=fut_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_rmse = load_fut_checkpoint(args, model_fut, optimizer, scheduler, device)
    mask_ratio = max(0.0, min(1.0, float(args.mask_prob)))
    random_mask_ratio = max(0.0, min(1.0, float(args.random_mask_ratio)))
    block_mask_start = int(args.block_mask_start) > 0

    print(
        f"[JointTrain] hist_frozen=1 | lr_fut={fut_lr:.2e}"
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
        )
        eval_rmse, eval_ade, eval_fde, eval_theta_deg, eval_v_mps = evaluate(
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

        fut_state = {
            "epoch": epoch + 1,
            "model_state_dict": model_fut.state_dict(),
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
            torch.save(fut_state, Path(args.checkpoint_dir) / f"epoch_{epoch + 1}.pth")
        if is_best:
            torch.save(fut_state, Path(args.checkpoint_dir) / "best.pth")

    writer.close()


if __name__ == "__main__":
    main()
