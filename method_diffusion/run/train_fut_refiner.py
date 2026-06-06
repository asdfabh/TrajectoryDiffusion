import csv
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.utils.fut_utils import TrajectoryMetrics, normalize_traj_valid_mask, select_closest_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
REFINER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut_refiner"


def get_refiner_checkpoint_dir(dataset_name):
    return REFINER_CHECKPOINT_DIR / str(dataset_name).strip().lower() / "temporal_basis"


def resolve_checkpoint_path(resume_arg, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg in ("none", "", None):
        resume_arg = "best"
    if Path(str(resume_arg)).exists():
        return Path(str(resume_arg))
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if re.fullmatch(r"epoch_\d+", str(resume_arg)):
        return checkpoint_dir / f"{resume_arg}.pth"
    return None


def load_frozen_fut_model(args, device):
    checkpoint_dir = FUT_CHECKPOINT_DIR / str(args.dataset).strip().lower()
    ckpt_path = resolve_checkpoint_path(args.resume_fut, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Fut checkpoint not found: resume_fut={args.resume_fut}, dir={checkpoint_dir}")

    model = DiffusionFut(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[RefinerTrain] Loaded frozen fut checkpoint: {ckpt_path}")
    return model


def build_loader(args, dataset_name, split, shuffle, drop_last):
    split_path = str(get_split_path(args, dataset_name, split))
    dataset = build_trajectory_dataset(split_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)
    print(f"[RefinerTrain] {split} path: {split_path}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=drop_last,
    )


def compute_refiner_loss(
    refined_all,
    fut,
    op_mask,
    aux,
    fde_weight,
    delta_weight,
    gate_weight,
):
    valid = normalize_traj_valid_mask(op_mask, fut).bool()
    target_xy = fut[..., :2].unsqueeze(1)
    pred_xy = refined_all[..., :2]
    dist = torch.linalg.norm(pred_xy - target_xy, dim=-1)
    bsz, k_size, t_len = dist.shape
    time_weight = torch.linspace(
        1.0 / max(t_len, 1),
        1.0,
        t_len,
        device=dist.device,
        dtype=dist.dtype,
    ).pow(2.0)
    weighted_valid = valid.unsqueeze(1).float() * time_weight.view(1, 1, t_len)
    xy_loss = (dist * weighted_valid).sum(dim=2) / (weighted_valid.sum(dim=2) + 1e-6)

    valid_counts = valid.sum(dim=1).long()
    has_valid = valid_counts > 0
    final_idx = torch.clamp(valid_counts - 1, min=0)
    final_dist = dist.gather(2, final_idx.view(bsz, 1, 1).expand(bsz, k_size, 1)).squeeze(2)
    final_dist = final_dist * has_valid.float().unsqueeze(1)

    delta_norm = aux.get("delta_regularizer", torch.linalg.norm(aux["delta_end"], dim=-1))
    gate = aux["gate"].squeeze(-1)
    loss_k = (
        xy_loss
        + float(fde_weight) * final_dist
        + float(delta_weight) * delta_norm
        + float(gate_weight) * gate
    )
    best_idx = torch.argmin((xy_loss + float(fde_weight) * final_dist).detach(), dim=1)
    selected_loss = loss_k.gather(1, best_idx.unsqueeze(1)).squeeze(1)
    return selected_loss[has_valid].mean(), {
        "xy_loss": xy_loss.gather(1, best_idx.unsqueeze(1)).squeeze(1)[has_valid].mean().detach(),
        "fde_loss": final_dist.gather(1, best_idx.unsqueeze(1)).squeeze(1)[has_valid].mean().detach(),
        "delta_norm": delta_norm.gather(1, best_idx.unsqueeze(1)).squeeze(1)[has_valid].mean().detach(),
        "gate": gate.gather(1, best_idx.unsqueeze(1)).squeeze(1)[has_valid].mean().detach(),
    }


def train_epoch(args, fut_model, refiner, dataloader, optimizer, device, epoch):
    refiner.train()
    totals = {
        "loss": 0.0,
        "xy_loss": 0.0,
        "fde_loss": 0.0,
        "delta_norm": 0.0,
        "gate": 0.0,
    }
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"E6 Ep{epoch} Train", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar, start=1):
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, args.feature_dim, device=device)
        with torch.no_grad():
            all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, fut, device, K=fut_model.fut_k)
        refined_all, aux = refiner(hist, all_preds, fut_model.fut_dt)
        loss, logs = compute_refiner_loss(
            refined_all,
            fut,
            op_mask,
            aux,
            args.fut_refiner_fde_weight,
            args.fut_refiner_delta_weight,
            args.fut_refiner_gate_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"] += float(loss.item())
        for key, value in logs.items():
            totals[key] += float(value.item())
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.5f}", "gate": f"{logs['gate'].item():.3f}"})

    denom = max(num_batches, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def evaluate(args, fut_model, refiner, dataloader, device, epoch):
    refiner.eval()
    baseline_metrics = TrajectoryMetrics(fut_model.T)
    refined_metrics = TrajectoryMetrics(fut_model.T)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"E6 Ep{epoch} Val", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar, start=1):
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, args.feature_dim, device=device)
        all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, fut, device, K=fut_model.fut_k)
        pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        refined_all, _ = refiner(hist, all_preds, fut_model.fut_dt)
        refined_pred, _, _ = select_closest_prediction(refined_all, fut, op_mask)
        baseline_metrics.update(pred_fut, fut, op_mask)
        refined_metrics.update(refined_pred, fut, op_mask)
        summary = refined_metrics.summary()
        last_idx = refined_pred.size(1) - 1
        pbar.set_postfix({
            "rmse": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
            "ade": f"{summary['ade_per_step_m'][last_idx]:.4f}",
            "fde": f"{summary['fde_per_step_m'][last_idx]:.4f}",
        })

    return baseline_metrics.summary(), refined_metrics.summary()


def write_csv_row(csv_path, epoch, train_stats, baseline_summary, refined_summary, lr):
    last_idx = len(refined_summary["rmse_per_step_m"]) - 1
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_xy_loss": train_stats["xy_loss"],
        "train_fde_loss": train_stats["fde_loss"],
        "train_delta_norm": train_stats["delta_norm"],
        "train_gate": train_stats["gate"],
        "baseline_rmse": baseline_summary["rmse_per_step_m"][last_idx].item(),
        "baseline_ade": baseline_summary["ade_per_step_m"][last_idx].item(),
        "baseline_fde": baseline_summary["fde_per_step_m"][last_idx].item(),
        "refined_rmse": refined_summary["rmse_per_step_m"][last_idx].item(),
        "refined_ade": refined_summary["ade_per_step_m"][last_idx].item(),
        "refined_fde": refined_summary["fde_per_step_m"][last_idx].item(),
        "lr": lr,
    }
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = get_args_parser().parse_args()
    dataset_name = str(args.dataset).strip().lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = get_refiner_checkpoint_dir(dataset_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = checkpoint_dir / "train_log.csv"

    print(f"[RefinerTrain] Dataset: {dataset_name}")
    print(f"[RefinerTrain] Device: {device}")
    print(f"[RefinerTrain] Checkpoint dir: {checkpoint_dir}")
    train_loader = build_loader(args, dataset_name, "Train", shuffle=True, drop_last=True)
    val_loader = build_loader(args, dataset_name, "Val", shuffle=False, drop_last=False)

    fut_model = load_frozen_fut_model(args, device)
    refiner = build_trajectory_refiner(args).to(device)
    print("[RefinerTrain] Refiner: TABR-temporal-basis")
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.fut_refiner_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.num_epochs), 1))

    best_rmse = float("inf")
    for epoch in range(1, int(args.num_epochs) + 1):
        train_stats = train_epoch(args, fut_model, refiner, train_loader, optimizer, device, epoch)
        baseline_summary, refined_summary = evaluate(args, fut_model, refiner, val_loader, device, epoch)
        last_idx = len(refined_summary["rmse_per_step_m"]) - 1
        selection_score = float(refined_summary["rmse_per_step_m"][last_idx].item())
        current_lr = optimizer.param_groups[0]["lr"]
        write_csv_row(csv_path, epoch, train_stats, baseline_summary, refined_summary, current_lr)
        print(
            f"Epoch {epoch}/{args.num_epochs} | "
            f"loss={train_stats['loss']:.6f} | "
            f"gate={train_stats['gate']:.4f} | "
            f"baseline_rmse={baseline_summary['rmse_per_step_m'][last_idx].item():.4f} | "
            f"refined_rmse={refined_summary['rmse_per_step_m'][last_idx].item():.4f} | "
            f"refined_ade={refined_summary['ade_per_step_m'][last_idx].item():.4f} | "
            f"refined_fde={refined_summary['fde_per_step_m'][last_idx].item():.4f}"
        )
        scheduler.step()

        is_best = selection_score < best_rmse
        if is_best:
            best_rmse = selection_score
        state = {
            "epoch": epoch,
            "model_state_dict": refiner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rmse_m": best_rmse,
            "refiner_type": "temporal_basis",
            "args": vars(args),
        }
        if epoch % int(args.save_interval) == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pth")


if __name__ == "__main__":
    main()
