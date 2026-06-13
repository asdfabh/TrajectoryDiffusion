import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_joint import (
    JOINT_FUT_CHECKPOINT_DIR,
    build_hist_outputs,
    hist_checkpoint_dirs_for_dataset,
    load_hist_checkpoint,
    normalize_dataset_name,
    resolve_fut_checkpoint,
)
from method_diffusion.run.train_refine import compute_refiner_loss
from method_diffusion.utils.fut_utils import TrajectoryMetrics, select_closest_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOINT_REFINER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "joint_refine"


def get_joint_refiner_checkpoint_dir(dataset_name):
    return JOINT_REFINER_CHECKPOINT_DIR / normalize_dataset_name(dataset_name)


def build_loader(args, dataset_name, split, shuffle, drop_last):
    split_path = str(get_split_path(args, dataset_name, split))
    dataset = build_trajectory_dataset(split_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)
    print(f"[JointRefineTrain] {split} path: {split_path}")
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


def load_frozen_joint_fut_model(args, dataset_name, device):
    checkpoint_dir = JOINT_FUT_CHECKPOINT_DIR / dataset_name
    resume_fut = "best" if args.resume_fut in ("none", "", None) else args.resume_fut
    ckpt_path = resolve_fut_checkpoint(resume_fut, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Joint Fut checkpoint not found: resume_fut={args.resume_fut}, dir={checkpoint_dir}. "
            "For joint refine training, use 'none', 'best', 'epoch_i' such as 'epoch_10', or an existing path."
        )

    model = DiffusionFut(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    print(f"[JointRefineTrain] Loaded frozen joint fut checkpoint: {ckpt_path}")
    return model


def load_frozen_hist_model(args, dataset_name, device):
    model = DiffusionPast(args).to(device)
    load_hist_checkpoint(
        model,
        args.resume_hist,
        hist_checkpoint_dirs_for_dataset(dataset_name),
        device,
        trainable=False,
        dataset_name=dataset_name,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def build_joint_proposals(args, hist_model, fut_model, batch, device, dataset_name, enable_latent_bridge):
    hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, args.feature_dim, device=device)
    pred_hist, past_latent_tokens = build_hist_outputs(
        model_hist=hist_model,
        hist=hist,
        mask_ratio=max(0.0, min(1.0, float(args.mask_prob))),
        random_mask_ratio=max(0.0, min(1.0, float(args.random_mask_ratio))),
        block_mask_start=int(args.block_mask_start) > 0,
        device=device,
        return_tokens=enable_latent_bridge,
    )
    all_preds = fut_model.forwardEvalMulti(
        pred_hist,
        hist_nbrs,
        mask,
        temporal_mask,
        device,
        K=fut_model.fut_k,
        past_latent_tokens=past_latent_tokens,
    )
    return pred_hist, all_preds, fut, op_mask


def train_epoch(args, hist_model, fut_model, refiner, dataloader, optimizer, device, epoch, dataset_name, enable_latent_bridge):
    refiner.train()
    totals = {
        "loss": 0.0,
        "xy_loss": 0.0,
        "fde_loss": 0.0,
        "delta_norm": 0.0,
        "gate": 0.0,
    }
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"JointRefine Ep{epoch} Train", dynamic_ncols=True)

    for batch in pbar:
        pred_hist, all_preds, fut, op_mask = build_joint_proposals(
            args,
            hist_model,
            fut_model,
            batch,
            device,
            dataset_name,
            enable_latent_bridge,
        )
        refined_all, aux = refiner(pred_hist, all_preds, fut_model.fut_dt)
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
def evaluate(args, hist_model, fut_model, refiner, dataloader, device, epoch, dataset_name, enable_latent_bridge):
    refiner.eval()
    baseline_metrics = TrajectoryMetrics(fut_model.T)
    refined_metrics = TrajectoryMetrics(fut_model.T)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"JointRefine Ep{epoch} Val", dynamic_ncols=True)

    for batch in pbar:
        pred_hist, all_preds, fut, op_mask = build_joint_proposals(
            args,
            hist_model,
            fut_model,
            batch,
            device,
            dataset_name,
            enable_latent_bridge,
        )
        pred_fut, _, _ = select_closest_prediction(all_preds, fut, op_mask)
        refined_all, _ = refiner(pred_hist, all_preds, fut_model.fut_dt)
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
    dataset_name = normalize_dataset_name(args.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = get_joint_refiner_checkpoint_dir(dataset_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = checkpoint_dir / "train_log.csv"
    enable_latent_bridge = int(args.enable_past_fut_latent_bridge) > 0

    print(f"[JointRefineTrain] Dataset: {dataset_name}")
    print(f"[JointRefineTrain] Device: {device}")
    print(f"[JointRefineTrain] Checkpoint dir: {checkpoint_dir}")
    print(f"[JointRefineTrain] latent_bridge={int(enable_latent_bridge)}")
    train_loader = build_loader(args, dataset_name, "Train", shuffle=True, drop_last=True)
    val_loader = build_loader(args, dataset_name, "Val", shuffle=False, drop_last=False)

    hist_model = load_frozen_hist_model(args, dataset_name, device)
    fut_model = load_frozen_joint_fut_model(args, dataset_name, device)
    refiner = build_trajectory_refiner(args).to(device)
    print("[JointRefineTrain] Refiner: TABR-temporal-basis")
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.fut_refiner_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.num_epochs), 1))

    best_rmse = float("inf")
    for epoch in range(1, int(args.num_epochs) + 1):
        train_stats = train_epoch(
            args,
            hist_model,
            fut_model,
            refiner,
            train_loader,
            optimizer,
            device,
            epoch,
            dataset_name,
            enable_latent_bridge,
        )
        baseline_summary, refined_summary = evaluate(
            args,
            hist_model,
            fut_model,
            refiner,
            val_loader,
            device,
            epoch,
            dataset_name,
            enable_latent_bridge,
        )
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
            "resume_hist": args.resume_hist,
            "resume_fut": args.resume_fut,
            "enable_past_fut_latent_bridge": int(enable_latent_bridge),
            "args": vars(args),
        }
        if epoch % int(args.save_interval) == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pth")


if __name__ == "__main__":
    main()
