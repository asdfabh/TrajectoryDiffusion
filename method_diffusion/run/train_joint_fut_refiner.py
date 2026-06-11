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
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_fut_refiner import (
    FUT_CHECKPOINT_DIR,
    REFINER_CHECKPOINT_DIR,
    compute_refiner_loss,
    resolve_checkpoint_path,
)
from method_diffusion.utils.fut_utils import TrajectoryMetrics, select_closest_prediction


def get_joint_checkpoint_dir(dataset_name):
    return REFINER_CHECKPOINT_DIR / str(dataset_name).strip().lower() / "temporal_basis_joint"


def load_trainable_fut_model(args, device):
    checkpoint_dir = FUT_CHECKPOINT_DIR / str(args.dataset).strip().lower()
    ckpt_path = resolve_checkpoint_path(args.resume_fut, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Fut checkpoint not found: resume_fut={args.resume_fut}, dir={checkpoint_dir}")

    model = DiffusionFut(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"[JointRefinerTrain] Loaded trainable fut checkpoint: {ckpt_path}")
    return model


def build_loader(args, dataset_name, split, shuffle, drop_last):
    split_path = str(get_split_path(args, dataset_name, split))
    dataset = build_trajectory_dataset(split_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)
    print(f"[JointRefinerTrain] {split} path: {split_path}")
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


def train_epoch(args, fut_model, refiner, dataloader, optimizer, device, epoch):
    train_fut = int(args.joint_train_fut) > 0
    fut_model.eval()
    refiner.train()
    totals = {
        "loss": 0.0,
        "xy_loss": 0.0,
        "fde_loss": 0.0,
        "delta_norm": 0.0,
        "gate": 0.0,
    }
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Joint Ep{epoch} Train", dynamic_ncols=True)

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, args.feature_dim, device=device)
        if train_fut:
            all_preds = fut_model.forward_eval_multi(
                hist,
                hist_nbrs,
                mask,
                temporal_mask,
                device,
                K=fut_model.fut_k,
                stage="joint_train",
            )
        else:
            with torch.no_grad():
                all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device, K=fut_model.fut_k)
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
        grad_params = list(refiner.parameters())
        if train_fut:
            grad_params += list(fut_model.parameters())
        torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
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
    fut_model.eval()
    refiner.eval()
    baseline_metrics = TrajectoryMetrics(fut_model.T)
    refined_metrics = TrajectoryMetrics(fut_model.T)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Joint Ep{epoch} Val", dynamic_ncols=True)

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, args.feature_dim, device=device)
        all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device, K=fut_model.fut_k)
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


def write_csv_row(csv_path, epoch, train_stats, baseline_summary, refined_summary, fut_lr, refiner_lr):
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
        "fut_lr": fut_lr,
        "refiner_lr": refiner_lr,
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
    checkpoint_dir = get_joint_checkpoint_dir(dataset_name)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = checkpoint_dir / "train_log.csv"

    print(f"[JointRefinerTrain] Dataset: {dataset_name}")
    print(f"[JointRefinerTrain] Device: {device}")
    print(f"[JointRefinerTrain] Checkpoint dir: {checkpoint_dir}")
    print("[JointRefinerTrain] Refiner: TABR-temporal-basis")

    train_loader = build_loader(args, dataset_name, "Train", shuffle=True, drop_last=True)
    val_loader = build_loader(args, dataset_name, "Val", shuffle=False, drop_last=False)

    fut_model = load_trainable_fut_model(args, device)
    fut_model.eval()
    train_fut = int(args.joint_train_fut) > 0
    for param in fut_model.parameters():
        param.requires_grad_(train_fut)
    refiner = build_trajectory_refiner(args).to(device)
    optimizer_groups = [
        {
            "params": refiner.parameters(),
            "lr": args.fut_refiner_lr,
        },
    ]
    if train_fut:
        optimizer_groups.append(
            {
                "params": fut_model.parameters(),
                "lr": args.joint_fut_lr,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(args.num_epochs), 1))
    print(
        f"[JointRefinerTrain] Training: fut_eval + refiner | "
        f"joint_train_fut={int(train_fut)} | "
        f"lr_refiner={float(args.fut_refiner_lr):.2e} | "
        f"lr_fut={float(args.joint_fut_lr) if train_fut else 0.0:.2e}"
    )

    best_rmse = float("inf")
    for epoch in range(1, int(args.num_epochs) + 1):
        train_stats = train_epoch(args, fut_model, refiner, train_loader, optimizer, device, epoch)
        baseline_summary, refined_summary = evaluate(args, fut_model, refiner, val_loader, device, epoch)
        last_idx = len(refined_summary["rmse_per_step_m"]) - 1
        selection_score = float(refined_summary["rmse_per_step_m"][last_idx].item())
        fut_lr = optimizer.param_groups[1]["lr"] if train_fut else 0.0
        refiner_lr = optimizer.param_groups[0]["lr"]
        write_csv_row(csv_path, epoch, train_stats, baseline_summary, refined_summary, fut_lr, refiner_lr)
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
            "fut_model_state_dict": fut_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rmse_m": best_rmse,
            "refiner_type": "temporal_basis_joint",
            "args": vars(args),
        }
        if epoch % int(args.save_interval) == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch}.pth")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pth")


if __name__ == "__main__":
    main()
