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
from method_diffusion.dataset.build import build_trajectory_dataset, get_split_path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.utils.fut_utils import (
    TrajectoryMetrics, reduce_trajectory_metrics, print_trajectory_metrics,
    validation_metric_values, write_horizon_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
LOSS_STAT_KEYS = [
    "loss",
    "loss_xy",
    "loss_theta",
    "loss_v",
    "loss_kin",
    "loss_final",
    "loss_cascade_aux",
    "kin_res_mean",
    "assign_same_rate",
    "assign_max_win_ratio",
]

# 解析 resume 标识并返回对应的 checkpoint 路径。
def resolve_resume_checkpoint(resume_arg, checkpoint_dir):
    if resume_arg in ("none", "", None):
        return None
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if re.fullmatch(r"epoch_\d+", str(resume_arg)):
        return checkpoint_dir / f"{resume_arg}.pth"
    print(f"[FutModel] Unsupported resume_fut='{resume_arg}', expected 'best' or 'epoch_i'.")
    return None

# 按需恢复训练状态并返回起始 epoch 与最佳 RMSE。
def load_checkpoint(resume_arg, checkpoint_dir, model, optimizer, scheduler, device):
    start_epoch = 0
    best_rmse = float("inf")
    ckpt_path = resolve_resume_checkpoint(resume_arg, Path(checkpoint_dir))

    if ckpt_path is not None:
        if not ckpt_path.exists():
            print(f"[FutModel] Checkpoint not found: {ckpt_path}")
            return start_epoch, best_rmse

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)

        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception:
            pass
        start_epoch = int(state.get("epoch", 0))
        best_rmse = (float(state.get("best_rmse_m", state.get("best_score", best_rmse)))
                     if state.get("selection_metric") == "rmse_full_m" else float("inf"))
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_rmse

# 将单个 epoch 的训练和验证结果追加写入 CSV。
def write_csv_log(csv_path, epoch, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_loss_xy": train_stats["loss_xy"],
        "train_loss_theta": train_stats["loss_theta"],
        "train_loss_v": train_stats["loss_v"],
        "train_loss_kin": train_stats["loss_kin"],
        "train_loss_final": train_stats["loss_final"],
        "train_loss_cascade_aux": train_stats["loss_cascade_aux"],
        "train_kin_res_mean": train_stats["kin_res_mean"],
        "train_assign_same_rate": train_stats["assign_same_rate"],
        "train_assign_max_win_ratio": train_stats["assign_max_win_ratio"],
        "val_rmse_full_selection_m": eval_rmse,
        "val_min_ade_final_horizon_m": eval_ade,
        "val_min_fde_final_horizon_m": eval_fde,
        "val_rmse_final_horizon_m": eval_rmse_5s,
        "val_theta_deg": eval_theta_deg,
        "val_v_mps": eval_v_mps,
        "lr": lr,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

# 覆盖创建 CSV 日志文件并写入固定表头。
def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_xy",
        "train_loss_theta",
        "train_loss_v",
        "train_loss_kin",
        "train_loss_final",
        "train_loss_cascade_aux",
        "train_kin_res_mean",
        "train_assign_same_rate",
        "train_assign_max_win_ratio",
        "val_rmse_full_selection_m",
        "val_min_ade_final_horizon_m",
        "val_min_fde_final_horizon_m",
        "val_rmse_final_horizon_m",
        "val_theta_deg",
        "val_v_mps",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


# 将训练和验证指标统一写入 TensorBoard。
def write_tensorboard_log(writer, epoch, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, lr):
    writer.add_scalar("Loss/Train", train_stats["loss"], epoch)
    writer.add_scalar("Loss/TrainXY", train_stats["loss_xy"], epoch)
    writer.add_scalar("Loss/TrainTheta", train_stats["loss_theta"], epoch)
    writer.add_scalar("Loss/TrainV", train_stats["loss_v"], epoch)
    writer.add_scalar("Loss/TrainKinematic", train_stats["loss_kin"], epoch)
    writer.add_scalar("Loss/TrainFinal", train_stats["loss_final"], epoch)
    writer.add_scalar("Loss/TrainCascadeAux", train_stats["loss_cascade_aux"], epoch)
    writer.add_scalar("Loss/KinematicResidualMean", train_stats["kin_res_mean"], epoch)
    writer.add_scalar("Train/AssignSameRate", train_stats["assign_same_rate"], epoch)
    writer.add_scalar("Train/AssignMaxWinRatio", train_stats["assign_max_win_ratio"], epoch)
    writer.add_scalar("Eval/RMSE_full_selection_m", eval_rmse, epoch)
    writer.add_scalar("Eval/minADE_final_horizon_m", eval_ade, epoch)
    writer.add_scalar("Eval/minFDE_final_horizon_m", eval_fde, epoch)
    writer.add_scalar("Eval/RMSE_final_horizon_m", eval_rmse_5s, epoch)
    writer.add_scalar("Eval/Theta_deg", eval_theta_deg, epoch)
    writer.add_scalar("Eval/V_mps", eval_v_mps, epoch)
    writer.add_scalar("LR", lr, epoch)


# 打印每个 epoch 的训练与验证摘要。
def print_eval_summary(epoch, total_epochs, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps):
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train={train_stats['loss']:.6f} | "
        f"xy={train_stats['loss_xy']:.6f} | "
        f"theta={train_stats['loss_theta']:.6f} | "
        f"v={train_stats['loss_v']:.6f} | "
        f"kin={train_stats['loss_kin']:.6f} | "
        f"final={train_stats['loss_final']:.6f} | "
        f"cascade_aux={train_stats['loss_cascade_aux']:.6f} | "
        f"kin_res={train_stats['kin_res_mean']:.6f} | "
        f"assign_same={train_stats['assign_same_rate']:.4f} | "
        f"assign_max={train_stats['assign_max_win_ratio']:.4f} | "
        f"rmse_full_selection_m={eval_rmse:.4f} | "
        f"minADE_final_horizon_m={eval_ade:.4f} | "
        f"minFDE_final_horizon_m={eval_fde:.4f} | "
        f"rmse_final_horizon_m={eval_rmse_5s:.4f} | "
        f"theta_deg={eval_theta_deg:.4f} | "
        f"v_mps={eval_v_mps:.4f}"
    )

# 整理 batch 数据并按特征维度拼接模型输入。
def prepare_input_data(batch, feature_dim, device="cuda"):
    hist = batch["hist"]
    va = batch["va"]
    lane = batch["lane"]
    cclass = batch["cclass"]
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    hist_nbrs = batch["nbrs"]
    va_nbrs = batch["nbrs_va"]
    lane_nbrs = batch["nbrs_lane"]
    cclass_nbrs = batch["nbrs_class"]
    mask = batch["mask"]
    temporal_mask = batch["temporal_mask"]

    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs, cclass_nbrs), dim=-1).to(device)
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)
    else:
        hist = hist.to(device)
        hist_nbrs = hist_nbrs.to(device)

    fut = fut.to(device)
    op_mask = op_mask.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)
    return hist, hist_nbrs, mask, temporal_mask, fut, op_mask

# 执行单个训练 epoch 并汇总平均损失。
def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim):
    model.train()
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", dynamic_ncols=True)

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_logs = model.forwardTrain(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for key in LOSS_STAT_KEYS:
            totals[key] += float(loss_logs[key].item())
        num_batches += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg_loss": f"{(totals['loss'] / num_batches):.6f}",
        })

    denom = max(num_batches, 1)
    return {key: totals[key] / denom for key in LOSS_STAT_KEYS}

@torch.no_grad()
# 使用完整验证集评估模型并返回平均指标。
def evaluate(model, dataloader, device, epoch, feature_dim, return_summary=False):
    was_training = model.training
    fut_model = model.module if hasattr(model, "module") else model
    model.eval()
    metrics = TrajectoryMetrics(fut_model.T, num_candidates=fut_model.fut_k)
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True)
    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        all_preds = fut_model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device, K=fut_model.fut_k)
        metrics.update(all_preds, fut, op_mask)
        summary = metrics.summary()
        pbar.set_postfix({
            "RMSE_full_selection": f"{summary['rmse_full_m']:.4f}",
            "RMSE_final_horizon": f"{summary['rmse_per_step_m'][-1]:.4f}",
            "minADE_final_horizon": f"{summary['min_ade_per_step_m'][-1]:.4f}",
            "minFDE_final_horizon": f"{summary['min_fde_per_step_m'][-1]:.4f}",
        })
    summary = metrics.summary()
    print_trajectory_metrics(summary, f"Val epoch {epoch}", fut_model.fut_dt)
    model.train(was_training)
    return summary if return_summary else validation_metric_values(summary)


# 初始化训练组件并执行 fut 训练主流程。
def main():
    args = get_args_parser().parse_args()
    dataset_name = str(args.dataset).lower()
    checkpoint_dir = FUT_CHECKPOINT_DIR / dataset_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = checkpoint_dir / "log"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = tensorboard_log_dir / "train_log.csv"
    init_csv_log(log_csv_path)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = str(get_split_path(args, dataset_name, "Train"))
    val_path = str(get_split_path(args, dataset_name, "Val"))
    print(f"[FutTrain] Dataset: {dataset_name}")
    print(f"[FutTrain] Train path: {train_path}")
    print(f"[FutTrain] Val path: {val_path}")

    train_dataset = build_trajectory_dataset(train_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)
    val_dataset = build_trajectory_dataset(val_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)

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

    model = DiffusionFut(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    start_epoch, best_rmse = load_checkpoint(args.resume_fut, checkpoint_dir, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim)
        eval_summary = evaluate(
            model, val_loader, device, epoch + 1, args.feature_dim, return_summary=True)
        eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps = validation_metric_values(eval_summary)
        selection_score = float(eval_rmse)
        current_lr = optimizer.param_groups[0]["lr"]
        write_horizon_metrics(checkpoint_dir / "log" / "val_metrics_per_second.csv", epoch + 1, eval_summary, model.module.fut_dt if hasattr(model, "module") else model.fut_dt, writer=writer)
        write_csv_log(log_csv_path, epoch + 1, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, current_lr)
        write_tensorboard_log(writer, epoch + 1, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps, current_lr)
        print_eval_summary(epoch + 1, args.num_epochs, train_stats, eval_rmse, eval_ade, eval_fde, eval_rmse_5s, eval_theta_deg, eval_v_mps)

        scheduler.step()
        is_best = selection_score < best_rmse
        if is_best:
            best_rmse = selection_score

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_stats["loss"],
            "eval_rmse_full_m": eval_rmse,
            "eval_min_ade_final_horizon_m": eval_ade,
            "eval_min_fde_final_horizon_m": eval_fde,
            "eval_rmse_final_horizon_m": eval_rmse_5s,
            "eval_theta_deg": eval_theta_deg,
            "eval_v_mps": eval_v_mps,
            "selection_score": selection_score,
            "selection_metric": "rmse_full_m",
            "eval_metrics": eval_summary,
            "best_score": best_rmse,
            "best_rmse_m": best_rmse,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch + 1}.pth")
        if is_best:
            torch.save(state, checkpoint_dir / "best.pth")

    writer.close()


if __name__ == "__main__":
    main()
