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
from method_diffusion.utils.fut_utils import compute_batch_ade_fde, select_minade_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
BEST_MODEL_ADE_WEIGHT = 1.0
BEST_MODEL_FDE_WEIGHT = 0.5
METER_PER_FOOT = 0.3048
LOSS_STAT_KEYS = [
    "loss",
    "loss_x0",
]


def compute_selection_score(eval_ade, eval_fde):
    return (BEST_MODEL_ADE_WEIGHT * float(eval_ade)) + (BEST_MODEL_FDE_WEIGHT * float(eval_fde))

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

# 按需恢复训练状态并返回起始 epoch 与最佳联合分数。
def load_checkpoint(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_score = float("inf")
    ckpt_path = resolve_resume_checkpoint(args.resume_fut, Path(args.checkpoint_dir))

    if ckpt_path is not None:
        if not ckpt_path.exists():
            print(f"[FutModel] Checkpoint not found: {ckpt_path}")
            return start_epoch, best_score

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)

        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception:
            pass
        start_epoch = int(state.get("epoch", 0))
        best_score = float(state.get("best_score", best_score))
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_score

# 将单个 epoch 的训练和验证结果追加写入 CSV。
def write_csv_log(csv_path, epoch, train_stats, val_stats, eval_ade, eval_fde, selection_score, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "val_loss": val_stats["loss"],
        "val_ade_m": eval_ade,
        "val_fde_m": eval_fde,
        "val_score_m": selection_score,
        "lr": lr,
        "fut_k": train_stats.get("fut_k", 0),
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)

# 覆盖创建 CSV 日志文件并写入固定表头。
def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_ade_m",
        "val_fde_m",
        "val_score_m",
        "lr",
        "fut_k",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


# 返回统一格式的空损失统计。
def build_zero_loss_stats():
    return {key: 0.0 for key in LOSS_STAT_KEYS}


# 将训练和验证指标统一写入 TensorBoard。
def write_tensorboard_log(writer, epoch, train_stats, val_stats, eval_ade, eval_fde, selection_score, lr):
    writer.add_scalar("Loss/Train", train_stats["loss"], epoch)
    writer.add_scalar("Loss/Val", val_stats["loss"], epoch)
    writer.add_scalar("Eval/ADE_m", eval_ade, epoch)
    writer.add_scalar("Eval/FDE_m", eval_fde, epoch)
    writer.add_scalar("Eval/SelectionScore_m", selection_score, epoch)
    writer.add_scalar("LR", lr, epoch)
    writer.add_scalar("Config/FutK", train_stats.get("fut_k", 0), epoch)


# 打印每个 epoch 的训练与验证摘要。
def print_eval_summary(epoch, total_epochs, train_stats, val_stats, eval_ade, eval_fde, selection_score):
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train={train_stats['loss']:.6f} | "
        f"val={val_stats['loss']:.6f} | "
        f"ade_m={eval_ade:.4f} | "
        f"fde_m={eval_fde:.4f} | "
        f"score_m={selection_score:.4f} | "
        f"k={int(train_stats.get('fut_k', 0))}"
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
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        dynamic_ncols=True
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_parts = model.forwardTrain(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, return_components=True)
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
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg_loss": f"{(totals['loss'] / num_batches):.6f}",
        })

    denom = max(num_batches, 1)
    stats = {key: totals[key] / denom for key in LOSS_STAT_KEYS}
    stats["fut_k"] = int(getattr(model, "fut_k", 0))
    return stats

@torch.no_grad()
# 使用完整验证集评估模型并返回平均指标。
def evaluate(model, dataloader, device, epoch, feature_dim):
    model.eval()
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    if len(dataloader) == 0:
        model.train()
        return build_zero_loss_stats(), 0.0, 0.0

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True)
    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        val_loss, val_parts = model.forwardTrain(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, return_components=True)
        if int(model.fut_k) > 1:
            all_preds = model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, fut, device, K=model.fut_k)
            pred_fut, _, _ = select_minade_prediction(all_preds, fut, op_mask)
        else:
            pred_fut = model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, device)
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
        pbar.set_postfix({
            "val_loss": f"{(totals['loss'] / num_batches):.6f}",
            "avg_ade_m": f"{(total_ade / num_batches):.4f}",
            "avg_fde_m": f"{(total_fde / num_batches):.4f}",
        })

    model.train()
    if num_batches == 0:
        return build_zero_loss_stats(), 0.0, 0.0

    denom = float(num_batches)
    val_stats = {key: totals[key] / denom for key in LOSS_STAT_KEYS}
    val_stats["fut_k"] = int(getattr(model, "fut_k", 0))
    return val_stats, total_ade / denom, total_fde / denom


# 初始化训练组件并执行 fut 训练主流程。
def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    tensorboard_log_dir = Path(args.checkpoint_dir) / "log"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = tensorboard_log_dir / "train_log.csv"
    init_csv_log(log_csv_path)
    writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    train_path = str(data_root / "TrainSet.mat")
    val_path = str(data_root / "ValSet.mat")
    print(f"[FutTrain] Dataset: {dataset_name}")
    print(f"[FutTrain] Train path: {train_path}")
    print(f"[FutTrain] Val path: {val_path}")

    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)
    val_dataset = NgsimDataset(val_path, t_h=30, t_f=50, d_s=2, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)

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
    start_epoch, best_score = load_checkpoint(args, model, optimizer, scheduler, device)

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim)
        val_stats, eval_ade, eval_fde = evaluate(model, val_loader, device, epoch + 1, args.feature_dim)
        selection_score = compute_selection_score(eval_ade, eval_fde)
        current_lr = optimizer.param_groups[0]["lr"]
        write_csv_log(log_csv_path, epoch + 1, train_stats, val_stats, eval_ade, eval_fde, selection_score, current_lr)
        write_tensorboard_log(writer, epoch + 1, train_stats, val_stats, eval_ade, eval_fde, selection_score, current_lr)
        print_eval_summary(epoch + 1, args.num_epochs, train_stats, val_stats, eval_ade, eval_fde, selection_score)

        scheduler.step()
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
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

    writer.close()


if __name__ == "__main__":
    main()
