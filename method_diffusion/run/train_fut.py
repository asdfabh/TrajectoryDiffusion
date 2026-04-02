import csv
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"

LOSS_STAT_KEYS = [
    "loss",
    "loss_intent_lat",
    "loss_intent_lon",
    "loss_mode",
    "loss_anchor",
    "loss_eps",
    "loss_score",
    "loss_rank",
]

VAL_METRIC_KEYS = [
    "top1_ade",
    "top1_fde",
    "lat_acc",
    "lon_acc",
    "joint_acc",
    "intent_topk_hit",
    "best_intent_acc",
    "route_hit",
    "route_gap",
]


def flatten_train_parts(parts):
    losses = parts["losses"]
    return {
        "loss": float(parts["loss_total"].item()),
        "loss_intent_lat": float(losses["intent_lat"].item()),
        "loss_intent_lon": float(losses["intent_lon"].item()),
        "loss_mode": float(losses["mode"].item()),
        "loss_anchor": float(losses["anchor"].item()),
        "loss_eps": float(losses["eps"].item()),
        "loss_score": float(losses["score"].item()),
        "loss_rank": float(losses["rank"].item()),
    }


def flatten_eval_aux(aux):
    metrics = aux["metrics"]
    return {
        "top1_ade": float(metrics["top1_ade"].item()),
        "top1_fde": float(metrics["top1_fde"].item()),
        "lat_acc": float(metrics["lat_acc"].item()),
        "lon_acc": float(metrics["lon_acc"].item()),
        "joint_acc": float(metrics["joint_acc"].item()),
        "intent_topk_hit": float(metrics["intent_topk_hit"].item()),
        "best_intent_acc": float(metrics["best_intent_acc"].item()),
        "route_hit": float(metrics["route_hit"].item()),
        "route_gap": float(metrics["route_gap"].item()),
    }


def compute_exec_selection_score(top1_ade, top1_fde):
    return float(top1_ade) + 0.5 * float(top1_fde)


def resolve_resume_checkpoint(resume_arg, checkpoint_dir):
    if resume_arg in ("none", "", None):
        return None
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if resume_arg == "latest":
        return checkpoint_dir / "latest.pth"
    if re.fullmatch(r"epoch_\d+", str(resume_arg)):
        return checkpoint_dir / f"{resume_arg}.pth"
    print(f"[FutModel] Unsupported resume_fut='{resume_arg}', expected 'best', 'latest' or 'epoch_i'.")
    return None


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
        if "best_score" in state:
            best_score = float(state["best_score"])
        elif "eval_ade" in state and "eval_fde" in state:
            best_score = compute_exec_selection_score(state["eval_ade"], state["eval_fde"])
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_score


def write_csv_log(csv_path, epoch, train_stats, val_stats, eval_metrics, score_exec, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_loss_intent_lat": train_stats["loss_intent_lat"],
        "train_loss_intent_lon": train_stats["loss_intent_lon"],
        "train_loss_mode": train_stats["loss_mode"],
        "train_loss_anchor": train_stats["loss_anchor"],
        "train_loss_eps": train_stats["loss_eps"],
        "train_loss_score": train_stats["loss_score"],
        "train_loss_rank": train_stats["loss_rank"],
        "val_loss": val_stats["loss"],
        "val_loss_intent_lat": val_stats["loss_intent_lat"],
        "val_loss_intent_lon": val_stats["loss_intent_lon"],
        "val_loss_mode": val_stats["loss_mode"],
        "val_loss_anchor": val_stats["loss_anchor"],
        "val_loss_eps": val_stats["loss_eps"],
        "val_loss_score": val_stats["loss_score"],
        "val_loss_rank": val_stats["loss_rank"],
        "val_top1_ade_ft": eval_metrics["top1_ade"],
        "val_top1_fde_ft": eval_metrics["top1_fde"],
        "val_lat_acc": eval_metrics["lat_acc"],
        "val_lon_acc": eval_metrics["lon_acc"],
        "val_joint_acc": eval_metrics["joint_acc"],
        "val_intent_topk_hit": eval_metrics["intent_topk_hit"],
        "val_best_intent_acc": eval_metrics["best_intent_acc"],
        "val_route_hit": eval_metrics["route_hit"],
        "val_route_gap": eval_metrics["route_gap"],
        "score_exec": score_exec,
        "lr": lr,
    }
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def init_csv_log(csv_path):
    fieldnames = [
        "epoch",
        "train_loss",
        "train_loss_intent_lat",
        "train_loss_intent_lon",
        "train_loss_mode",
        "train_loss_anchor",
        "train_loss_eps",
        "train_loss_score",
        "train_loss_rank",
        "val_loss",
        "val_loss_intent_lat",
        "val_loss_intent_lon",
        "val_loss_mode",
        "val_loss_anchor",
        "val_loss_eps",
        "val_loss_score",
        "val_loss_rank",
        "val_top1_ade_ft",
        "val_top1_fde_ft",
        "val_lat_acc",
        "val_lon_acc",
        "val_joint_acc",
        "val_intent_topk_hit",
        "val_best_intent_acc",
        "val_route_hit",
        "val_route_gap",
        "score_exec",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def build_zero_loss_stats():
    return {key: 0.0 for key in LOSS_STAT_KEYS}


def build_zero_eval_metrics():
    return {key: 0.0 for key in VAL_METRIC_KEYS}


def write_tensorboard_log(writer, epoch, train_stats, val_stats, eval_metrics, score_exec, lr):
    for key in LOSS_STAT_KEYS:
        writer.add_scalar(f"LossTrain/{key}", train_stats[key], epoch)
        writer.add_scalar(f"LossVal/{key}", val_stats[key], epoch)
    writer.add_scalar("Eval/top1_ADE_ft", eval_metrics["top1_ade"], epoch)
    writer.add_scalar("Eval/top1_FDE_ft", eval_metrics["top1_fde"], epoch)
    writer.add_scalar("Eval/lat_acc", eval_metrics["lat_acc"], epoch)
    writer.add_scalar("Eval/lon_acc", eval_metrics["lon_acc"], epoch)
    writer.add_scalar("Eval/joint_acc", eval_metrics["joint_acc"], epoch)
    writer.add_scalar("Eval/intent_topk_hit", eval_metrics["intent_topk_hit"], epoch)
    writer.add_scalar("Eval/best_intent_acc", eval_metrics["best_intent_acc"], epoch)
    writer.add_scalar("Eval/route_hit", eval_metrics["route_hit"], epoch)
    writer.add_scalar("Eval/route_gap", eval_metrics["route_gap"], epoch)
    writer.add_scalar("Eval/score_exec", score_exec, epoch)
    writer.add_scalar("LR", lr, epoch)


def print_eval_summary(epoch, total_epochs, train_stats, val_stats, eval_metrics, score_exec):
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train={train_stats['loss']:.6f} | "
        f"lat={train_stats['loss_intent_lat']:.6f} | "
        f"lon={train_stats['loss_intent_lon']:.6f} | "
        f"eps={train_stats['loss_eps']:.6f} | "
        f"val={val_stats['loss']:.6f} | "
        f"top1_ade={eval_metrics['top1_ade']:.4f}ft | "
        f"top1_fde={eval_metrics['top1_fde']:.4f}ft | "
        f"intent_topk={eval_metrics['intent_topk_hit']:.4f} | "
        f"route_hit={eval_metrics['route_hit']:.4f} | "
        f"score={score_exec:.4f}"
    )


def prepare_input_data(batch, feature_dim, device="cuda"):
    hist = batch["hist"]
    va = batch["va"]
    lane = batch["lane"]
    cclass = batch["cclass"]
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    lat_enc = batch["lat_enc"]
    lon_enc = batch["lon_enc"]
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
    lat_enc = lat_enc.to(device)
    lon_enc = lon_enc.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)
    return hist, hist_nbrs, mask, temporal_mask, fut, op_mask, lat_enc, lon_enc


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim):
    model.train()
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    num_batches = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Train", dynamic_ncols=True)

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask, lat_enc, lon_enc = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        loss, loss_parts = model.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_components=True,
            lat_targets=lat_enc,
            lon_targets=lon_enc,
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        flat_loss_parts = flatten_train_parts(loss_parts)
        for key in LOSS_STAT_KEYS:
            totals[key] += flat_loss_parts[key]
        num_batches += 1
        pbar.set_postfix(
            {
                "loss": f"{flat_loss_parts['loss']:.6f}",
                "avg": f"{(totals['loss'] / num_batches):.6f}",
                "lat": f"{(totals['loss_intent_lat'] / num_batches):.6f}",
                "lon": f"{(totals['loss_intent_lon'] / num_batches):.6f}",
                "eps": f"{(totals['loss_eps'] / num_batches):.6f}",
                "anchor": f"{(totals['loss_anchor'] / num_batches):.6f}",
                "score": f"{(totals['loss_score'] / num_batches):.6f}",
            }
        )

    denom = max(num_batches, 1)
    return {key: totals[key] / denom for key in LOSS_STAT_KEYS}


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, feature_dim):
    model.eval()
    totals = {key: 0.0 for key in LOSS_STAT_KEYS}
    metric_totals = {key: 0.0 for key in VAL_METRIC_KEYS}
    num_batches = 0

    if len(dataloader) == 0:
        model.train()
        return build_zero_loss_stats(), build_zero_eval_metrics()

    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Ep{epoch} Val", dynamic_ncols=True)
    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask, lat_enc, lon_enc = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        val_loss, val_parts = model.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_components=True,
            lat_targets=lat_enc,
            lon_targets=lon_enc,
        )
        _, eval_aux = model.forwardEval(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_aux=True,
            lat_targets=lat_enc,
            lon_targets=lon_enc,
        )

        flat_val_parts = flatten_train_parts(val_parts)
        flat_eval_metrics = flatten_eval_aux(eval_aux)
        for key in LOSS_STAT_KEYS:
            totals[key] += flat_val_parts[key]
        for key in VAL_METRIC_KEYS:
            metric_totals[key] += flat_eval_metrics[key]
        num_batches += 1

        pbar.set_postfix(
            {
                "val": f"{(totals['loss'] / num_batches):.6f}",
                "top1_ade": f"{(metric_totals['top1_ade'] / num_batches):.4f}",
                "intent_topk": f"{(metric_totals['intent_topk_hit'] / num_batches):.4f}",
                "route_hit": f"{(metric_totals['route_hit'] / num_batches):.4f}",
            }
        )

    model.train()
    denom = max(num_batches, 1)
    val_stats = {key: totals[key] / denom for key in LOSS_STAT_KEYS}
    eval_metrics = {key: metric_totals[key] / denom for key in VAL_METRIC_KEYS}
    return val_stats, eval_metrics


def save_best_checkpoint(state, checkpoint_dir):
    torch.save(state, checkpoint_dir / "best.pth")


def save_latest_checkpoint(state, checkpoint_dir):
    torch.save(state, checkpoint_dir / "latest.pth")


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
        val_stats, eval_metrics = evaluate(model, val_loader, device, epoch + 1, args.feature_dim)
        score_exec = compute_exec_selection_score(eval_metrics["top1_ade"], eval_metrics["top1_fde"])
        current_lr = optimizer.param_groups[0]["lr"]

        write_csv_log(log_csv_path, epoch + 1, train_stats, val_stats, eval_metrics, score_exec, current_lr)
        write_tensorboard_log(writer, epoch + 1, train_stats, val_stats, eval_metrics, score_exec, current_lr)
        print_eval_summary(epoch + 1, args.num_epochs, train_stats, val_stats, eval_metrics, score_exec)

        scheduler.step()
        is_best = score_exec < best_score
        if is_best:
            best_score = score_exec

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_stats["loss"],
            "eval_metrics": eval_metrics,
            "score_exec": score_exec,
            "best_score": best_score,
            "eval_ade": eval_metrics["top1_ade"],
            "eval_fde": eval_metrics["top1_fde"],
        }

        checkpoint_dir = Path(args.checkpoint_dir)
        save_latest_checkpoint(state, checkpoint_dir)
        if (epoch + 1) % 5 == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch + 1}.pth")
        if is_best:
            save_best_checkpoint(state, checkpoint_dir)

    writer.close()


if __name__ == "__main__":
    main()
