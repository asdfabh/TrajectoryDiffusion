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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
LOSS_STAT_KEYS = [
    "loss",
    "loss_intent_lat",
    "loss_intent_lon",
    "loss_mode",
    "loss_anchor",
    "loss_div",
    "loss_eps",
    "loss_x0",
    "loss_end",
]
VAL_METRIC_KEYS = [
    "top1_ade",
    "top1_fde",
    "minade_m",
    "minfde_m",
    "mode_nll",
]


def compute_selection_score(minade_m, minfde_m, mode_nll=0.0):
    return float(minade_m) + 0.5 * float(minfde_m) + 0.1 * float(mode_nll)


def compute_exec_selection_score(top1_ade, top1_fde):
    return float(top1_ade) + 0.5 * float(top1_fde)


def compute_joint_selection_score(top1_ade, top1_fde, minade_m, minfde_m, mode_nll=0.0):
    return 0.5 * compute_exec_selection_score(top1_ade, top1_fde) + 0.5 * compute_selection_score(minade_m, minfde_m, mode_nll)


def build_selection_scores(eval_metrics):
    return {
        "multi": compute_selection_score(eval_metrics["minade_m"], eval_metrics["minfde_m"], eval_metrics["mode_nll"]),
        "exec": compute_exec_selection_score(eval_metrics["top1_ade"], eval_metrics["top1_fde"]),
        "joint": compute_joint_selection_score(
            eval_metrics["top1_ade"],
            eval_metrics["top1_fde"],
            eval_metrics["minade_m"],
            eval_metrics["minfde_m"],
            eval_metrics["mode_nll"],
        ),
    }


def resolve_resume_checkpoint(resume_arg, checkpoint_dir):
    if resume_arg in ("none", "", None):
        return None
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if re.fullmatch(r"epoch_\d+", str(resume_arg)):
        return checkpoint_dir / f"{resume_arg}.pth"
    print(f"[FutModel] Unsupported resume_fut='{resume_arg}', expected 'best' or 'epoch_i'.")
    return None


def load_checkpoint(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_scores = {"multi": float("inf"), "exec": float("inf"), "joint": float("inf")}
    ckpt_path = resolve_resume_checkpoint(args.resume_fut, Path(args.checkpoint_dir))

    if ckpt_path is not None:
        if not ckpt_path.exists():
            print(f"[FutModel] Checkpoint not found: {ckpt_path}")
            return start_epoch, best_scores

        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)

        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception:
            pass
        start_epoch = int(state.get("epoch", 0))
        stored_best_scores = state.get("best_scores", None)
        if isinstance(stored_best_scores, dict):
            for key in best_scores:
                if key in stored_best_scores:
                    best_scores[key] = float(stored_best_scores[key])
        elif "best_score" in state:
            best_scores[str(getattr(args, "save_best_metric", "multi")).strip().lower()] = float(state["best_score"])
        elif "eval_ade" in state and "eval_fde" in state:
            best_scores["exec"] = compute_exec_selection_score(state["eval_ade"], state["eval_fde"])
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_scores


def write_csv_log(csv_path, epoch, train_stats, val_stats, eval_metrics, selection_scores, lr):
    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_loss_intent_lat": train_stats["loss_intent_lat"],
        "train_loss_intent_lon": train_stats["loss_intent_lon"],
        "train_loss_mode": train_stats["loss_mode"],
        "train_loss_anchor": train_stats["loss_anchor"],
        "train_loss_div": train_stats["loss_div"],
        "train_loss_eps": train_stats["loss_eps"],
        "train_loss_x0": train_stats["loss_x0"],
        "train_loss_end": train_stats["loss_end"],
        "val_loss": val_stats["loss"],
        "val_loss_intent_lat": val_stats["loss_intent_lat"],
        "val_loss_intent_lon": val_stats["loss_intent_lon"],
        "val_loss_mode": val_stats["loss_mode"],
        "val_loss_anchor": val_stats["loss_anchor"],
        "val_loss_div": val_stats["loss_div"],
        "val_loss_eps": val_stats["loss_eps"],
        "val_loss_x0": val_stats["loss_x0"],
        "val_loss_end": val_stats["loss_end"],
        "val_top1_ade_ft": eval_metrics["top1_ade"],
        "val_top1_fde_ft": eval_metrics["top1_fde"],
        "val_minade_m_ft": eval_metrics["minade_m"],
        "val_minfde_m_ft": eval_metrics["minfde_m"],
        "val_mode_nll": eval_metrics["mode_nll"],
        "score_multi": selection_scores["multi"],
        "score_exec": selection_scores["exec"],
        "score_joint": selection_scores["joint"],
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
        "train_loss_div",
        "train_loss_eps",
        "train_loss_x0",
        "train_loss_end",
        "val_loss",
        "val_loss_intent_lat",
        "val_loss_intent_lon",
        "val_loss_mode",
        "val_loss_anchor",
        "val_loss_div",
        "val_loss_eps",
        "val_loss_x0",
        "val_loss_end",
        "val_top1_ade_ft",
        "val_top1_fde_ft",
        "val_minade_m_ft",
        "val_minfde_m_ft",
        "val_mode_nll",
        "score_multi",
        "score_exec",
        "score_joint",
        "lr",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def build_zero_loss_stats():
    return {key: 0.0 for key in LOSS_STAT_KEYS}


def build_zero_eval_metrics():
    return {key: 0.0 for key in VAL_METRIC_KEYS}


def write_tensorboard_log(writer, epoch, train_stats, val_stats, eval_metrics, selection_scores, lr):
    for key in LOSS_STAT_KEYS:
        writer.add_scalar(f"LossTrain/{key}", train_stats[key], epoch)
        writer.add_scalar(f"LossVal/{key}", val_stats[key], epoch)
    writer.add_scalar("Eval/top1_ADE_ft", eval_metrics["top1_ade"], epoch)
    writer.add_scalar("Eval/top1_FDE_ft", eval_metrics["top1_fde"], epoch)
    writer.add_scalar("Eval/minADE_M_ft", eval_metrics["minade_m"], epoch)
    writer.add_scalar("Eval/minFDE_M_ft", eval_metrics["minfde_m"], epoch)
    writer.add_scalar("Eval/mode_nll", eval_metrics["mode_nll"], epoch)
    writer.add_scalar("Eval/score_multi", selection_scores["multi"], epoch)
    writer.add_scalar("Eval/score_exec", selection_scores["exec"], epoch)
    writer.add_scalar("Eval/score_joint", selection_scores["joint"], epoch)
    writer.add_scalar("LR", lr, epoch)


def print_eval_summary(epoch, total_epochs, train_stats, val_stats, eval_metrics, selection_scores):
    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train={train_stats['loss']:.6f} | "
        f"train_eps={train_stats['loss_eps']:.6f} | "
        f"val={val_stats['loss']:.6f} | "
        f"val_eps={val_stats['loss_eps']:.6f} | "
        f"top1_ade={eval_metrics['top1_ade']:.4f}ft | "
        f"top1_fde={eval_metrics['top1_fde']:.4f}ft | "
        f"minade@M={eval_metrics['minade_m']:.4f}ft | "
        f"minfde@M={eval_metrics['minfde_m']:.4f}ft | "
        f"mode_nll={eval_metrics['mode_nll']:.4f} | "
        f"score={selection_scores['multi']:.4f}"
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

        totals["loss"] += float(loss.item())
        for key in LOSS_STAT_KEYS:
            if key == "loss":
                continue
            totals[key] += float(loss_parts[key].item())
        num_batches += 1
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.6f}",
                "avg_loss": f"{(totals['loss'] / num_batches):.6f}",
                "eps": f"{(totals['loss_eps'] / num_batches):.6f}",
                "lat": f"{(totals['loss_intent_lat'] / num_batches):.6f}",
                "lon": f"{(totals['loss_intent_lon'] / num_batches):.6f}",
                "mode": f"{(totals['loss_mode'] / num_batches):.6f}",
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
        _, _, eval_aux = model.forwardEvalMulti(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_aux=True,
        )

        totals["loss"] += float(val_loss.item())
        for key in LOSS_STAT_KEYS:
            if key == "loss":
                continue
            totals[key] += float(val_parts[key].item())
        metric_totals["top1_ade"] += float(eval_aux["top1_ade"].item())
        metric_totals["top1_fde"] += float(eval_aux["top1_fde"].item())
        metric_totals["minade_m"] += float(eval_aux["minade_m"].item())
        metric_totals["minfde_m"] += float(eval_aux["minfde_m"].item())
        metric_totals["mode_nll"] += float(eval_aux["mode_nll"].item())
        num_batches += 1

        pbar.set_postfix(
            {
                "val_loss": f"{(totals['loss'] / num_batches):.6f}",
                "top1_ade": f"{(metric_totals['top1_ade'] / num_batches):.4f}",
                "minade@M": f"{(metric_totals['minade_m'] / num_batches):.4f}",
                "mode_nll": f"{(metric_totals['mode_nll'] / num_batches):.4f}",
            }
        )

    model.train()
    denom = max(num_batches, 1)
    val_stats = {key: totals[key] / denom for key in LOSS_STAT_KEYS}
    eval_metrics = {key: metric_totals[key] / denom for key in VAL_METRIC_KEYS}
    return val_stats, eval_metrics


def save_checkpoint_aliases(state, checkpoint_dir, best_flags, selected_metric):
    if best_flags["multi"]:
        torch.save(state, checkpoint_dir / "best_multi.pth")
    if best_flags["exec"]:
        torch.save(state, checkpoint_dir / "best_exec.pth")
    if best_flags["joint"]:
        torch.save(state, checkpoint_dir / "best_joint.pth")

    selected_name = {
        "multi": "best_multi.pth",
        "exec": "best_exec.pth",
        "joint": "best_joint.pth",
    }[selected_metric]
    selected_path = checkpoint_dir / selected_name
    if selected_path.exists():
        torch.save(torch.load(selected_path, map_location="cpu"), checkpoint_dir / "best.pth")


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
    start_epoch, best_scores = load_checkpoint(args, model, optimizer, scheduler, device)
    selected_metric = str(getattr(args, "save_best_metric", "multi")).strip().lower()

    for epoch in range(start_epoch, args.num_epochs):
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim)
        val_stats, eval_metrics = evaluate(model, val_loader, device, epoch + 1, args.feature_dim)
        selection_scores = build_selection_scores(eval_metrics)
        current_lr = optimizer.param_groups[0]["lr"]

        write_csv_log(log_csv_path, epoch + 1, train_stats, val_stats, eval_metrics, selection_scores, current_lr)
        write_tensorboard_log(writer, epoch + 1, train_stats, val_stats, eval_metrics, selection_scores, current_lr)
        print_eval_summary(epoch + 1, args.num_epochs, train_stats, val_stats, eval_metrics, selection_scores)

        scheduler.step()
        best_flags = {}
        for key, score in selection_scores.items():
            is_best = score < best_scores[key]
            best_flags[key] = is_best
            if is_best:
                best_scores[key] = score

        state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_stats["loss"],
            "eval_metrics": eval_metrics,
            "selection_scores": selection_scores,
            "best_scores": best_scores,
            "best_score": best_scores[selected_metric],
            "eval_ade": eval_metrics["top1_ade"],
            "eval_fde": eval_metrics["top1_fde"],
        }

        checkpoint_dir = Path(args.checkpoint_dir)
        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, checkpoint_dir / f"epoch_{epoch + 1}.pth")
        save_checkpoint_aliases(state, checkpoint_dir, best_flags, selected_metric)

    writer.close()


if __name__ == "__main__":
    main()
