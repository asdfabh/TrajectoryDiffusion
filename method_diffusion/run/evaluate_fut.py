import sys
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.utils.fut_utils import TrajectoryMetrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"


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


def load_checkpoint(model, resume_arg, checkpoint_dir, device):
    ckpt_path = resolve_checkpoint_path(resume_arg, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Fut checkpoint not found: resume_fut={resume_arg}, dir={checkpoint_dir}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    print(f"[FutEval] Loaded checkpoint: {ckpt_path}")
    return model


def print_metric_block(title, top1_metrics, multi_metrics, mode_nll):
    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    print(
        f"Top1 ADE/FDE (m): {top1_metrics['overall_ade_m']:.6f} / {top1_metrics['overall_fde_m']:.6f} | "
        f"minADE@M/minFDE@M (m): {multi_metrics['overall_ade_m']:.6f} / {multi_metrics['overall_fde_m']:.6f} | "
        f"modeNLL: {mode_nll:.6f}"
    )
    print("-" * 90)
    time_pairs = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    print(f"{'Horizon':<8} | {'Top1 ADE(m)':<14} | {'Top1 FDE(m)':<14} | {'minADE@M(m)':<14} | {'minFDE@M(m)':<14}")
    print("-" * 90)
    for label, idx in time_pairs:
        if idx >= len(top1_metrics["ade_prefix_m"]):
            continue
        print(
            f"{label:<8} | "
            f"{top1_metrics['ade_prefix_m'][idx].item():<14.6f} | "
            f"{top1_metrics['de_per_step_m'][idx].item():<14.6f} | "
            f"{multi_metrics['ade_prefix_m'][idx].item():<14.6f} | "
            f"{multi_metrics['de_per_step_m'][idx].item():<14.6f}"
        )
    print("=" * 90)


def print_metrics(metrics, title, metric_name="Future"):
    mode_nll = float(metrics.get("mode_nll", 0.0)) if isinstance(metrics, dict) else 0.0
    top1_metrics = metrics.get("top1", metrics) if isinstance(metrics, dict) else metrics
    multi_metrics = metrics.get("multi", metrics) if isinstance(metrics, dict) else metrics
    print_metric_block(title, top1_metrics, multi_metrics, mode_nll)


def build_test_loader(args):
    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    test_path = data_root / "TestSet.mat"
    if not test_path.exists():
        test_path = data_root / "ValSet.mat"
    print(f"[FutEval] Dataset: {dataset_name}")
    print(f"[FutEval] Test path: {test_path}")
    test_dataset = NgsimDataset(
        str(test_path),
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model, dataloader, device, feature_dim):
    model.eval()
    top1_metrics = TrajectoryMetrics(model.T)
    multi_metrics = TrajectoryMetrics(model.T)
    total_mode_nll = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc="Fut explicit modes", ncols=140)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )

        _, _, aux = model.forwardEvalMulti(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            device,
            return_aux=True,
        )

        top1_metrics.update(aux["top1_pred"], fut, op_mask)
        multi_metrics.update(aux["best_pred"], fut, op_mask)
        total_mode_nll += float(aux["mode_nll"].item())
        num_batches += 1

        top1_summary = top1_metrics.summary()
        multi_summary = multi_metrics.summary()
        pbar.set_postfix(
            {
                "top1_ade_m": f"{top1_summary['overall_ade_m']:.4f}",
                "top1_fde_m": f"{top1_summary['overall_fde_m']:.4f}",
                "minade_m": f"{multi_summary['overall_ade_m']:.4f}",
                "minfde_m": f"{multi_summary['overall_fde_m']:.4f}",
                "mode_nll": f"{(total_mode_nll / num_batches):.4f}",
            }
        )

        if batch_idx % 100 == 0:
            print_metric_block(
                f"Test Iteration {batch_idx} - Future",
                top1_summary,
                multi_summary,
                total_mode_nll / max(num_batches, 1),
            )

    top1_summary = top1_metrics.summary()
    multi_summary = multi_metrics.summary()
    avg_mode_nll = total_mode_nll / max(num_batches, 1)
    print_metric_block("Final Test Result - Future", top1_summary, multi_summary, avg_mode_nll)
    return {
        "top1": top1_summary,
        "multi": multi_summary,
        "mode_nll": avg_mode_nll,
    }


def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[FutEval] Device: {device}")
    print(f"[FutEval] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[FutEval] num_modes={args.num_modes}, num_inference_steps={args.num_inference_steps}")

    test_loader = build_test_loader(args)
    model = DiffusionFut(args).to(device)
    load_checkpoint(model, args.resume_fut, args.checkpoint_dir, device)
    evaluate(model, test_loader, device, args.feature_dim)


if __name__ == "__main__":
    main()
