import argparse
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.utils.fut_utils import TrajectoryMetrics, gather_by_index
from method_diffusion.utils.visualization import maybe_visualize_future_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"


def get_eval_args():
    parser = argparse.ArgumentParser("Evaluate future diffusion model", parents=[get_args_parser()])
    parser.add_argument("--enable_vis", default=0, type=int)
    parser.add_argument("--oracle_topk", default=0, type=int)
    return parser.parse_args()


def resolve_checkpoint_path(resume_arg, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg in ("none", "", None):
        resume_arg = "best"
    if Path(str(resume_arg)).exists():
        return Path(str(resume_arg))
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if resume_arg == "latest":
        return checkpoint_dir / "latest.pth"
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


def print_metric_block(title, top1_metrics, multi_metrics, mode_nll, lat_acc=0.0, lon_acc=0.0, joint_acc=0.0, hit_global=0.0, hit_joint_global=0.0):
    line_width = 140
    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    print(
        f"modeNLL: {mode_nll:.6f} | latAcc: {lat_acc:.4f} | lonAcc: {lon_acc:.4f} | "
        f"jointAcc: {joint_acc:.4f} | hitGlobal: {hit_global:.4f} | hitJointGlobal: {hit_joint_global:.4f}"
    )
    print("-" * line_width)
    time_pairs = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    print(
        f"{'Horizon':<8} | {'Top1 ADE(m)':<13} | {'Top1 FDE(m)':<13} | {'Top1 RMSE(m)':<14} | "
        f"{'min ADE(m)':<13} | {'min FDE(m)':<13} | {'min RMSE(m)':<14}"
    )
    print("-" * line_width)
    for label, idx in time_pairs:
        if idx >= len(top1_metrics["rmse_per_step_m"]):
            continue
        print(
            f"{label:<8} | "
            f"{top1_metrics['ade_prefix_m'][idx].item():<13.6f} | "
            f"{top1_metrics['de_per_step_m'][idx].item():<13.6f} | "
            f"{top1_metrics['rmse_per_step_m'][idx].item():<14.6f} | "
            f"{multi_metrics['ade_prefix_m'][idx].item():<13.6f} | "
            f"{multi_metrics['de_per_step_m'][idx].item():<13.6f} | "
            f"{multi_metrics['rmse_per_step_m'][idx].item():<14.6f}"
        )
    print("=" * line_width)


def print_metrics(metrics, title, metric_name="Future"):
    top1_metrics = metrics.get("top1", metrics) if isinstance(metrics, dict) else metrics
    multi_metrics = metrics.get("multi", metrics) if isinstance(metrics, dict) else metrics
    print_metric_block(
        title,
        top1_metrics,
        multi_metrics,
        float(metrics.get("mode_nll", 0.0)),
        float(metrics.get("lat_acc", 0.0)),
        float(metrics.get("lon_acc", 0.0)),
        float(metrics.get("joint_acc", 0.0)),
        float(metrics.get("hit_global", 0.0)),
        float(metrics.get("hit_joint_global", 0.0)),
    )


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


def maybe_run_visualization(args, model, hist, hist_nbrs, mask, temporal_mask, fut, op_mask, lat_enc, lon_enc, top1_pred, eval_aux, all_outputs):
    if int(args.enable_vis) <= 0:
        return

    joint_idx = (torch.argmax(lat_enc, dim=1) * lon_enc.size(1) + torch.argmax(lon_enc, dim=1)).long()
    routing = eval_aux["routing"]
    selected_mode_idx = routing["selected_mode_idx"]
    best_mode_idx = all_outputs["best_mode_idx"]
    selected_matches = all_outputs["mode_indices"] == selected_mode_idx.unsqueeze(1)
    selected_mode_pos = torch.argmax(selected_matches.long(), dim=1)
    selected_mode_pos = torch.where(
        selected_matches.any(dim=1),
        selected_mode_pos,
        torch.full_like(selected_mode_pos, -1),
    )
    oracle_joint_idx = torch.div(best_mode_idx, model.num_submodes, rounding_mode="floor")
    oracle_sub_idx = torch.remainder(best_mode_idx, model.num_submodes)
    selected_hit_global = (selected_mode_idx == best_mode_idx).long()

    maybe_visualize_future_prediction(
        hist=hist,
        hist_nbrs=hist_nbrs,
        temporal_mask=temporal_mask,
        future=fut,
        pred=top1_pred,
        valid_mask=(op_mask[:, :, 0] > 0),
        stage="eval",
        enable_eval_vis=True,
        pred_all=all_outputs["all_pred_phys"],
        pred_best_idx=all_outputs["best_mode_pos"],
        pred_selected_idx=selected_mode_pos,
        anchor_all=all_outputs["anchor_pos_phys"],
        intent_probs={
            "lat": routing["lat_probs"],
            "lon": routing["lon_probs"],
            "joint": routing["joint_probs"],
        },
        intent_meta={
            "pred_joint_idx": routing["pred_joint_idx"],
            "gt_joint_idx": joint_idx,
            "oracle_joint_idx": oracle_joint_idx,
            "routed_sub_idx": routing["routed_sub_idx"],
            "best_sub_idx": oracle_sub_idx,
            "best_sub_label": "best_sub(global)",
            "num_submodes": torch.full_like(joint_idx, model.num_submodes),
            "selected_hit_global": selected_hit_global,
        },
    )


@torch.no_grad()
def evaluate(model, dataloader, device, feature_dim, args):
    model.eval()
    top1_metrics = TrajectoryMetrics(model.T)
    multi_metrics = TrajectoryMetrics(model.T)
    total_mode_nll = 0.0
    total_lat_correct = 0.0
    total_lon_correct = 0.0
    total_joint_correct = 0.0
    total_hit_global = 0.0
    total_hit_joint_global = 0.0
    total_samples = 0.0
    num_batches = 0

    select_topk = None if int(args.oracle_topk) <= 0 else min(int(args.oracle_topk), model.num_modes)
    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc="Fut evaluate", ncols=140)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask, lat_enc, lon_enc = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        joint_idx = (torch.argmax(lat_enc, dim=1) * lon_enc.size(1) + torch.argmax(lon_enc, dim=1)).long()

        top1_pred, eval_aux = model.forwardEval(
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

        all_outputs = model.getAllModePredictions(
            hist=hist,
            hist_nbrs=hist_nbrs,
            mask=mask,
            temporal_mask=temporal_mask,
            future=fut,
            op_mask=op_mask,
            device=device,
            select_topk=select_topk,
        )

        oracle_global = gather_by_index(all_outputs["all_pred_phys"], all_outputs["best_mode_pos"])
        routing = eval_aux["routing"]
        selected_mode_idx = routing["selected_mode_idx"]
        pred_joint_idx = routing["pred_joint_idx"]
        oracle_joint_idx = torch.div(all_outputs["best_mode_idx"], model.num_submodes, rounding_mode="floor")
        hit_global = (selected_mode_idx == all_outputs["best_mode_idx"]).float()
        hit_joint_global = (pred_joint_idx == oracle_joint_idx).float()

        top1_metrics.update(top1_pred, fut, op_mask)
        multi_metrics.update(oracle_global, fut, op_mask)
        total_mode_nll += float(all_outputs["mode_nll"].item())
        total_lat_correct += float((torch.argmax(routing["lat_probs"], dim=1) == torch.argmax(lat_enc, dim=1)).sum().item())
        total_lon_correct += float((torch.argmax(routing["lon_probs"], dim=1) == torch.argmax(lon_enc, dim=1)).sum().item())
        total_joint_correct += float((torch.argmax(routing["joint_probs"], dim=1) == joint_idx).sum().item())
        total_hit_global += float(hit_global.sum().item())
        total_hit_joint_global += float(hit_joint_global.sum().item())
        total_samples += float(lat_enc.size(0))
        num_batches += 1

        maybe_run_visualization(
            args=args,
            model=model,
            hist=hist,
            hist_nbrs=hist_nbrs,
            mask=mask,
            temporal_mask=temporal_mask,
            fut=fut,
            op_mask=op_mask,
            lat_enc=lat_enc,
            lon_enc=lon_enc,
            top1_pred=top1_pred,
            eval_aux=eval_aux,
            all_outputs=all_outputs,
        )

        top1_summary = top1_metrics.summary()
        multi_summary = multi_metrics.summary()
        avg_lat_acc = total_lat_correct / max(total_samples, 1.0)
        avg_lon_acc = total_lon_correct / max(total_samples, 1.0)
        avg_joint_acc = total_joint_correct / max(total_samples, 1.0)
        avg_hit_global = total_hit_global / max(total_samples, 1.0)
        avg_hit_joint_global = total_hit_joint_global / max(total_samples, 1.0)
        pbar.set_postfix(
            {
                "top1_ade_m": f"{top1_summary['overall_ade_m']:.4f}",
                "top1_fde_m": f"{top1_summary['overall_fde_m']:.4f}",
                "top1_rmse_m": f"{top1_summary['overall_rmse_m']:.4f}",
                "minade_m": f"{multi_summary['overall_ade_m']:.4f}",
                "minfde_m": f"{multi_summary['overall_fde_m']:.4f}",
                "gap_m": f"{(top1_summary['overall_ade_m'] - multi_summary['overall_ade_m']):.4f}",
                "lat_acc": f"{avg_lat_acc:.4f}",
                "lon_acc": f"{avg_lon_acc:.4f}",
                "joint_acc": f"{avg_joint_acc:.4f}",
                "hit_global": f"{avg_hit_global:.4f}",
                "hit_joint_global": f"{avg_hit_joint_global:.4f}",
                "mode_nll": f"{(total_mode_nll / num_batches):.4f}",
            }
        )

        if batch_idx % 100 == 0:
            print_metric_block(
                f"Test Iteration {batch_idx} - Future",
                top1_summary,
                multi_summary,
                total_mode_nll / max(num_batches, 1),
                avg_lat_acc,
                avg_lon_acc,
                avg_joint_acc,
                avg_hit_global,
                avg_hit_joint_global,
            )

    top1_summary = top1_metrics.summary()
    multi_summary = multi_metrics.summary()
    avg_mode_nll = total_mode_nll / max(num_batches, 1)
    avg_lat_acc = total_lat_correct / max(total_samples, 1.0)
    avg_lon_acc = total_lon_correct / max(total_samples, 1.0)
    avg_joint_acc = total_joint_correct / max(total_samples, 1.0)
    avg_hit_global = total_hit_global / max(total_samples, 1.0)
    avg_hit_joint_global = total_hit_joint_global / max(total_samples, 1.0)
    print_metric_block("Final Test Result - Future", top1_summary, multi_summary, avg_mode_nll, avg_lat_acc, avg_lon_acc, avg_joint_acc, avg_hit_global, avg_hit_joint_global)
    return {
        "top1": top1_summary,
        "multi": multi_summary,
        "mode_nll": avg_mode_nll,
        "lat_acc": avg_lat_acc,
        "lon_acc": avg_lon_acc,
        "joint_acc": avg_joint_acc,
        "hit_global": avg_hit_global,
        "hit_joint_global": avg_hit_joint_global,
    }


def main():
    args = get_eval_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionFut(args).to(device)

    print(f"[FutEval] Device: {device}")
    print(f"[FutEval] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[FutEval] num_modes={model.num_modes}, num_inference_steps={args.num_inference_steps}")

    test_loader = build_test_loader(args)
    load_checkpoint(model, args.resume_fut, args.checkpoint_dir, device)
    evaluate(model, test_loader, device, args.feature_dim, args)


if __name__ == "__main__":
    main()
