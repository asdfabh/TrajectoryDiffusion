import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.run.evaluate import (
    HIST_CHECKPOINT_DIR,
    prepare_input_data as prepare_hist_input_data,
)
from method_diffusion.run.evaluate_fut import (
    build_test_loader,
    load_checkpoint as load_fut_checkpoint,
    print_metrics,
)
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.run.train_joint import (
    JOINT_FUT_CHECKPOINT_DIR,
    JOINT_HIST_CHECKPOINT_DIR,
    load_hist_checkpoint,
    normalize_dataset_name,
)
from method_diffusion.utils.fut_utils import TrajectoryMetrics, select_closest_prediction
from method_diffusion.utils.visualization import visualize_scene_prediction


def get_eval_args():
    return get_args_parser().parse_args()


def resolve_hist_checkpoint_dir(resume_hist, dataset_name):
    resume_path = Path(str(resume_hist))
    if resume_path.exists():
        return resume_path.parent

    for base_dir in [JOINT_HIST_CHECKPOINT_DIR, HIST_CHECKPOINT_DIR]:
        candidate = base_dir / dataset_name / "checkpoint_best.pth"
        if candidate.exists():
            return candidate.parent
    return HIST_CHECKPOINT_DIR / dataset_name


@torch.no_grad()
def evaluate(model_hist, model_fut, dataloader, device, feature_dim, fut_k, enable_eval_vis, mask_ratio, random_mask_ratio, block_mask_start, dataset_name=None, enable_latent_bridge=False):
    model_hist.eval()
    model_fut.eval()
    metrics = TrajectoryMetrics(model_fut.T)
    k_samples = max(1, int(fut_k))
    eval_name = f"Joint Fut ClosestGT-RMSE@{k_samples}" if k_samples > 1 else "Joint Fut single-mode"

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=eval_name, ncols=140)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(
            batch,
            feature_dim,
            device=device,
        )
        _, hist_masked, _ = prepare_hist_input_data(
            batch,
            feature_dim,
            mask_ratio=mask_ratio,
            random_mask_ratio=random_mask_ratio,
            block_mask_start=block_mask_start,
            device=device,
        )
        hist_outputs = model_hist.forward_eval(hist, hist_masked, device, return_tokens=enable_latent_bridge)
        if enable_latent_bridge:
            _, pred_hist, past_latent_tokens = hist_outputs
        else:
            _, pred_hist = hist_outputs
            past_latent_tokens = None

        if k_samples > 1:
            all_preds = model_fut.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, device, K=k_samples, past_latent_tokens=past_latent_tokens)
            pred_fut, best_idx, _ = select_closest_prediction(all_preds, fut, op_mask)
        else:
            all_preds = model_fut.forwardEvalMulti(pred_hist, hist_nbrs, mask, temporal_mask, device, K=1, past_latent_tokens=past_latent_tokens)
            pred_fut = all_preds.squeeze(1)
            best_idx = None

        if enable_eval_vis:
            visualize_scene_prediction(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=fut,
                pred=pred_fut,
                valid_mask=(op_mask[..., 0] > 0.5).float(),
                pred_all=all_preds,
                pred_best_idx=best_idx,
                title="Joint Hist Reconstruction + Future Prediction",
                highlight_label="Best",
                hist_masked=hist_masked,
                hist_reconstructed=pred_hist,
                dataset_name=dataset_name,
            )

        metrics.update(pred_fut, fut, op_mask)
        summary = metrics.summary()
        last_idx = min(model_fut.T, len(summary["rmse_per_step_m"])) - 1
        last_sec = int(model_fut.T * 0.2)
        pbar.set_postfix({
            f"ade_{last_sec}s": f"{summary['ade_per_step_m'][last_idx]:.4f}",
            f"fde_{last_sec}s": f"{summary['fde_per_step_m'][last_idx]:.4f}",
            f"rmse_{last_sec}s": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
            f"theta_{last_sec}s": f"{summary['theta_mae_per_step_deg'][last_idx]:.4f}",
            f"v_{last_sec}s": f"{summary['v_mae_per_step_mps'][last_idx]:.4f}",
        })

        if batch_idx % 100 == 0:
            print_metrics(summary, f"Joint Test Iteration {batch_idx}")

    final_metrics = metrics.summary()
    print_metrics(final_metrics, "Joint Final Test Result")
    return final_metrics


def main():
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = normalize_dataset_name(args.dataset)
    hist_checkpoint_dir = resolve_hist_checkpoint_dir(args.resume_hist, dataset_name)
    fut_checkpoint_dir = JOINT_FUT_CHECKPOINT_DIR / dataset_name

    print(f"[JointEval] Device: {device}")
    print(f"[JointEval] Hist checkpoint dir: {hist_checkpoint_dir}")
    print(f"[JointEval] Fut checkpoint dir: {fut_checkpoint_dir}")
    print(
        f"[JointEval] fut_k={args.fut_k}, num_inference_steps={args.num_inference_steps}, "
        f"latent_bridge={int(int(args.enable_past_fut_latent_bridge) > 0)}"
    )

    test_loader = build_test_loader(args)
    model_hist = DiffusionPast(args).to(device)
    model_fut = DiffusionFut(args).to(device)
    load_hist_checkpoint(model_hist, args.resume_hist, [hist_checkpoint_dir], device, trainable=False, dataset_name=dataset_name)
    load_fut_checkpoint(model_fut, args.resume_fut, fut_checkpoint_dir, device)
    evaluate(
        model_hist=model_hist,
        model_fut=model_fut,
        dataloader=test_loader,
        device=device,
        feature_dim=args.feature_dim,
        fut_k=args.fut_k,
        enable_eval_vis=int(args.fut_enable_eval_vis) > 0,
        mask_ratio=max(0.0, min(1.0, float(args.mask_prob))),
        random_mask_ratio=max(0.0, min(1.0, float(args.random_mask_ratio))),
        block_mask_start=int(args.block_mask_start) > 0,
        dataset_name=args.dataset,
        enable_latent_bridge=int(args.enable_past_fut_latent_bridge) > 0,
    )


if __name__ == "__main__":
    main()
