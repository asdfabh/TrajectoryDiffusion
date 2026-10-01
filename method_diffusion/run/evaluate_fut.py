import sys
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.build import build_trajectory_dataset, get_raw_dt, get_test_split_path, get_time_params
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.trajectory_refiner import build_trajectory_refiner
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.utils.fut_utils import (
    SampleImprovementStats,
    TrajectoryMetrics,
    print_sample_improvement,
    get_horizon_pairs,
    print_trajectory_metrics,
    select_closest_prediction,
)
from method_diffusion.utils.trajectory_kinematics import PhysicalDiagnostics, print_kinematic_diagnostics
from method_diffusion.utils.visualization import visualize_scene_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
REFINER_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "refine"


def get_refiner_checkpoint_dir(dataset_name):
    return REFINER_CHECKPOINT_DIR / str(dataset_name).strip().lower()


# 解析 fut checkpoint 标识并返回实际文件路径。
def resolve_checkpoint_path(resume_arg, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg in ("none", "", None):
        resume_arg = "best"
    resume_arg = str(resume_arg)
    resume_path = Path(resume_arg)
    if resume_path.exists():
        return resume_path
    if resume_arg == "best":
        return checkpoint_dir / "best.pth"
    if re.fullmatch(r"epoch_\d+", resume_arg):
        return checkpoint_dir / f"{resume_arg}.pth"
    return None


# 加载 fut 模型参数并切换到评估模式。
def load_checkpoint(model, resume_arg, checkpoint_dir, device):
    ckpt_path = resolve_checkpoint_path(resume_arg, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Fut checkpoint not found: resume_fut={resume_arg}, dir={checkpoint_dir}. "
            "Use 'none', 'best', 'epoch_i' such as 'epoch_10', or an existing path."
        )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    print(f"[FutEval] Loaded checkpoint: {ckpt_path}")
    return model


def load_residual_refiner(args, checkpoint_dir, device):
    ckpt_path = resolve_checkpoint_path(args.fut_refiner_checkpoint, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"Residual refiner checkpoint not found: fut_refiner_checkpoint={args.fut_refiner_checkpoint}, "
            f"dir={checkpoint_dir}. Use 'none', 'best', 'epoch_i' such as 'epoch_10', or an existing path."
        )

    refiner = build_trajectory_refiner(args).to(device)
    state = torch.load(ckpt_path, map_location=device)
    refiner.load_state_dict(state["model_state_dict"], strict=True)
    refiner.eval()
    print(f"[FutEval] Loaded residual refiner checkpoint: {ckpt_path}")
    print("[FutEval] Refiner: TABR-temporal-basis")
    return refiner


def get_time_pairs(fut_steps, dt=0.2):
    return get_horizon_pairs(fut_steps, dt)


def print_metrics(metrics, title, dt=0.2):
    print_trajectory_metrics(metrics, title, dt)


# 构建 TestSet dataloader。
def build_test_loader(args):
    dataset_name = str(args.dataset).lower()
    test_path = get_test_split_path(args, dataset_name)
    print(f"[FutEval] Dataset: {dataset_name}")
    print(f"[FutEval] Test path: {test_path}")
    test_dataset = build_trajectory_dataset(test_path, dataset_name, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)

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


# 执行 TestSet 评估并打印周期性与最终指标。
@torch.no_grad()
def evaluate(model, dataloader, device, feature_dim, fut_k, enable_eval_vis, fut_vis_enable_refine, dataset_name, residual_refiner):
    model.eval()
    baseline_metrics = TrajectoryMetrics(model.T, num_candidates=max(1, int(fut_k)))
    refined_metrics = TrajectoryMetrics(model.T, num_candidates=max(1, int(fut_k))) if residual_refiner is not None else None
    k_samples = max(1, int(fut_k))
    _, _, d_s, _ = get_time_params(dataset_name)
    fut_dt = get_raw_dt(dataset_name) * int(d_s)
    baseline_physics = PhysicalDiagnostics(fut_dt)
    refined_physics = PhysicalDiagnostics(fut_dt) if residual_refiner is not None else None
    sample_stats = SampleImprovementStats() if residual_refiner is not None else None
    eval_name = f"Fut BestOfK@{k_samples}" if k_samples > 1 else "Fut single-mode"
    print(f"[FutEval] dt={fut_dt:.3f}s | refine={int(residual_refiner is not None)}")

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=eval_name, ncols=120)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)

        if k_samples > 1:
            all_preds = model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device=device, K=k_samples)
            pred_fut, best_idx, _ = select_closest_prediction(all_preds, fut, op_mask)
            refined_all_preds = None
            refined_pred_fut = None
            refined_best_idx = None
            if residual_refiner is not None:
                refined_all_preds, _ = residual_refiner(hist, all_preds, fut_dt)
                refined_pred_fut, refined_best_idx, _ = select_closest_prediction(refined_all_preds, fut, op_mask)
            if enable_eval_vis:
                # 旧的 diffusion 过程可视化已停用，这里仅保留最终预测结果可视化。
                show_refined_vis = int(fut_vis_enable_refine) > 0 and refined_pred_fut is not None
                visualize_scene_prediction(
                    hist=hist,
                    hist_nbrs=hist_nbrs,
                    temporal_mask=temporal_mask,
                    future=fut,
                    pred=pred_fut,
                    valid_mask=(op_mask[..., 0] > 0.5).float(),
                    pred_all=all_preds,
                    pred_best_idx=best_idx,
                    refined_pred=refined_pred_fut if show_refined_vis else None,
                    refined_pred_all=refined_all_preds if show_refined_vis else None,
                    refined_pred_best_idx=refined_best_idx if show_refined_vis else None,
                    batch_idx=0,
                    title="Future Prediction: Raw + Refined" if show_refined_vis else "Future Prediction",
                    highlight_label="Best RMSE (full trajectory)",
                    dataset_name=dataset_name,
                )
        else:
            all_preds = model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, device=device, K=1)
            pred_fut = all_preds.squeeze(1)
            refined_pred_fut = None
            if residual_refiner is not None:
                refined_all_preds, _ = residual_refiner(hist, all_preds, fut_dt)
                refined_pred_fut = refined_all_preds[:, 0]

        baseline_metrics.update(all_preds, fut, op_mask)
        baseline_physics.update(pred_fut, op_mask)
        if residual_refiner is not None:
            refined_metrics.update(refined_all_preds, fut, op_mask)
            refined_physics.update(refined_pred_fut, op_mask)
            sample_stats.update(all_preds, refined_all_preds, fut, op_mask)
            summary = refined_metrics.summary()
        else:
            summary = baseline_metrics.summary()
        last_idx = min(model.T, len(summary["rmse_per_step_m"])) - 1
        last_sec = int(round(model.T * fut_dt))
        pbar.set_postfix({
            f"minADE_{last_sec}s": f"{summary['min_ade_per_step_m'][last_idx]:.4f}",
            f"minFDE_{last_sec}s": f"{summary['min_fde_per_step_m'][last_idx]:.4f}",
            f"rmse_{last_sec}s": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
            f"theta_{last_sec}s": f"{summary['theta_mae_per_step_deg'][last_idx]:.4f}",
            f"v_{last_sec}s": f"{summary['v_mae_per_step_mps'][last_idx]:.4f}",
        })

        if batch_idx % 100 == 0:
            print_metrics(baseline_metrics.summary(), f"Baseline Test Iteration {batch_idx}", dt=fut_dt)
            if residual_refiner is not None:
                print_metrics(refined_metrics.summary(), f"TABR-temporal-basis-refiner Test Iteration {batch_idx}", dt=fut_dt)
    baseline_final_metrics = baseline_metrics.summary()
    print_metrics(baseline_final_metrics, "Baseline Final Test Result", dt=fut_dt)
    print_kinematic_diagnostics(baseline_physics.summary(), "Baseline Physical Diagnostics")
    if residual_refiner is None:
        return baseline_final_metrics
    refined_final_metrics = refined_metrics.summary()
    postprocess_title = "TABR-temporal-basis-refiner"
    print_metrics(refined_final_metrics, f"{postprocess_title} Final Test Result", dt=fut_dt)
    print_kinematic_diagnostics(refined_physics.summary(), f"{postprocess_title} Physical Diagnostics")
    print_sample_improvement(sample_stats.summary(), f"{postprocess_title} Sample Improvement")
    return refined_final_metrics


# 初始化模型、数据与 checkpoint，并执行 fut 测试评估。
def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR / str(args.dataset).strip().lower())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[FutEval] Device: {device}")
    print(f"[FutEval] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[FutEval] fut_k={args.fut_k}, num_inference_steps={args.num_inference_steps}")

    test_loader = build_test_loader(args)
    model = DiffusionFut(args).to(device)
    load_checkpoint(model, args.resume_fut, args.checkpoint_dir, device)
    residual_refiner = None
    if int(args.enable_refine) > 0:
        residual_refiner = load_residual_refiner(args, get_refiner_checkpoint_dir(args.dataset), device)
    evaluate(model, test_loader, device, args.feature_dim, args.fut_k, enable_eval_vis=int(args.fut_enable_eval_vis) > 0,
        fut_vis_enable_refine=args.fut_vis_enable_refine, dataset_name=args.dataset, residual_refiner=residual_refiner)

if __name__ == "__main__":
    main()
