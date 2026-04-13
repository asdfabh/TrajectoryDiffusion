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
from method_diffusion.run.train_fut import prepare_input_data
from method_diffusion.utils.fut_utils import TrajectoryMetrics, normalize_traj_valid_mask, select_minade_prediction
from method_diffusion.utils.visualization import maybe_visualize_future_prediction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
METER_PER_FOOT = 0.3048


# 解析 fut checkpoint 标识并返回实际文件路径。
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


# 加载 fut 模型参数并切换到评估模式。
def load_checkpoint(model, resume_arg, checkpoint_dir, device):
    ckpt_path = resolve_checkpoint_path(resume_arg, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"Fut checkpoint not found: resume_fut={resume_arg}, dir={checkpoint_dir}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    print(f"[FutEval] Loaded checkpoint: {ckpt_path}")
    return model


def print_metrics(metrics, title):
    time_pairs = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    valid_pairs = [(label, idx) for label, idx in time_pairs if idx < len(metrics["rmse_per_step_m"])]

    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    if not valid_pairs:
        print("No valid per-second horizon in current prediction length.")
        print("=" * 75)
        return

    print(f"{'Horizon':<8} | {'RMSE (m)':<12} | {'ADE (m)':<12} | {'FDE (m)':<12}")
    print("-" * 75)
    for label, idx in valid_pairs:
        print(
            f"{label:<8} | "
            f"{metrics['rmse_per_step_m'][idx].item():<12.6f} | "
            f"{metrics['ade_per_step_m'][idx].item():<12.6f} | "
            f"{metrics['fde_per_step_m'][idx].item():<12.6f}"
        )
    print("=" * 75)


# 构建 TestSet dataloader。
def build_test_loader(args):
    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    test_path = data_root / "TestSet.mat"
    # test_path = data_root / "ValSet.mat"
    if not test_path.exists():
        test_path = data_root / "ValSet.mat"
    print(f"[FutEval] Dataset: {dataset_name}")
    print(f"[FutEval] Test path: {test_path}")
    test_dataset = NgsimDataset(str(test_path), t_h=30, t_f=50, d_s=2, enc_size=args.encoder_input_dim, feature_dim=args.feature_dim)

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
def evaluate(model, dataloader, device, feature_dim, fut_k, enable_eval_vis):
    model.eval()
    metrics = TrajectoryMetrics(model.T)
    k_samples = max(1, int(fut_k))
    eval_name = f"Fut minADE@{k_samples}" if k_samples > 1 else "Fut single-mode"

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=eval_name, ncols=120)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)

        if k_samples > 1:
            all_preds = model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, fut, device, K=k_samples)
            pred_fut, best_idx, _ = select_minade_prediction(all_preds, fut, op_mask)
            if enable_eval_vis:
                maybe_visualize_future_prediction(
                    hist=hist,
                    hist_nbrs=hist_nbrs,
                    temporal_mask=temporal_mask,
                    future=fut,
                    pred=pred_fut,
                    valid_mask=normalize_traj_valid_mask(op_mask, pred_fut),
                    pred_all=all_preds,
                    pred_best_idx=best_idx,
                    meter_per_foot=METER_PER_FOOT,
                )
        else:
            pred_fut = model.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, fut, device, K=1).squeeze(1)

        metrics.update(pred_fut, fut, op_mask)
        summary = metrics.summary()
        last_idx = min(model.T, len(summary["rmse_per_step_m"])) - 1
        pbar.set_postfix({
            "ade_5s": f"{summary['ade_per_step_m'][last_idx]:.4f}",
            "fde_5s": f"{summary['fde_per_step_m'][last_idx]:.4f}",
            "rmse_5s": f"{summary['rmse_per_step_m'][last_idx]:.4f}",
        })

        if batch_idx % 100 == 0:
            print_metrics(summary, f"Test Iteration {batch_idx}")

    final_metrics = metrics.summary()
    print_metrics(final_metrics, "Final Test Result")
    return final_metrics


# 初始化模型、数据与 checkpoint，并执行 fut 测试评估。
def main():
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(FUT_CHECKPOINT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[FutEval] Device: {device}")
    print(f"[FutEval] Checkpoint dir: {args.checkpoint_dir}")
    print(f"[FutEval] fut_k={args.fut_k}, num_inference_steps={args.num_inference_steps}")

    test_loader = build_test_loader(args)
    model = DiffusionFut(args).to(device)
    load_checkpoint(model, args.resume_fut, args.checkpoint_dir, device)
    evaluate(model, test_loader, device, args.feature_dim, args.fut_k, enable_eval_vis=int(args.fut_enable_eval_vis) > 0)


if __name__ == "__main__":
    main()
