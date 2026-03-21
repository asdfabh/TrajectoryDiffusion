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
# 整理 batch 数据并按特征维度拼接评估输入。
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

# 按 TAME 风格打印阶段性评估摘要。
def print_metrics(metrics, title):
    time_pairs = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    valid_pairs = [(label, idx) for label, idx in time_pairs if idx < len(metrics["rmse_per_step_m"])]

    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    print(f"Average ADE (m): {metrics['overall_ade_m']:.6f}")
    print(f"Average FDE (m): {metrics['overall_fde_m']:.6f}")
    print(f"Average RMSE (m): {metrics['overall_rmse_m']:.6f}")
    print("-" * 75)

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
            f"{metrics['ade_prefix_m'][idx].item():<12.6f} | "
            f"{metrics['de_per_step_m'][idx].item():<12.6f}"
        )
    print("=" * 75)

# 构建 TestSet dataloader。
def build_test_loader(args):
    data_root = Path(args.data_root)
    test_path = data_root / "TestSet.mat"
    # test_path = data_root / "ValSet.mat"
    if not test_path.exists():
        test_path = data_root / "ValSet.mat"
    print(f"[FutEval] Loading test data from: {test_path}")
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

# 执行 TestSet 评估并打印周期性与最终指标。
@torch.no_grad()
def evaluate(model, dataloader, device, feature_dim, num_samples):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model.eval()
    metrics = TrajectoryMetrics(model.T)
    k_samples = max(1, int(num_samples))
    eval_name = f"Fut minADE@{k_samples}" if k_samples > 1 else "Fut single-mode"

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc=eval_name, ncols=120)
    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        if k_samples > 1:
            pred_fut, _, _ = model.forwardEval_minADE(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, K=k_samples)
        else:
            pred_fut, _, _ = model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        metrics.update(pred_fut, fut, op_mask)
        summary = metrics.summary()
        pbar.set_postfix({
            "ade_m": f"{summary['overall_ade_m']:.4f}",
            "fde_m": f"{summary['overall_fde_m']:.4f}",
            "rmse_m": f"{summary['overall_rmse_m']:.4f}",
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
    print(f"[FutEval] num_samples={args.num_samples}, num_inference_steps={args.num_inference_steps}")

    test_loader = build_test_loader(args)
    model = DiffusionFut(args).to(device)
    load_checkpoint(model, args.resume_fut, args.checkpoint_dir, device)
    evaluate(model, test_loader, device, args.feature_dim, args.num_samples)


if __name__ == "__main__":
    main()
