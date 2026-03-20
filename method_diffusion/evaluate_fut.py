import sys
import os
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FUT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "fut"
METER_PER_FOOT = 0.3048


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


# 累积测试集上的整体与分时刻轨迹误差指标。
class FutMetrics:
    def __init__(self, pred_len, meter_per_foot=METER_PER_FOOT):
        self.pred_len = int(pred_len)
        self.meter_per_foot = float(meter_per_foot)
        self.total_se = torch.zeros(self.pred_len)
        self.total_de = torch.zeros(self.pred_len)
        self.total_counts = torch.zeros(self.pred_len)
        self.total_dist_sum = 0.0
        self.total_valid_points = 0.0
        self.total_fde_sum = 0.0
        self.total_fde_count = 0.0

    # 用一个 batch 的预测结果更新全局指标统计。
    def update(self, pred, target, op_mask):
        pred = pred[:, :self.pred_len, :2].detach().cpu()
        target = target[:, :self.pred_len, :2].detach().cpu()
        valid_mask = (op_mask[:, :self.pred_len, 0] > 0.5).float().detach().cpu()

        diff = pred - target
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq)

        self.total_se += torch.sum(dist_sq * valid_mask, dim=0)
        self.total_de += torch.sum(dist * valid_mask, dim=0)
        self.total_counts += torch.sum(valid_mask, dim=0)
        self.total_dist_sum += float(torch.sum(dist * valid_mask).item())
        self.total_valid_points += float(torch.sum(valid_mask).item())

        valid_counts = torch.sum(valid_mask, dim=1).long()
        has_valid = valid_counts > 0
        last_idx = torch.clamp(valid_counts - 1, min=0)
        final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        self.total_fde_sum += float(torch.sum(final_dist * has_valid.float()).item())
        self.total_fde_count += float(torch.sum(has_valid.float()).item())

    # 汇总当前累计的整体和分时间步指标。
    def summary(self):
        counts = self.total_counts.clamp(min=1.0)
        rmse_ft = torch.sqrt(self.total_se / counts)
        de_ft = self.total_de / counts
        ade_ft = 0.0 if self.total_valid_points == 0 else self.total_dist_sum / self.total_valid_points
        fde_ft = 0.0 if self.total_fde_count == 0 else self.total_fde_sum / self.total_fde_count
        return {
            "rmse_ft": rmse_ft,
            "rmse_m": rmse_ft * self.meter_per_foot,
            "de_ft": de_ft,
            "de_m": de_ft * self.meter_per_foot,
            "ade_ft": ade_ft,
            "ade_m": ade_ft * self.meter_per_foot,
            "fde_ft": fde_ft,
            "fde_m": fde_ft * self.meter_per_foot,
        }


# 按 TAME 风格打印阶段性评估摘要。
def print_metrics(metrics, title):
    time_indices = [4, 9, 14, 19, 24]
    time_labels = ["1s", "2s", "3s", "4s", "5s"]
    valid_pairs = [(label, idx) for label, idx in zip(time_labels, time_indices) if idx < len(metrics["rmse_ft"])]

    print(f"\n{'=' * 28} {title} {'=' * 28}")
    print(f"Overall ADE: {metrics['ade_m']:.4f} m | {metrics['ade_ft']:.4f} ft")
    print(f"Overall FDE: {metrics['fde_m']:.4f} m | {metrics['fde_ft']:.4f} ft")
    print("-" * 76)

    rmse_line = " | ".join(
        [f"{label}: {metrics['rmse_m'][idx].item():.2f} m / {metrics['rmse_ft'][idx].item():.2f} ft" for label, idx in valid_pairs]
    )
    de_line = " | ".join(
        [f"{label}: {metrics['de_m'][idx].item():.2f} m / {metrics['de_ft'][idx].item():.2f} ft" for label, idx in valid_pairs]
    )

    print(f"RMSE: {rmse_line}")
    print(f"DE:   {de_line}")


# 构建 TestSet dataloader。
def build_test_loader(args):
    data_root = Path(args.data_root)
    test_path = str(data_root / "TestSet.mat")
    # test_path = str(data_root / "ValSet.mat")
    test_dataset = NgsimDataset(
        test_path,
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
    metrics = FutMetrics(model.T)
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
            "ade_ft": f"{summary['ade_ft']:.4f}",
            "fde_ft": f"{summary['fde_ft']:.4f}",
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
