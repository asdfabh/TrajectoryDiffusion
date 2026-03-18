import inspect

import torch

from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.utils.mask_util import continuous_mask, random_mask


def build_ngsim_dataset(mat_path, args):
    """按数据集构造函数签名过滤参数，避免不同版本接口不兼容。"""
    dataset_kwargs = {
        "t_h": 30,
        "t_f": 50,
        "d_s": 2,
        "enc_size": args.encoder_input_dim,
        "feature_dim": args.feature_dim,
        "future_only": True,
    }
    sig = inspect.signature(NgsimDataset.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return NgsimDataset(mat_path, **dataset_kwargs)
    filtered = {k: v for k, v in dataset_kwargs.items() if k in sig.parameters}
    return NgsimDataset(mat_path, **filtered)


def build_hist_mask(hist, mask_type="random", mask_prob=0.4):
    """为 joint 评估构造历史观测掩码。"""
    if mask_type == "random":
        return random_mask(hist, p=mask_prob)
    if mask_type == "block":
        return continuous_mask(hist, p=mask_prob)
    return random_mask(hist, p=mask_prob)


def prepare_fut_batch(batch, feature_dim, device="cuda", include_hist_mask=False, mask_type="random", mask_prob=0.4):
    """整理 future 分支所需的 batch 字段。"""
    if int(feature_dim) != 4:
        raise ValueError("future 分支当前仅支持 feature_dim=4: [rel_x, rel_y, v, a]。")

    hist = torch.cat((batch["hist"], batch["va"]), dim=-1).to(device)
    hist_nbrs = torch.cat((batch["nbrs"], batch["nbrs_va"]), dim=-1).to(device)

    prepared = {
        "hist": hist,
        "hist_nbrs": hist_nbrs,
        "fut": batch["fut"].to(device),
        "op_mask": batch["op_mask"].to(device),
        "mask": batch["mask"].to(device),
        "temporal_mask": batch["temporal_mask"].to(device),
        "extras": {
            "ego_lane": batch["lane"].to(device),
            "nbr_lane": batch["nbrs_lane"].to(device),
            "nbr_dist": batch["nbrs_distance"].to(device),
            "lat_gt": batch["lat_enc"].argmax(dim=-1).long().to(device),
            "lon_gt": batch["lon_enc"].argmax(dim=-1).long().to(device),
        },
    }

    if include_hist_mask:
        hist_mask = build_hist_mask(hist, mask_type=mask_type, mask_prob=mask_prob).to(device)
        hist_masked = torch.cat((hist_mask * hist, hist_mask), dim=-1).to(device)
        prepared["hist_mask"] = hist_mask
        prepared["hist_masked"] = hist_masked

    return prepared


class TrajectoryMetrics:
    """累计 future 轨迹的 RMSE / DE / ADE / FDE 统计量。"""

    def __init__(self, pred_len, meter_per_unit=0.3048):
        """初始化固定预测长度的指标容器。"""
        self.pred_len = int(pred_len)
        self.meter_per_unit = float(meter_per_unit)
        self.total_se = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_de = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_counts = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_dist_sum = 0.0
        self.total_valid_points = 0.0
        self.total_fde_sum = 0.0
        self.total_fde_count = 0.0

    @staticmethod
    def normalize_valid_mask(valid_mask, pred):
        """将不同形状的有效位掩码统一成 `[B, T]` 浮点张量。"""
        if valid_mask is None:
            return torch.ones(pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask[..., 0]
        return (valid_mask > 0.5).to(pred.device).float()

    def update(self, pred, target, valid_mask=None):
        """累积一个 batch 的轨迹误差。"""
        pred = pred[:, :self.pred_len, :2]
        target = target[:, :self.pred_len, :2]
        valid_mask = self.normalize_valid_mask(valid_mask, pred)[:, :pred.size(1)]

        diff = pred - target
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq)

        self.total_se += torch.sum(dist_sq * valid_mask, dim=0).double().cpu()
        self.total_de += torch.sum(dist * valid_mask, dim=0).double().cpu()
        self.total_counts += torch.sum(valid_mask, dim=0).double().cpu()
        self.total_dist_sum += float(torch.sum(dist * valid_mask).item())
        self.total_valid_points += float(torch.sum(valid_mask).item())

        t_idx = torch.arange(dist.size(1), device=dist.device).unsqueeze(0).expand_as(dist)
        masked_idx = torch.where(valid_mask > 0, t_idx, t_idx.new_full(t_idx.shape, -1))
        last_idx = masked_idx.max(dim=1).values
        has_valid = last_idx >= 0
        final_dist = dist.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        self.total_fde_sum += float(torch.sum(final_dist * has_valid.float()).item())
        self.total_fde_count += float(torch.sum(has_valid.float()).item())

    def summary(self):
        """输出累计后的标量和逐时刻指标。"""
        counts = self.total_counts.clamp(min=1.0)
        rmse_per_step_ft = torch.sqrt(self.total_se / counts)
        de_per_step_ft = self.total_de / counts
        cumsum_de = torch.cumsum(self.total_de, dim=0)
        cumsum_counts = torch.cumsum(self.total_counts, dim=0).clamp(min=1.0)
        ade_prefix_ft = cumsum_de / cumsum_counts

        overall_ade_ft = 0.0 if self.total_valid_points == 0 else self.total_dist_sum / self.total_valid_points
        overall_fde_ft = 0.0 if self.total_fde_count == 0 else self.total_fde_sum / self.total_fde_count
        total_counts = self.total_counts.sum().clamp(min=1.0)
        overall_rmse_ft = 0.0 if self.total_valid_points == 0 else float(torch.sqrt(self.total_se.sum() / total_counts).item())

        return {
            "rmse_per_step_ft": rmse_per_step_ft,
            "rmse_per_step_m": rmse_per_step_ft * self.meter_per_unit,
            "de_per_step_ft": de_per_step_ft,
            "de_per_step_m": de_per_step_ft * self.meter_per_unit,
            "ade_prefix_ft": ade_prefix_ft,
            "ade_prefix_m": ade_prefix_ft * self.meter_per_unit,
            "overall_ade_ft": overall_ade_ft,
            "overall_fde_ft": overall_fde_ft,
            "overall_rmse_ft": overall_rmse_ft,
            "overall_ade_m": overall_ade_ft * self.meter_per_unit,
            "overall_fde_m": overall_fde_ft * self.meter_per_unit,
            "overall_rmse_m": overall_rmse_ft * self.meter_per_unit,
        }


def format_timestep_metrics(metric_tensor, meter_per_unit=0.3048, time_step_labels=None):
    """将逐时刻指标格式化为便于日志输出的字符串。"""
    labels = time_step_labels or [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    values = []
    for label, t_idx in labels:
        if t_idx < metric_tensor.numel():
            val_ft = float(metric_tensor[t_idx].item())
            values.append(f"{label}: {val_ft:.3f} ft ({val_ft * meter_per_unit:.3f} m)")
    return " | ".join(values) if values else "no valid timestep"
