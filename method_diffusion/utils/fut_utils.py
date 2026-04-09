import inspect

import torch

from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.utils.mask_util import mixed_mask


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


def build_hist_mask(hist, mask_ratio=0.4, random_mask_ratio=0.7, block_mask_start=False):
    """为 future / joint 工具构造历史观测掩码。"""
    return mixed_mask(
        hist,
        p=mask_ratio,
        random_ratio=random_mask_ratio,
        block_start=block_mask_start,
    )

def normalize_traj_valid_mask(valid_mask, pred):
    """将不同形状的 future 有效位掩码统一成 `[B, T]` 浮点张量。"""
    if valid_mask is None:
        return torch.ones(pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask[..., 0]
    return (valid_mask > 0.5).to(pred.device).float()


def compute_batch_ade_fde(pred, target, valid_mask=None):
    """计算单个 batch 的 ADE / FDE。"""
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    valid_mask = normalize_traj_valid_mask(valid_mask, pred_xy)

    diff = pred_xy - target_xy
    dist = torch.norm(diff, dim=-1)
    ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    valid_counts = valid_mask.sum(dim=1).long()
    has_valid = valid_counts > 0
    last_idx = torch.clamp(valid_counts - 1, min=0)
    final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
    return ade, fde


def select_minade_prediction(all_preds, target, valid_mask=None):
    """从多模态预测中选择 minADE 对应轨迹。"""
    target_xy = target[..., :2].unsqueeze(1)
    valid_mask = normalize_traj_valid_mask(valid_mask, all_preds[:, 0]).unsqueeze(1)

    diff = torch.norm(all_preds[..., :2] - target_xy, dim=-1)
    ade_k = (diff * valid_mask).sum(dim=2) / (valid_mask.sum(dim=2) + 1e-6)
    best_idx = torch.argmin(ade_k, dim=1)

    bsz, _, t_len, feat_dim = all_preds.shape
    gather_idx = best_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, feat_dim)
    best_pred = all_preds.gather(1, gather_idx).squeeze(1)
    return best_pred, best_idx, ade_k


class TrajectoryMetrics:
    """
    累计逐时刻 RMSE / FDE，以及逐时刻前缀 ADE。

    口径约定：
    - RMSE/FDE: 只统计当前时刻 t
    - ADE: 统计从起点到当前时刻 t 的前缀平均
    - RMSE 按 TAME 口径，以点数量为分母
    """

    def __init__(self, pred_len, meter_per_unit=0.3048):
        self.pred_len = int(pred_len)
        self.meter_per_unit = float(meter_per_unit)
        self.total_coord_se = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_de = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_counts = torch.zeros(self.pred_len, dtype=torch.float64)

    @staticmethod
    def normalize_valid_mask(valid_mask, pred):
        """将不同形状的有效位掩码统一成 `[B, T]` 浮点张量。"""
        return normalize_traj_valid_mask(valid_mask, pred)

    def update(self, pred, target, valid_mask=None):
        """累积一个 batch 的逐时刻平方误差、位移误差和有效样本数。"""
        pred = pred[:, :self.pred_len, :2]
        target = target[:, :self.pred_len, :2]
        valid_mask = self.normalize_valid_mask(valid_mask, pred)[:, :pred.size(1)]

        diff = pred - target
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq)

        self.total_coord_se += torch.sum(dist_sq * valid_mask, dim=0).double().cpu()
        self.total_de += torch.sum(dist * valid_mask, dim=0).double().cpu()
        self.total_counts += torch.sum(valid_mask, dim=0).double().cpu()

    def summary(self):
        """输出逐时刻 RMSE / FDE 与逐时刻前缀 ADE。"""
        counts = self.total_counts.clamp(min=1.0)
        rmse_per_step_ft = torch.sqrt(self.total_coord_se / counts)
        fde_per_step_ft = self.total_de / counts
        ade_per_step_ft = torch.cumsum(self.total_de, dim=0) / torch.cumsum(self.total_counts, dim=0).clamp(min=1.0)

        return {
            "rmse_per_step_ft": rmse_per_step_ft,
            "rmse_per_step_m": rmse_per_step_ft * self.meter_per_unit,
            "fde_per_step_ft": fde_per_step_ft,
            "fde_per_step_m": fde_per_step_ft * self.meter_per_unit,
            "ade_per_step_ft": ade_per_step_ft,
            "ade_per_step_m": ade_per_step_ft * self.meter_per_unit,
        }
