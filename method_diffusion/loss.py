import torch
from torch import nn
from typing import Optional


class DiffusionLoss(nn.Module):

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction 必须为 'mean' 或 'sum'"
        self.reduction = reduction

    def masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred:   [..., T, D]
        target: [..., T, D]
        mask:   [..., T, 1] 或 [..., T]，1 表示参与损失，0 表示忽略
        """
        # 保证形状一致
        assert pred.shape == target.shape, "pred 和 target 形状必须一致"

        if mask is None:
            mask = torch.ones_like(pred[..., :1])  # 默认全部参与

        # 将 mask 扩展到特征维度
        if mask.dim() < pred.dim():
            # [..., T] -> [..., T, 1]
            mask = mask.unsqueeze(-1)
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)

        # 计算平方误差
        mse = (pred - target) ** 2

        # 只在 mask==1 的位置计算
        mse = mse * mask

        if self.reduction == "sum":
            return mse.sum()

        # mean: 除以有效元素个数
        valid = mask.sum().clamp(min=1.0)
        return mse.sum() / valid

    def forward(
        self,
        pred_hist: torch.Tensor,
        gt_hist: torch.Tensor,
        hist_mask: Optional[torch.Tensor] = None,
        pred_nbrs: Optional[torch.Tensor] = None,
        gt_nbrs: Optional[torch.Tensor] = None,
        nbrs_mask: Optional[torch.Tensor] = None,
        lambda_nbrs: float = 1.0,
    ) -> torch.Tensor:
        """
        统一接口:
        - pred_hist / gt_hist: 自车历史轨迹 [B, T, D]
        - pred_nbrs / gt_nbrs: 周车历史轨迹 [B', T, D]
        - *_mask: 对应的掩码 [B, T, 1] / [B', T, 1]
        - lambda_nbrs: 周车损失的权重
        """
        loss = self.masked_mse(pred_hist, gt_hist, hist_mask)

        if (pred_nbrs is not None) and (gt_nbrs is not None):
            loss_nbrs = self.masked_mse(pred_nbrs, gt_nbrs, nbrs_mask)
            loss = loss + lambda_nbrs * loss_nbrs

        return loss