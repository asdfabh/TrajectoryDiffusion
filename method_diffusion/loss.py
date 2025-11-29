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
            mask = mask.unsqueeze(-1)
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)

        mse = (pred - target) ** 2

        # 只在 mask==1 的位置计算
        mse = mse * mask

        if self.reduction == "sum":
            return mse.sum()

        valid = mask.sum().clamp(min=1.0)
        return mse.sum() / valid

    def forward(self, pred, gt, mask):
        loss = self.masked_mse(pred, gt, mask)
        return loss