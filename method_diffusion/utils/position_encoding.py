import math
import torch
import torch.nn as nn

class SequentialPositionalEncoding(nn.Module):
    """标准的 Transformer 序列位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x):
        """
        Args:
            x: [B, N, d] - d 可以是任意维度,这里只用来获取 shape
        Returns:
            pos_enc: [B, N, d_model]
        """
        B, N, _ = x.shape
        pos_enc = self.pe[:N, :].unsqueeze(0).expand(B, -1, -1)  # [B, N, d_model]
        return pos_enc
