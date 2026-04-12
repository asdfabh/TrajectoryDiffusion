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


def gen_sineembed_for_position(pos_tensor, hidden_dim=64):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos