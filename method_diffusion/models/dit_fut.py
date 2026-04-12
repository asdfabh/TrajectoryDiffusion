import math

import torch
import torch.nn as nn
from timm.layers import Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def timestep_embedding(self, t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class DiT(nn.Module):
    """
    Fut 初版去噪器：
    1. 每个 mode token 直接与 history context 做一次 cross-attention
    2. 经过 LayerNorm + FFN 做逐 mode 特征变换
    3. 用 timestep 做一次调制
    4. 输出每个 mode 的整条 future 修正量
    """

    def __init__(self, hidden_size, output_dim, heads=4, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.output_dim = int(output_dim)

        self.cross_norm = nn.LayerNorm(self.hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            self.hidden_size,
            heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.LayerNorm(self.hidden_size)
        self.ffn = Mlp(
            in_features=self.hidden_size,
            hidden_features=int(self.hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout,
        )

        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True),
        )
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.final_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(self.hidden_size * 4),
            nn.Linear(self.hidden_size * 4, self.output_dim, bias=True),
        )

    def forward(self, x, t_cond, cross):
        cross_attn_in = self.cross_norm(x)
        x = x + self.cross_attn(cross_attn_in, cross, cross)[0]

        x = x + self.ffn(self.ffn_norm(x))

        shift, scale = self.time_modulation(t_cond).chunk(2, dim=1)
        x = modulate(self.out_norm(x), shift, scale)
        return self.final_layer(x)
