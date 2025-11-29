import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp

"""
参数：
- x (torch.Tensor): 形状 (B, N, D)，输入特征序列。
- shift (torch.Tensor): 形状 (B, D)，平移项，在 token 维度上广播。
- scale (torch.Tensor): 形状 (B, D)，缩放项，在 token 维度上广播。
- only_first (bool): 若为 True，仅对第一个 token 应用变换。

功能：
对输入 x 逐 token 应用仿射变换：x * (1 + scale) + shift。若 only_first 为 True，
仅变换第一个 token，返回与 x 相同形状的 torch.Tensor
实现Dit中的Scale 和 shift 操作
"""
def modulate(x, shift, scale, only_first=False):
    # x [size]: (B, N, D), shift/scale [size]: (B, D)
    # x_first: (B, 1, D), x_rest: (B, N-1, D)
    # only_first 仅对第一个token进行变换
    # cat dim=1 在token维度上拼接
    # 使用unsqueeze(1)将shift/scale从(B, D)变为(B, 1, D)以便广播
    # 在dim=1上拼接变换后的第一个token和未变换的其余tokens
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x

def scale(x, scale, only_first=False):
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
         t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
         t_emb =  self.mlp(t_freq)
         return t_emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning for ego and Cross-Attention.
    """
    def __init__(self, dim=128, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # self.norm3 = nn.LayerNorm(dim)
        # self.cross_attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, y, attn_mask, cross_c=None):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulated_x, modulated_x, modulated_x, key_padding_mask=attn_mask)[0]

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        # x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True)
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        B, P, _ = x.shape

        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x

class DiT(nn.Module):
    def __init__(self, dit_block, final_layer, time_embedder, depth, model_type="x_start"):
        super().__init__()

        assert model_type in ["score", "x_start"], f"Unknown model type: {model_type}"
        self._model_type = model_type
        
        self.blocks = nn.ModuleList([dit_block for i in range(depth)])
        self.final_layer = final_layer
        self.time_embedder = time_embedder

    @property
    def model_type(self):
        return self._model_type

    def forward(self, x, t, cross_c=None, neighbor_current_mask=None):

        B, P, _ = x.shape
        y = self.time_embedder(t)  

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        attn_mask[:, 1:] = neighbor_current_mask[:, 1:] if neighbor_current_mask is not None else True

        for block in self.blocks:
            x = block(x, y, attn_mask, cross_c=cross_c)

        x = self.final_layer(x, y)

        if self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


































