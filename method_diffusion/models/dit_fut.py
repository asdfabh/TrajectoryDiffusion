import math
import torch
import torch.nn as nn
from timm.layers import Mlp
import copy


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def scale(x, scale):
    return x * (1 + scale.unsqueeze(1))


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
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)

        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )

        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# 优化点 1：全局缓存管理器，用于统一处理 V 和 K 的追加，避免重复计算 RMSNorm
def append_source(sources_v, sources_k, v, eps=1e-6):
    sources_v.append(v)
    # 仅对当前单步新产生的 v 计算 RMSNorm 并缓存
    k = v * torch.rsqrt(v.pow(2).mean(dim=-1, keepdim=True) + eps)
    sources_k.append(k)


class DepthAttentionResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(dim))

    def forward(self, sources_v, sources_k):
        # sources_v, sources_k 均已在外部独立计算完毕
        v = torch.stack(sources_v, dim=0)  # [S, B, T, D]
        k = torch.stack(sources_k, dim=0)  # [S, B, T, D]

        # 优化点 2：使用原生 matmul 替换 einsum，利用底层 cuBLAS 优化
        # k shape: [S, B, T, D], query shape: [D]
        # matmul 会在最后两个维度做点积，结果 shape 自动变为 [S, B, T]
        logits = torch.matmul(k, self.query)

        alpha = torch.softmax(logits.float(), dim=0).to(dtype=v.dtype) # [S, B, T]

        # 优化点 3：使用 element-wise 乘法 + sum 替换 einsum
        # 这种方式内存局部性更好，CUDA 并行效率更高
        out = (alpha.unsqueeze(-1) * v).sum(dim=0)  # [B, T, D]
        return out


class DiTBlock(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.attn_res_cross = DepthAttentionResidual(dim)
        self.attn_res_mlp = DepthAttentionResidual(dim)

    def forward(self, x, sources_v, sources_k, t_cond, cross, attn_mask=None):
        shift_msa, scale_msa, gate_msa = self.adaLN_modulation(t_cond).chunk(3, dim=1)

        # 1. Self-attn
        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        self_update = gate_msa.unsqueeze(1) * self.attn(
            modulated_x,
            modulated_x,
            modulated_x,
            key_padding_mask=attn_mask,
        )[0]

        x = x + self_update
        append_source(sources_v, sources_k, self_update)

        # 2. Cross-attn
        h_cross = self.attn_res_cross(sources_v, sources_k)
        cross_update = self.cross_attn(h_cross, cross, cross)[0]

        x = x + cross_update
        append_source(sources_v, sources_k, cross_update)

        # 3. MLP
        h_mlp = self.attn_res_mlp(sources_v, sources_k)
        mlp_update = self.mlp(self.norm2(h_mlp))

        x = x + mlp_update
        append_source(sources_v, sources_k, mlp_update)

        return x, sources_v, sources_k


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_dim=2):
        super().__init__()
        self.output_dim = output_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, self.output_dim, bias=True),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DiT(nn.Module):
    def __init__(self, dit_block, final_layer, depth):
        super().__init__()
        self.blocks = nn.ModuleList(
            [copy.deepcopy(dit_block) for _ in range(depth)]
        )
        self.final_layer = final_layer
        self.final_attn_res = DepthAttentionResidual(dit_block.dim)
        self.final_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, t_cond, cross):
        # 初始化两个列表，并将原始输入存入
        sources_v = []
        sources_k = []
        append_source(sources_v, sources_k, x)

        for block in self.blocks:
            x, sources_v, sources_k = block(x, sources_v, sources_k, t_cond, cross)

        # Final AttnRes readout
        x_attn = self.final_attn_res(sources_v, sources_k)
        gate = torch.tanh(self.final_gate)
        x = x + gate * x_attn

        x = self.final_layer(x)
        return x