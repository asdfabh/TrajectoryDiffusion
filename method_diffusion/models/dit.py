import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
import copy

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

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
        t_emb = self.mlp(t_freq)
        return t_emb

class MixerBlock(nn.Module):
    def __init__(self, tokens_dim, channels_dim, drop_path_rate=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels_dim)
        self.token_mlp = Mlp(in_features=tokens_dim, hidden_features=tokens_dim, act_layer=nn.GELU,
                             drop=drop_path_rate)
        self.norm2 = nn.LayerNorm(channels_dim)
        self.channel_mlp = Mlp(in_features=channels_dim, hidden_features=channels_dim, act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, x):
        # x: [B*N, T, D]
        y = self.norm1(x)
        y = y.transpose(1, 2)  # [B*N, D, T]
        y = self.token_mlp(y)
        y = y.transpose(1, 2)  # [B*N, T, D]
        x = x + y
        # Mix Channels (Features)
        y = self.norm2(x)
        y = self.channel_mlp(y)
        x = x + y
        return x

class RelativePositionBias(nn.Module):
    """
    将相对坐标 (dx, dy) 映射为 Attention 的 Bias Map
    Input: [B, N, N, 2]
    Output: [B, Heads, N, N]
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        # 使用 MLP 学习几何关系到注意力权重的映射
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads)
        )

    def forward(self, rel_pos_matrix):
        # [B, N, N, 2] -> [B, N, N, Heads]
        bias = self.mlp(rel_pos_matrix)
        # [B, N, N, Heads] -> [B, Heads, N, N]
        return bias.permute(0, 3, 1, 2)

class DiTBlockFut(nn.Module):
    """
    极简版 Block: [B, N, H]
    1. Spatial Interaction (多车交互)
    2. FFN (处理单车的时间/动力学特征)
    """

    def __init__(self, dim, heads, dropout=0.1, mlp_ratio=4.0):
        super().__init__()
        self.dim = dim

        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = SpatialInteractionLayer(dim, heads, dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=dropout)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, t_emb, rel_pos_matrix, agent_mask=None):
        B, N, D = x.shape

        shift_s, scale_s, gate_s, shift_m, scale_m, gate_m = self.adaLN_modulation(t_emb).chunk(6, dim=1)

        gate_s = gate_s.unsqueeze(1)
        gate_m = gate_m.unsqueeze(1)

        x_norm = modulate(self.norm1(x), shift_s, scale_s)
        x_attn = self.spatial_attn(x_norm, rel_pos_matrix, agent_mask)
        x = x + gate_s * x_attn

        x_norm = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m * self.mlp(x_norm)

        return x

class SpatialInteractionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rel_pos_bias = RelativePositionBias(num_heads)

    def forward(self, x, rel_pos_matrix, agent_mask=None):
        # x: [B, N, D]
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, Heads, N, Head_Dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        spatial_bias = self.rel_pos_bias(rel_pos_matrix)  # [B, Heads, N, N]
        attn = attn + spatial_bias

        if agent_mask is not None:
            mask_expanded = agent_mask.view(B, 1, 1, N).expand(-1, self.num_heads, N, -1)
            attn = attn.masked_fill(~mask_expanded.bool(), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class DiTFut(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, depth, mlp_ratio,
                 N, T, time_embedder):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.N = N
        self.T = T
        self.input_flat_dim = T * 2
        self.output_flat_dim = T * 2
        self.time_embedder = time_embedder

        # Input Projection: [B, N, T*2] -> [B, N, H]
        self.input_proj = nn.Linear(self.input_flat_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            DiTBlockFut(hidden_dim, heads, dropout, mlp_ratio)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_dim)
        # Output Projection: [B, N, H] -> [B, N, T*2]
        self.final_layer = nn.Linear(hidden_dim, self.output_flat_dim)

    def forward(self, x, t, hist_feat, rel_pos_matrix, agents_current_pos=None, agent_mask=None):
        # x: [B, N, T*2] (Flattened)
        t_emb = self.time_embedder(t)
        x = self.input_proj(x)

        x = x + hist_feat  # [B, N, H] + [B, N, H] 直接相加

        for block in self.blocks:
            x = block(x, t_emb, rel_pos_matrix, agent_mask)

        x = self.final_norm(x)
        x = self.final_layer(x)

        return x

# class DiTBlockFut(nn.Module):
#     """
#     SOTA 结构:
#     1. Temporal Attention (处理自身历史/未来一致性)
#     2. Spatial Interaction (处理车车交互，带相对位置Bias)
#     3. FFN
#     """
#
#     def __init__(self, dim, heads, dropout=0.1, mlp_ratio=4.0):
#         super().__init__()
#         self.dim = dim
#
#         self.norm1 = nn.LayerNorm(dim)
#         self.attn_temporal = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
#
#         self.spatial_interaction = SpatialInteractionLayer(dim, heads, dropout)
#
#         self.norm2 = nn.LayerNorm(dim)
#         self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=dropout)
#
#         # AdaLN Modulation (Conditioning on t and History)
#         self.adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(dim, 6 * dim, bias=True)  # shift/scale/gate for Temp, shift/scale/gate for MLP
#         )
#
#     def forward(self, x, t_emb, rel_pos_matrix, agent_mask=None):
#         """
#         x: [B, T, N, D] (注意这里保持 T, N 分开)
#         t_emb: [B, D]
#         rel_pos_matrix: [B, N, N, 2]
#         """
#         B, T, N, D = x.shape
#
#         # AdaLN params
#         shift_t, scale_t, gate_t, shift_m, scale_m, gate_m = self.adaLN_modulation(t_emb).chunk(6, dim=1)
#
#         # --- 1. Temporal Attention (Batch = B * N) ---
#         x_temp = x.permute(0, 2, 1, 3).reshape(B * N, T, D)  # [B*N, T, D]
#
#         # Modulate
#         # expand shift/scale to [B*N, D]
#         shift_t_ex = shift_t.repeat_interleave(N, dim=0)
#         scale_t_ex = scale_t.repeat_interleave(N, dim=0)
#         gate_t_ex = gate_t.repeat_interleave(N, dim=0)
#
#         residual = x_temp
#         x_norm = self.norm1(x_temp)
#         x_norm = modulate(x_norm, shift_t_ex, scale_t_ex)
#
#         x_attn, _ = self.attn_temporal(x_norm, x_norm, x_norm)
#         x_temp = residual + gate_t_ex.unsqueeze(1) * x_attn
#
#         x = x_temp.view(B, N, T, D).permute(0, 2, 1, 3)  # [B, T, N, D]
#
#         # --- 2. Spatial Interaction (Batch = B * T) ---
#         # 这个 Layer 内部处理了 Res+Norm，直接调用
#         x = self.spatial_interaction(x, rel_pos_matrix, agent_mask)
#
#         # --- 3. FFN ---
#         x_mlp = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
#
#         shift_m_ex = shift_m.repeat_interleave(N, dim=0)
#         scale_m_ex = scale_m.repeat_interleave(N, dim=0)
#         gate_m_ex = gate_m.repeat_interleave(N, dim=0)
#
#         residual = x_mlp
#         x_norm = self.norm2(x_mlp)
#         x_norm = modulate(x_norm, shift_m_ex, scale_m_ex)
#
#         x_out = self.mlp(x_norm)
#         x_mlp = residual + gate_m_ex.unsqueeze(1) * x_out
#
#         x = x_mlp.view(B, N, T, D).permute(0, 2, 1, 3)  # Back to [B, T, N, D]
#
#         return x
#
# class DiTFut(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim,
#                  heads, dropout, depth, mlp_ratio,
#                  N, T, time_embedder):
#         super().__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.N = N
#         self.T = T
#         self.time_embedder = time_embedder
#
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#
#         self.hist_proj = nn.Linear(hidden_dim, hidden_dim)
#
#         self.blocks = nn.ModuleList([
#             DiTBlockFut(hidden_dim, heads, dropout, mlp_ratio)
#             for _ in range(depth)
#         ])
#
#         self.final_norm = nn.LayerNorm(hidden_dim)
#         self.final_layer = nn.Linear(hidden_dim, output_dim)
#
#         self.init_mlp = nn.Sequential(
#             nn.Linear(2, hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )
#
#     def forward(self, x, t, hist_feat, rel_pos_matrix, agents_current_pos, agent_mask=None):
#
#         B, T, N, _ = x.shape
#         t_emb = self.time_embedder(t)  # [B, D_emb]
#         x = self.input_proj(x)  # [B, T, N, H]
#         x = x + hist_feat.unsqueeze(1)
#
#         # init_emb = self.init_mlp(agents_current_pos)  # [B, N, H]
#         # x = x + init_emb.unsqueeze(1)
#
#         for block in self.blocks:
#             x = block(x, t_emb, rel_pos_matrix, agent_mask)
#
#         x = self.final_norm(x)
#         x = self.final_layer(x)  # [B, T, N, 2]
#
#         return x

# class SpatialInteractionLayer(nn.Module):
#     def __init__(self, d_model, num_heads, dropout=0.1):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = d_model // num_heads
#         self.scale = self.head_dim ** -0.5
#
#         self.qkv = nn.Linear(d_model, d_model * 3)
#         self.proj = nn.Linear(d_model, d_model)
#         self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#
#         self.rel_pos_bias = RelativePositionBias(num_heads)
#
#     def forward(self, x, rel_pos_matrix, agent_mask=None):
#         """
#         x: [B, T, N, D] - 我们将对 N 维度进行交互，T 维度视为 Batch 的一部分或独立处理
#         rel_pos_matrix: [B, N, N, 2]
#         agent_mask: [B, N]
#         """
#         B, T, N, C = x.shape
#
#         x_flat = x.reshape(B * T, N, C)
#
#         shortcut = x_flat
#         x_flat = self.norm(x_flat)
#
#         qkv = self.qkv(x_flat).reshape(B * T, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # [B*T, Heads, N, Head_Dim]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*T, Heads, N, N]
#
#         spatial_bias = self.rel_pos_bias(rel_pos_matrix)
#
#         spatial_bias = spatial_bias.unsqueeze(1).expand(-1, T, -1, -1, -1).reshape(B * T, self.num_heads, N, N)
#
#         attn = attn + spatial_bias
#
#         if agent_mask is not None:
#             mask_expanded = agent_mask.unsqueeze(1).expand(-1, T, -1).reshape(B * T, N)  # [B*T, N]
#             mask_expanded = mask_expanded.view(B * T, 1, 1, N).expand(-1, self.num_heads, N, -1)
#             attn = attn.masked_fill(~mask_expanded.bool(), float("-inf"))
#
#         attn = attn.softmax(dim=-1)
#         attn = self.dropout(attn)
#
#         x_out = (attn @ v).transpose(1, 2).reshape(B * T, N, C)
#         x_out = self.proj(x_out)
#
#         x_out = shortcut + x_out
#
#         return x_out.view(B, T, N, C)

