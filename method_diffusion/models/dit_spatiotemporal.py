import torch
import torch.nn as nn
from timm.models.layers import Mlp


def modulate(x, shift, scale):
    """AdaLN 调制: x * (1 + scale) + shift"""
    scale = torch.clamp(scale, -5, 5)
    shift = torch.clamp(shift, -5, 5)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SpatioTemporalDiTBlock(nn.Module):
    def __init__(self, dim=128, heads=6, dropout=0.1, mlp_ratio=4.0, N=40, T=16):
        super().__init__()
        self.N = N
        self.T = T
        self.dim = dim

        # 时间注意力
        self.norm_temporal = nn.LayerNorm(dim)
        self.attn_temporal = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        # 空间注意力
        self.norm_spatial = nn.LayerNorm(dim)
        self.attn_spatial = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=dropout
        )

        # AdaLN 调制参数
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        # ✅ 使用 xavier 初始化而非零初始化
        nn.init.xavier_uniform_(self.adaLN_modulation[-1].weight, gain=0.02)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, y, attn_mask=None):
        """
        Args:
            x: [B, N*T, dim]
            y: [B, dim]
            attn_mask: [B, N*T] - True 表示无效位置
        """
        B = x.shape[0]

        # AdaLN 参数生成
        shift_t, scale_t, gate_t, shift_s, scale_s, gate_s = \
            self.adaLN_modulation(y).chunk(6, dim=1)

        # ========== 时间注意力 ==========
        x_temporal = x.view(B, self.N, self.T, self.dim)
        x_temporal = x_temporal.transpose(1, 2).contiguous()
        x_temporal = x_temporal.view(B * self.T, self.N, self.dim)

        # 扩展 AdaLN 参数
        shift_t_exp = shift_t.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)
        scale_t_exp = scale_t.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)
        gate_t_exp = gate_t.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)

        # ✅ 关键修复: 正确处理掩码
        if attn_mask is not None:
            # attn_mask: [B, N*T] → [B*T, N]
            temporal_mask = attn_mask.view(B, self.N, self.T)  # [B, N, T]
            temporal_mask = temporal_mask.transpose(1, 2).contiguous()  # [B, T, N]
            temporal_mask = temporal_mask.view(B * self.T, self.N)  # [B*T, N]

            # ✅ 检查是否存在全 True 的行 (会导致注意力全为 NaN)
            all_masked = temporal_mask.all(dim=1)
            if all_masked.any():
                # 将全掩码的行设为全 False (允许自注意力)
                temporal_mask[all_masked] = False
        else:
            temporal_mask = None

        # AdaLN 调制
        x_temporal_normed = self.norm_temporal(x_temporal)
        x_temporal_mod = modulate(x_temporal_normed, shift_t_exp, scale_t_exp)

        # ✅ 添加数值稳定性检查
        if torch.isnan(x_temporal_mod).any():
            print(f"⚠️ x_temporal_mod contains NaN before attention")
            print(f"shift_t_exp range: [{shift_t_exp.min():.4f}, {shift_t_exp.max():.4f}]")
            print(f"scale_t_exp range: [{scale_t_exp.min():.4f}, {scale_t_exp.max():.4f}]")
            x_temporal_mod = torch.nan_to_num(x_temporal_mod, nan=0.0)

        # 注意力计算
        try:
            attn_out, attn_weights = self.attn_temporal(
                x_temporal_mod, x_temporal_mod, x_temporal_mod,
                key_padding_mask=temporal_mask,
                need_weights=True
            )
        except RuntimeError as e:
            print(f"❌ Temporal attention error: {e}")
            print(f"temporal_mask shape: {temporal_mask.shape if temporal_mask is not None else None}")
            print(f"x_temporal_mod shape: {x_temporal_mod.shape}")
            raise

        # ✅ 检查并修复 NaN
        if torch.isnan(attn_out).any():
            print(f"⚠️ Temporal attention output contains NaN, replacing with zeros")
            print(f"attn_weights range: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        # 门控残差
        gate_t_exp = torch.clamp(gate_t_exp, -1, 1)  # ✅ 限制门控范围
        x_temporal = x_temporal + gate_t_exp.unsqueeze(1) * attn_out

        # 恢复形状
        x_temporal = x_temporal.view(B, self.T, self.N, self.dim)
        x_temporal = x_temporal.transpose(1, 2).contiguous()
        x = x_temporal.view(B, self.N * self.T, self.dim)

        # ========== 空间注意力 ==========
        x_spatial = x.view(B, self.N, self.T, self.dim)
        x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()
        x_spatial = x_spatial.view(B * self.T, self.N, self.dim)

        shift_s_exp = shift_s.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)
        scale_s_exp = scale_s.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)
        gate_s_exp = gate_s.unsqueeze(1).repeat(1, self.T, 1).view(B * self.T, self.dim)

        if attn_mask is not None:
            # attn_mask: [B, N*T] → [B*T, N]
            spatial_mask = attn_mask.view(B, self.N, self.T)  # [B, N, T]
            spatial_mask = spatial_mask.permute(0, 2, 1).contiguous()  # [B, T, N]
            spatial_mask = spatial_mask.view(B * self.T, self.N)  # [B*T, N]

            # ✅ 检查全掩码行
            all_masked = spatial_mask.all(dim=1)
            if all_masked.any():
                spatial_mask[all_masked] = False
        else:
            spatial_mask = None

        x_spatial_normed = self.norm_spatial(x_spatial)
        x_spatial_mod = modulate(x_spatial_normed, shift_s_exp, scale_s_exp)

        if torch.isnan(x_spatial_mod).any():
            x_spatial_mod = torch.nan_to_num(x_spatial_mod, nan=0.0)

        attn_out, _ = self.attn_spatial(
            x_spatial_mod, x_spatial_mod, x_spatial_mod,
            key_padding_mask=spatial_mask
        )

        if torch.isnan(attn_out).any():
            attn_out = torch.nan_to_num(attn_out, nan=0.0)

        gate_s_exp = torch.clamp(gate_s_exp, -1, 1)
        x_spatial = x_spatial + gate_s_exp.unsqueeze(1) * attn_out

        x_spatial = x_spatial.view(B, self.T, self.N, self.dim)
        x_spatial = x_spatial.permute(0, 2, 1, 3).contiguous()
        x = x_spatial.view(B, self.N * self.T, self.dim)

        # ========== MLP ==========
        x = x + self.mlp(self.norm_mlp(x))

        # ✅ 最终检查
        if torch.isnan(x).any():
            print(f"⚠️ Block output contains NaN, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        return x
