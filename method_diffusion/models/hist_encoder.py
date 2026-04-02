import math

import torch
import torch.nn as nn
from einops import repeat
from torch.nn import functional as F

from method_diffusion.models.transformer import TransformerEncoder, TransformerEncoderLayer, Residual


class PositionalEncodingSine(nn.Module):

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2)


def build_position_encoding(args):
    return PositionalEncodingSine(args.encoder_input_dim // 2, temperature=10000)


class HistEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.context_dim = int(args.hidden_dim_fut)

        self.input_embedding = nn.Linear(args.feature_dim, args.encoder_input_dim)
        self.position_encoding = build_position_encoding(args)

        encoder_layer = TransformerEncoderLayer(
            args.encoder_input_dim,
            args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=0.1,
            activation=args.activation,
        )
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)

        self.temporal_q = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_k = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_v = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_residual = Residual(args.encoder_input_dim)

        self.qf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.kf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.vf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(self.context_dim),
            nn.Linear(self.context_dim, self.context_dim),
            nn.GELU(),
            nn.Linear(self.context_dim, self.context_dim),
        )

    def forward(self, src, nbrs, mask, temporal_mask):
        bsz, steps, _ = src.shape
        num_heads = int(self.args.attn_nhead)
        head_dim = int(self.args.attn_out)

        src_t = src.permute(1, 0, 2).contiguous()
        nbrs_t = nbrs.permute(1, 0, 2).contiguous()

        temporal_mask = temporal_mask.view(
            temporal_mask.size(0),
            temporal_mask.size(1) * temporal_mask.size(2),
            temporal_mask.size(3),
        ).bool()
        temporal_occ = temporal_mask.any(dim=-1)
        temporal_mask = repeat(temporal_mask, "b c n -> t b c n", t=steps)
        temporal_grid = torch.zeros_like(temporal_mask, dtype=nbrs_t.dtype, device=nbrs_t.device)
        temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs_t)

        temporal_query = self.temporal_q(src_t)
        temporal_query = torch.cat(
            torch.split(torch.unsqueeze(temporal_query, dim=2), head_dim, dim=-1),
            dim=0,
        ).permute(1, 0, 2, 3)
        temporal_key = self.temporal_k(temporal_grid)
        temporal_key = torch.cat(torch.split(temporal_key, head_dim, dim=-1), dim=0).permute(1, 0, 3, 2)
        temporal_value = self.temporal_v(temporal_grid)
        temporal_value = torch.cat(torch.split(temporal_value, head_dim, dim=-1), dim=0).permute(1, 0, 2, 3)
        temporal_attn_mask = repeat(temporal_occ, "b n -> b (t h) n", t=steps, h=num_heads).unsqueeze(2)
        temporal_attn_weights = torch.matmul(temporal_query, temporal_key)
        temporal_attn_weights /= math.sqrt(head_dim)
        temporal_attn_weights = temporal_attn_weights.masked_fill(~temporal_attn_mask, -1e9)
        temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
        temporal_attn_weights = temporal_attn_weights * temporal_attn_mask.float()
        temporal_attn_weights = temporal_attn_weights / (temporal_attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        temporal_value = torch.matmul(temporal_attn_weights, temporal_value)
        temporal_value = torch.cat(torch.split(temporal_value, int(steps), dim=1), dim=-1).squeeze(2)
        temporal_value = self.temporal_residual(self.input_embedding(src_t).permute(1, 0, 2), temporal_value)

        hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src_t)), pos=self.position_encoding(src_t))
        hist_enc = hist_enc.permute(1, 0, 2)

        nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs_t)), pos=self.position_encoding(nbrs_t))
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3)).bool()
        social_occ = mask.any(dim=-1)
        mask = repeat(mask, "b c n -> t b c n", t=steps)

        soc_enc = torch.zeros_like(mask, dtype=nbrs_enc.dtype, device=nbrs_enc.device)
        soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)

        query = self.qf(hist_enc)
        _, _, embed_size = query.shape
        head_dim_social = int(embed_size // num_heads)
        query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), head_dim_social, dim=-1), dim=1)
        key = torch.cat(torch.split(self.kf(soc_enc), head_dim_social, dim=-1), dim=0).permute(1, 0, 3, 2)
        value = torch.cat(torch.split(self.vf(soc_enc), head_dim_social, dim=-1), dim=0).permute(1, 0, 2, 3)
        social_attn_mask = repeat(social_occ, "b n -> b (t h) n", t=steps, h=num_heads).unsqueeze(2)
        attn_weights = torch.matmul(query, key)
        attn_weights /= math.sqrt(head_dim_social)
        attn_weights = attn_weights.masked_fill(~social_attn_mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights * social_attn_mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        value = torch.matmul(attn_weights, value)
        value = torch.cat(torch.split(value, int(steps), dim=1), dim=-1).squeeze(2)

        temporal_spatial_agg = self.leaky_relu(temporal_value + value)
        cross_tokens = torch.cat((hist_enc, temporal_spatial_agg), dim=-1)
        global_token = self.global_mlp(cross_tokens.mean(dim=1))

        return {
            "cross_tokens": cross_tokens,
            "global_token": global_token,
        }
