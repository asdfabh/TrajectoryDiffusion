import math

import torch
import torch.nn.functional as F
from torch import nn


class IntentExtractor(nn.Module):
    def __init__(self, hidden_dim, hist_len, use_recent_bias=True, num_prototypes=9):
        super().__init__()
        if int(num_prototypes) != 9:
            raise ValueError("IntentExtractor 目前固定使用 9 个 joint prototypes。")

        self.hidden_dim = int(hidden_dim)
        self.hist_len = int(hist_len)
        self.use_recent_bias = bool(use_recent_bias)

        self.q_lat = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * (self.hidden_dim ** -0.5))
        self.q_lon = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * (self.hidden_dim ** -0.5))
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.use_recent_bias:
            self.time_bias_lat = nn.Parameter(torch.zeros(1, self.hist_len))
            self.time_bias_lon = nn.Parameter(torch.zeros(1, self.hist_len))
        else:
            self.register_buffer("time_bias_lat", torch.zeros(1, self.hist_len), persistent=False)
            self.register_buffer("time_bias_lon", torch.zeros(1, self.hist_len), persistent=False)

        self.lat_feat_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.lon_feat_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hist_global_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.lat_head = nn.Linear(self.hidden_dim, 3)
        self.lon_head = nn.Linear(self.hidden_dim, 3)
        self.intent_prototypes = nn.Parameter(torch.randn(9, self.hidden_dim) * (self.hidden_dim ** -0.5))

    def _resolve_bias(self, bias, t_len):
        if bias.size(-1) == t_len:
            return bias
        if bias.size(-1) > t_len:
            return bias[:, -t_len:]
        return F.pad(bias, (t_len - bias.size(-1), 0))

    def _attention_pool(self, query, keys, values, bias):
        scores = torch.matmul(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.hidden_dim)
        scores = scores + self._resolve_bias(bias, keys.size(1))
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), values).squeeze(1)

    def forward(self, context):
        bsz, t_len, _ = context.shape
        keys = self.k_proj(context)
        values = self.v_proj(context)

        q_lat = self.q_lat.expand(bsz, -1, -1)
        q_lon = self.q_lon.expand(bsz, -1, -1)
        z_pool_lat = self._attention_pool(q_lat, keys, values, self.time_bias_lat)
        z_pool_lon = self._attention_pool(q_lon, keys, values, self.time_bias_lon)

        recent_len = min(3, t_len)
        early_len = min(3, t_len)
        c_last = context[:, -1, :]
        z_recent = context[:, -recent_len:, :].mean(dim=1)
        z_early = context[:, :early_len, :].mean(dim=1)
        z_trend = z_recent - z_early

        feat_lat = self.lat_feat_mlp(torch.cat([z_pool_lat, c_last, z_trend], dim=-1))
        feat_lon = self.lon_feat_mlp(torch.cat([z_pool_lon, c_last, z_trend], dim=-1))

        lat_logits = self.lat_head(feat_lat)
        lon_logits = self.lon_head(feat_lon)
        p_lat = torch.softmax(lat_logits, dim=-1)
        p_lon = torch.softmax(lon_logits, dim=-1)

        p_joint = (p_lat.unsqueeze(-1) * p_lon.unsqueeze(1)).reshape(bsz, 9)
        e_int = torch.matmul(p_joint, self.intent_prototypes)
        z_hist = self.hist_global_mlp(torch.cat([context.mean(dim=1), c_last], dim=-1))
        return lat_logits, lon_logits, p_lat, p_lon, p_joint, e_int, z_hist
