import math

import torch
import torch.nn as nn

from method_diffusion.models.transformer import TransformerEncoder, TransformerEncoderLayer


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
        super(HistEncoder, self).__init__()
        self.args = args
        if int(args.feature_dim) != 4:
            raise ValueError(
                "HistEncoder in unified future branch expects feature_dim=4: [rel_x, rel_y, v, a]"
            )

        self.model_dim = int(args.encoder_input_dim)
        self.hidden_dim = int(getattr(args, "hidden_dim_fut", self.model_dim * 2))
        self.hist_length = int(getattr(args, "T", 16))
        self.interaction_topk = max(
            0,
            int(getattr(args, "interaction_topk", getattr(args, "hist_memory_topk", 6))),
        )
        self.interaction_segments = max(1, int(getattr(args, "interaction_segments", 4)))
        self.interaction_dist_thresh = max(1.0, float(getattr(args, "interaction_dist_thresh", 120.0)))
        self.lane_emb_dim = max(1, int(getattr(args, "lane_emb_dim", 8)))
        self.max_lane_index = 8

        self.ego_input_embedding = nn.Linear(args.feature_dim, self.model_dim)
        self.position_encoding = build_position_encoding(args)
        encoder_layer = TransformerEncoderLayer(
            args.encoder_input_dim,
            args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=0.1,
            activation=args.activation,
        )
        self.ego_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)
        self.ego_lane_embedding = nn.Embedding(self.max_lane_index, self.lane_emb_dim)
        self.ego_meta_proj = nn.Sequential(
            nn.LayerNorm(self.lane_emb_dim),
            nn.Linear(self.lane_emb_dim, self.model_dim),
            nn.GELU(approximate="tanh"),
        )
        self.ego_to_hidden = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
        )

        relation_heads = max(1, int(getattr(args, "heads_fut", 4)))
        if self.hidden_dim % relation_heads != 0:
            relation_heads = 1
        self.relation_mlp = nn.Sequential(
            nn.LayerNorm(9),
            nn.Linear(9, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        relation_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=relation_heads,
            dim_feedforward=max(self.hidden_dim * 2, 128),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation_temporal_encoder = nn.TransformerEncoder(relation_layer, num_layers=1)
        self.time_embedding = nn.Parameter(torch.randn(1, self.hist_length, self.hidden_dim) * 0.02)
        self.slot_embedding = nn.Embedding(max(self.interaction_topk, 1), self.hidden_dim)
        self.segment_embedding = nn.Embedding(self.interaction_segments, self.hidden_dim)

        self.lat_cls = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.lon_cls = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.intent_memory_norm = nn.LayerNorm(self.hidden_dim)
        self.intent_attn = nn.MultiheadAttention(self.hidden_dim, relation_heads, dropout=0.1, batch_first=True)
        self.intent_cls_norm = nn.LayerNorm(self.hidden_dim)
        self.lat_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.lon_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )

    def _sanitize_index(self, values, num_embeddings):
        indices = values.squeeze(-1).round().long() - 1
        return indices.clamp_(0, num_embeddings - 1)

    @staticmethod
    def _scatter_neighbors(stacked_tensor, social_occ):
        batch_size, n_grid = social_occ.shape
        seq_len = stacked_tensor.size(1)
        feat_dim = stacked_tensor.size(2)
        grid_tensor = stacked_tensor.new_zeros((batch_size, n_grid, seq_len, feat_dim))

        offset = 0
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(social_occ[batch_idx], as_tuple=False).squeeze(-1)
            count = int(valid_idx.numel())
            if count > 0:
                grid_tensor[batch_idx, valid_idx] = stacked_tensor[offset:offset + count]
                offset += count

        if offset != int(stacked_tensor.size(0)):
            raise RuntimeError(
                f"Neighbor scatter mismatch: consumed={offset}, stacked={stacked_tensor.size(0)}"
            )
        return grid_tensor

    def _select_neighbor_indices(self, social_occ, ego_state_raw, nbr_state_raw_grid, ego_lane, nbr_lane_grid, nbr_dist_grid):
        batch_size, n_grid = social_occ.shape
        topk = min(self.interaction_topk, n_grid)
        if topk <= 0:
            empty_idx = social_occ.new_zeros((batch_size, 0), dtype=torch.long)
            empty_valid = social_occ.new_zeros((batch_size, 0), dtype=torch.bool)
            return empty_idx, empty_valid

        ego_last = ego_state_raw[:, -1, :]
        nbr_last = nbr_state_raw_grid[:, :, -1, :]
        lane_delta_last = nbr_lane_grid[:, :, -1, 0] - ego_lane[:, None, -1, 0]
        same_lane = (lane_delta_last.abs() < 0.5).to(ego_state_raw.dtype)
        adjacent_lane = ((lane_delta_last.abs() - 1.0).abs() < 0.5).to(ego_state_raw.dtype)

        dx_last = (nbr_last[..., 0] - ego_last[:, None, 0]).abs()
        dy_last = (nbr_last[..., 1] - ego_last[:, None, 1]).abs()
        dist_min = nbr_dist_grid[..., 0].masked_fill(~social_occ.unsqueeze(-1), float("inf")).amin(dim=-1)
        closing_speed = torch.relu(ego_last[:, None, 2] - nbr_last[..., 2])

        inv_dy = 1.0 / (dy_last + 1.0)
        inv_dist = 1.0 / (dist_min + 1.0)
        inv_dx = 1.0 / (dx_last + 1.0)
        score = (
            1.20 * inv_dy
            + 0.90 * inv_dist
            + 0.35 * torch.clamp(closing_speed / 10.0, 0.0, 2.0)
            + 0.25 * same_lane
            + 0.10 * adjacent_lane
            + 0.15 * inv_dx
        )
        score = score.masked_fill(~social_occ, float("-inf"))

        selected_idx = torch.full((batch_size, topk), -1, dtype=torch.long, device=social_occ.device)
        selected_valid = torch.zeros((batch_size, topk), dtype=torch.bool, device=social_occ.device)
        for batch_idx in range(batch_size):
            available = social_occ[batch_idx].clone()
            close_enough = dist_min[batch_idx] <= self.interaction_dist_thresh
            if (available & close_enough).any():
                available = available & close_enough
            batch_score = score[batch_idx].masked_fill(~available, float("-inf"))
            topk_score, topk_idx = torch.topk(batch_score, k=topk, dim=-1)
            topk_valid = torch.isfinite(topk_score)
            selected_idx[batch_idx] = topk_idx
            selected_valid[batch_idx] = topk_valid
        return selected_idx, selected_valid

    def _build_relation_features(self, ego_state_raw, selected_nbr_state_raw, ego_lane, selected_nbr_lane, selected_nbr_dist):
        ego_expand = ego_state_raw.unsqueeze(1).expand_as(selected_nbr_state_raw)
        dx = selected_nbr_state_raw[..., 0:1] - ego_expand[..., 0:1]
        dy = selected_nbr_state_raw[..., 1:2] - ego_expand[..., 1:2]
        dv = selected_nbr_state_raw[..., 2:3] - ego_expand[..., 2:3]
        da = selected_nbr_state_raw[..., 3:4] - ego_expand[..., 3:4]
        lane_delta = selected_nbr_lane - ego_lane.unsqueeze(1)
        same_lane = (lane_delta.abs() < 0.5).to(dx.dtype)
        adjacent_lane = ((lane_delta.abs() - 1.0).abs() < 0.5).to(dx.dtype)
        closing_speed = torch.clamp((ego_expand[..., 2:3] - selected_nbr_state_raw[..., 2:3]) / 10.0, -2.0, 2.0)
        relation_feat = torch.cat(
            (
                torch.clamp(dx / 20.0, -10.0, 10.0),
                torch.clamp(dy / 80.0, -10.0, 10.0),
                torch.clamp(dv / 15.0, -10.0, 10.0),
                torch.clamp(da / 5.0, -10.0, 10.0),
                torch.clamp(selected_nbr_dist / self.interaction_dist_thresh, 0.0, 2.0),
                torch.clamp(lane_delta / 2.0, -2.0, 2.0),
                same_lane,
                adjacent_lane,
                closing_speed,
            ),
            dim=-1,
        )
        return relation_feat

    def _segment_relation_tokens(self, selected_relation_tokens, selected_valid):
        batch_size, topk, seq_len, hidden_dim = selected_relation_tokens.shape
        if topk == 0:
            empty_tokens = selected_relation_tokens.new_zeros((batch_size, 0, hidden_dim))
            empty_mask = selected_valid.new_zeros((batch_size, 0))
            return empty_tokens, empty_mask

        slot_ids = torch.arange(topk, device=selected_relation_tokens.device)
        slot_emb = self.slot_embedding(slot_ids).view(1, topk, 1, hidden_dim)
        segment_tokens = []
        for seg_idx in range(self.interaction_segments):
            start = (seg_idx * seq_len) // self.interaction_segments
            end = ((seg_idx + 1) * seq_len) // self.interaction_segments
            if end <= start:
                end = min(start + 1, seq_len)
            pooled = selected_relation_tokens[:, :, start:end, :].mean(dim=2)
            pooled = pooled + slot_emb.squeeze(2) + self.segment_embedding.weight[seg_idx].view(1, 1, hidden_dim)
            pooled = pooled * selected_valid.unsqueeze(-1).to(pooled.dtype)
            segment_tokens.append(pooled)

        relation_tokens = torch.stack(segment_tokens, dim=2).reshape(batch_size, topk * self.interaction_segments, hidden_dim)
        relation_mask = (~selected_valid).unsqueeze(-1).expand(-1, -1, self.interaction_segments)
        relation_mask = relation_mask.reshape(batch_size, topk * self.interaction_segments)
        return relation_tokens, relation_mask

    def forward(
        self,
        ego_state_norm,
        nbr_state_norm,
        mask,
        temporal_mask,
        ego_lane,
        nbr_lane,
        nbr_dist,
        ego_state_raw=None,
        nbr_state_raw=None,
    ):
        del temporal_mask

        batch_size, seq_len, _ = ego_state_norm.shape
        if seq_len > self.hist_length:
            raise ValueError(f"History length {seq_len} exceeds configured hist_length {self.hist_length}")

        ego_tokens = self.ego_encoder(
            self.ego_input_embedding(ego_state_norm.permute(1, 0, 2).contiguous()),
            pos=self.position_encoding(ego_state_norm.permute(1, 0, 2).contiguous()),
        )
        ego_tokens = ego_tokens.permute(1, 0, 2)
        ego_lane_idx = self._sanitize_index(ego_lane, self.max_lane_index)
        ego_tokens = ego_tokens + self.ego_meta_proj(self.ego_lane_embedding(ego_lane_idx))
        ego_hidden = self.ego_to_hidden(ego_tokens)

        social_occ = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3)).any(dim=-1)
        n_grid = social_occ.size(1)
        if ego_state_raw is None:
            ego_state_raw = ego_state_norm
        if nbr_state_raw is None:
            nbr_state_raw = nbr_state_norm

        device = ego_state_norm.device
        if nbr_state_norm.size(0) > 0:
            nbr_lane_grid = self._scatter_neighbors(nbr_lane, social_occ)
            nbr_dist_grid = self._scatter_neighbors(nbr_dist, social_occ)
            nbr_state_raw_grid = self._scatter_neighbors(nbr_state_raw, social_occ)
        else:
            nbr_lane_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_lane.dtype)
            nbr_dist_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_state_norm.dtype)
            nbr_state_raw_grid = torch.zeros(batch_size, n_grid, seq_len, ego_state_raw.size(-1), device=device, dtype=ego_state_raw.dtype)

        selected_idx, selected_valid = self._select_neighbor_indices(
            social_occ=social_occ,
            ego_state_raw=ego_state_raw,
            nbr_state_raw_grid=nbr_state_raw_grid,
            ego_lane=ego_lane,
            nbr_lane_grid=nbr_lane_grid,
            nbr_dist_grid=nbr_dist_grid,
        )

        if selected_idx.size(1) > 0:
            safe_idx = selected_idx.clamp(min=0)
            batch_index = torch.arange(batch_size, device=device).unsqueeze(1)
            selected_nbr_state_raw = nbr_state_raw_grid[batch_index, safe_idx]
            selected_nbr_lane = nbr_lane_grid[batch_index, safe_idx]
            selected_nbr_dist = nbr_dist_grid[batch_index, safe_idx]
            relation_feat = self._build_relation_features(
                ego_state_raw,
                selected_nbr_state_raw,
                ego_lane,
                selected_nbr_lane,
                selected_nbr_dist,
            )
            relation_tokens = self.relation_mlp(relation_feat)
            relation_tokens = relation_tokens + self.time_embedding[:, :seq_len, :].unsqueeze(1)

            topk = selected_idx.size(1)
            relation_flat = relation_tokens.reshape(batch_size * topk, seq_len, self.hidden_dim)
            selected_valid_flat = selected_valid.reshape(batch_size * topk)
            encoded_relation_flat = relation_flat.new_zeros(relation_flat.shape)
            if selected_valid_flat.any():
                encoded_relation_flat[selected_valid_flat] = self.relation_temporal_encoder(relation_flat[selected_valid_flat])
            encoded_relation = encoded_relation_flat.reshape(batch_size, topk, seq_len, self.hidden_dim)
            encoded_relation = encoded_relation * selected_valid.unsqueeze(-1).unsqueeze(-1).to(encoded_relation.dtype)
            relation_memory_tokens, relation_memory_mask = self._segment_relation_tokens(encoded_relation, selected_valid)
            memory_tokens = torch.cat((ego_hidden, relation_memory_tokens), dim=1)
            ego_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            memory_mask = torch.cat((ego_mask, relation_memory_mask), dim=1)
        else:
            memory_tokens = ego_hidden
            memory_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        intent_query = torch.cat(
            (
                self.lat_cls.expand(batch_size, -1, -1),
                self.lon_cls.expand(batch_size, -1, -1),
            ),
            dim=1,
        )
        memory_norm = self.intent_memory_norm(memory_tokens)
        intent_tokens = self.intent_attn(
            query=intent_query,
            key=memory_norm,
            value=memory_norm,
            key_padding_mask=memory_mask,
        )[0]
        intent_tokens = self.intent_cls_norm(intent_tokens + intent_query)
        lat_logits = self.lat_head(intent_tokens[:, 0, :])
        lon_logits = self.lon_head(intent_tokens[:, 1, :])
        return memory_tokens, memory_mask, lat_logits, lon_logits
