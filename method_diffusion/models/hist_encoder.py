import math

import torch
import torch.nn as nn

from method_diffusion.models.transformer import TransformerEncoder, TransformerEncoderLayer


class PositionalEncodingSine(nn.Module):
    """为 ego 历史轨迹构造二维正弦位置编码。"""

    def __init__(self, num_pos_feats=128, temperature=10000):
        """初始化位置编码超参数。

        Args:
            num_pos_feats: 每个空间轴使用的位置编码维度。
            temperature: 正弦位置编码的温度系数。
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = 2 * math.pi

    def forward(self, x):
        """根据 `[x, y]` 位置生成正弦编码。

        Args:
            x: 形状为 `[T, B, C]` 的轨迹状态张量，前两维为坐标。

        Returns:
            形状为 `[T, B, 2 * num_pos_feats]` 的位置编码张量。
        """
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        return torch.cat((pos_y, pos_x), dim=2)

class HistEncoder(nn.Module):
    """将 ego 历史和关键邻车编码为 memory token 与显式意图预测。"""

    def __init__(self, args):
        """构建 ego 编码器、邻车筛选器和意图分类头。

        Args:
            args: 模型配置对象，包含历史长度、隐藏维度、TopK 等超参数。
        """
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
            int(getattr(args, "interaction_topk", 6)),
        )
        self.interaction_dist_thresh = max(1.0, float(getattr(args, "interaction_dist_thresh", 120.0)))
        self.lane_emb_dim = max(1, int(getattr(args, "lane_emb_dim", 8)))
        self.max_lane_index = 8

        self.ego_input_embedding = nn.Linear(args.feature_dim, self.model_dim)
        # 原 `build_position_encoding(args)` 的核心功能：
        # 按 `encoder_input_dim // 2` 构造 ego 历史轨迹使用的二维正弦位置编码。
        self.position_encoding = PositionalEncodingSine(args.encoder_input_dim // 2, temperature=10000)
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

    def sanitizeIndex(self, values, num_embeddings):
        """将车道编号裁剪到 embedding 可接受的合法范围。

        Args:
            values: 原始车道编号张量，通常形状为 `[B, T, 1]`。
            num_embeddings: embedding 表支持的类别总数。

        Returns:
            裁剪后的整型索引张量。
        """
        indices = values.squeeze(-1).round().long() - 1
        return indices.clamp_(0, num_embeddings - 1)

    @staticmethod
    def scatterNeighbors(stacked_tensor, social_occ):
        """按样本内的占位顺序将邻车张量恢复到固定网格布局。

        Args:
            stacked_tensor: 按有效邻车顺序堆叠的邻车张量。
            social_occ: 每个样本的邻车占位布尔矩阵。

        Returns:
            形状为 `[B, N, T, C]` 的固定网格邻车张量。
        """
        batch_size, n_grid = social_occ.shape
        seq_len = stacked_tensor.size(1)
        feat_dim = stacked_tensor.size(2)
        if int(social_occ.sum().item()) != int(stacked_tensor.size(0)):
            raise RuntimeError(
                f"Neighbor scatter mismatch: occupied={int(social_occ.sum().item())}, stacked={stacked_tensor.size(0)}"
            )
        grid_tensor = stacked_tensor.new_zeros((batch_size * n_grid, seq_len, feat_dim))
        grid_tensor[social_occ.reshape(-1)] = stacked_tensor
        return grid_tensor.view(batch_size, n_grid, seq_len, feat_dim)

    def selectNeighborIndices(self, social_occ, ego_state_raw, nbr_state_raw_grid, ego_lane, nbr_lane_grid, nbr_dist_grid):
        """按物理影响分数前置选择 TopK 关键邻车。

        Args:
            social_occ: 每个样本的邻车存在性布尔矩阵。
            ego_state_raw: ego 原始历史状态，形状为 `[B, T, 4]`。
            nbr_state_raw_grid: 恢复到固定网格的邻车原始状态。
            ego_lane: ego 车道序列。
            nbr_lane_grid: 恢复到固定网格的邻车车道序列。
            nbr_dist_grid: 恢复到固定网格的邻车距离序列。

        Returns:
            一个二元组：
            - selected_idx: 每个样本选中的邻车索引。
            - selected_valid: 每个索引位置是否有效。
        """
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
        dist_last = nbr_dist_grid[:, :, -1, 0]
        closing_speed = torch.relu(ego_last[:, None, 2] - nbr_last[..., 2])

        inv_dy = 1.0 / (dy_last + 1.0)
        inv_dist = 1.0 / (dist_last + 1.0)
        inv_dx = 1.0 / (dx_last + 1.0)
        score = (
            1.25 * inv_dy
            + 0.75 * inv_dist
            + 0.25 * torch.clamp(closing_speed / 10.0, 0.0, 2.0)
            + 0.20 * same_lane
            + 0.10 * adjacent_lane
            + 0.15 * inv_dx
        )
        score = score.masked_fill(~social_occ, float("-inf"))

        close_enough = dist_last <= self.interaction_dist_thresh
        has_close = (social_occ & close_enough).any(dim=1, keepdim=True)
        available = social_occ & ((~has_close) | close_enough)
        topk_score, selected_idx = torch.topk(score.masked_fill(~available, float("-inf")), k=topk, dim=-1)
        selected_valid = torch.isfinite(topk_score)
        return selected_idx, selected_valid

    def buildRelationSummaryFeatures(self, ego_state_raw, selected_nbr_state_raw, ego_lane, selected_nbr_lane, selected_nbr_dist):
        """为每辆邻车提取一个紧凑的关系摘要特征。

        Args:
            ego_state_raw: ego 原始历史状态。
            selected_nbr_state_raw: 选中邻车的原始历史状态。
            ego_lane: ego 车道序列。
            selected_nbr_lane: 选中邻车的车道序列。
            selected_nbr_dist: 选中邻车的距离序列。

        Returns:
            形状为 `[B, K, 9]` 的关系摘要特征张量。
        """
        ego_last = ego_state_raw[:, -1, :].unsqueeze(1)
        nbr_last = selected_nbr_state_raw[:, :, -1, :]
        lane_delta_last = selected_nbr_lane[:, :, -1, 0] - ego_lane[:, -1, 0].unsqueeze(1)
        dist_last = selected_nbr_dist[:, :, -1, 0]

        dx_last = nbr_last[..., 0:1] - ego_last[..., 0:1]
        dy_last = nbr_last[..., 1:2] - ego_last[..., 1:2]
        dv_last = nbr_last[..., 2:3] - ego_last[..., 2:3]
        da_last = nbr_last[..., 3:4] - ego_last[..., 3:4]
        same_lane = (lane_delta_last.abs() < 0.5).to(dx_last.dtype).unsqueeze(-1)
        adjacent_lane = ((lane_delta_last.abs() - 1.0).abs() < 0.5).to(dx_last.dtype).unsqueeze(-1)
        closing_speed = torch.clamp((ego_last[..., 2:3] - nbr_last[..., 2:3]) / 10.0, -2.0, 2.0)

        return torch.cat(
            (
                torch.clamp(dx_last / 20.0, -10.0, 10.0),
                torch.clamp(dy_last / 80.0, -10.0, 10.0),
                torch.clamp(dv_last / 15.0, -10.0, 10.0),
                torch.clamp(da_last / 5.0, -10.0, 10.0),
                torch.clamp(dist_last.unsqueeze(-1) / self.interaction_dist_thresh, 0.0, 2.0),
                torch.clamp(lane_delta_last.unsqueeze(-1) / 2.0, -2.0, 2.0),
                same_lane,
                adjacent_lane,
                closing_speed,
            ),
            dim=-1,
        )

    def forward(
        self,
        ego_state_norm,
        nbr_state_norm,
        mask,
        ego_lane,
        nbr_lane,
        nbr_dist,
        ego_state_raw=None,
        nbr_state_raw=None,
    ):
        """输出供 DiT cross-attn 使用的 memory token 与意图 logits。

        Args:
            ego_state_norm: 标准化后的 ego 历史状态，形状为 `[B, T, 4]`。
            nbr_state_norm: 标准化后的邻车历史状态。
            mask: 邻车网格存在性掩码。
            ego_lane: ego 车道序列。
            nbr_lane: 邻车车道序列。
            nbr_dist: 邻车距离序列。
            ego_state_raw: 可选的 ego 原始历史状态。
            nbr_state_raw: 可选的邻车原始历史状态。

        Returns:
            一个四元组：
            - memory_tokens: 提供给 future DiT cross-attn 的上下文 token。
            - memory_mask: `memory_tokens` 的 padding mask。
            - lat_logits: 横向意图分类 logits。
            - lon_logits: 纵向意图分类 logits。
        """

        batch_size, seq_len, _ = ego_state_norm.shape
        if seq_len > self.hist_length:
            raise ValueError(f"History length {seq_len} exceeds configured hist_length {self.hist_length}")

        ego_tokens = self.ego_encoder(
            self.ego_input_embedding(ego_state_norm.permute(1, 0, 2).contiguous()),
            pos=self.position_encoding(ego_state_norm.permute(1, 0, 2).contiguous()),
        )
        ego_tokens = ego_tokens.permute(1, 0, 2)
        ego_lane_idx = self.sanitizeIndex(ego_lane, self.max_lane_index)
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
            nbr_lane_grid = self.scatterNeighbors(nbr_lane, social_occ)
            nbr_dist_grid = self.scatterNeighbors(nbr_dist, social_occ)
            nbr_state_raw_grid = self.scatterNeighbors(nbr_state_raw, social_occ)
        else:
            nbr_lane_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_lane.dtype)
            nbr_dist_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_state_norm.dtype)
            nbr_state_raw_grid = torch.zeros(batch_size, n_grid, seq_len, ego_state_raw.size(-1), device=device, dtype=ego_state_raw.dtype)

        selected_idx, selected_valid = self.selectNeighborIndices(
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
            relation_feat = self.buildRelationSummaryFeatures(
                ego_state_raw,
                selected_nbr_state_raw,
                ego_lane,
                selected_nbr_lane,
                selected_nbr_dist,
            )
            relation_tokens = self.relation_mlp(relation_feat) * selected_valid.unsqueeze(-1).to(ego_hidden.dtype)
            memory_tokens = torch.cat((ego_hidden, relation_tokens), dim=1)
            ego_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
            memory_mask = torch.cat((ego_mask, ~selected_valid), dim=1)
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
