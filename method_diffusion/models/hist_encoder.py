import math

import torch
import torch.nn as nn

from method_diffusion.models.transformer import Residual, TransformerEncoder, TransformerEncoderLayer


class PositionalEncodingSine(nn.Module):
    """为 ego / 邻车历史轨迹构造二维正弦位置编码。"""

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
    """TAME-style 历史交互编码器，输出 fused ego tokens 与显式意图 logits。"""

    def __init__(self, args):
        """构建 temporal / social 双分支和显式意图头。

        Args:
            args: 模型配置对象，包含历史长度、编码维度与注意力头数等超参数。
        """
        super(HistEncoder, self).__init__()
        if int(args.feature_dim) != 4:
            raise ValueError("HistEncoder expects feature_dim=4: [rel_x, rel_y, v, a]")

        self.model_dim = int(args.encoder_input_dim)
        self.hidden_dim = int(getattr(args, "hidden_dim_fut", self.model_dim * 2))
        self.hist_length = int(getattr(args, "T", 16))
        self.max_lane_index = 8
        self.use_relative_lane_delta = int(getattr(args, "use_relative_lane_delta", 1)) > 0

        self.ego_input_embedding = nn.Linear(4, self.model_dim)
        self.nbr_input_embedding = nn.Linear(4, self.model_dim)
        self.ego_lane_embedding = nn.Embedding(self.max_lane_index, self.model_dim)
        self.lane_delta_embedding = nn.Embedding(self.max_lane_index * 2 + 1, self.model_dim)

        # 原 `build_position_encoding(args)` 的核心功能：
        # 按 `encoder_input_dim // 2` 构造二维正弦位置编码。
        self.position_encoding = PositionalEncodingSine(args.encoder_input_dim // 2, temperature=10000)

        encoder_layer = TransformerEncoderLayer(
            self.model_dim,
            args.nheads,
            dim_feedforward=args.dim_feedforward,
            dropout=0.1,
            activation=args.activation,
        )
        self.ego_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)
        self.nbr_encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)

        self.temporal_q = nn.Linear(self.model_dim, self.model_dim)
        self.temporal_k = nn.Linear(self.model_dim, self.model_dim)
        self.temporal_v = nn.Linear(self.model_dim, self.model_dim)
        self.temporal_residual = Residual(self.model_dim)
        self.social_q = nn.Linear(self.model_dim, self.model_dim)
        self.social_k = nn.Linear(self.model_dim, self.model_dim)
        self.social_v = nn.Linear(self.model_dim, self.model_dim)

        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(self.model_dim * 3),
            nn.Linear(self.model_dim * 3, self.hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        intent_heads = max(1, int(getattr(args, "heads_fut", 4)))
        if self.hidden_dim % intent_heads != 0:
            intent_heads = 1
        self.lat_cls = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.lon_cls = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.intent_attn = nn.MultiheadAttention(self.hidden_dim, intent_heads, dropout=0.1, batch_first=True)
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
        """按样本内占位顺序将邻车张量恢复到固定网格布局。

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

    def forward(
        self,
        ego_state_norm,
        nbr_state_norm,
        mask,
        temporal_mask,
        ego_lane,
        nbr_lane,
        ego_state_raw=None,
        nbr_state_raw=None,
    ):
        """输出 fused ego memory tokens 与显式意图 logits。

        Args:
            ego_state_norm: 标准化后的 ego 历史状态，形状为 `[B, T, 4]`。
            nbr_state_norm: 标准化后的邻车历史状态，形状为 `[N_total, T, 4]`。
            mask: 邻车空间存在性掩码，形状为 `[B, 3, 13, enc_size]`。
            temporal_mask: 邻车 temporal 分支掩码，形状为 `[B, 3, 13, feature_dim]`。
            ego_lane: ego 车道序列，形状为 `[B, T, 1]`。
            nbr_lane: 邻车车道序列，形状为 `[N_total, T, 1]`。
            ego_state_raw: 可选的 ego 原始历史状态。
            nbr_state_raw: 可选的邻车原始历史状态。

        Returns:
            一个四元组：
            - memory_tokens: 形状为 `[B, T, H]` 的 fused ego tokens。
            - memory_mask: 形状为 `[B, T]` 的 memory mask。
            - lat_logits: 横向意图 logits。
            - lon_logits: 纵向意图 logits。
        """
        del ego_state_raw, nbr_state_raw

        batch_size, seq_len, _ = ego_state_norm.shape
        if seq_len > self.hist_length:
            raise ValueError(f"History length {seq_len} exceeds configured hist_length {self.hist_length}")

        ego_embed = self.ego_input_embedding(ego_state_norm)
        ego_lane_idx = self.sanitizeIndex(ego_lane, self.max_lane_index)
        ego_embed = ego_embed + self.ego_lane_embedding(ego_lane_idx)

        ego_src = ego_embed.permute(1, 0, 2).contiguous()
        ego_pos = self.position_encoding(ego_state_norm.permute(1, 0, 2).contiguous())
        ego_enc = self.ego_encoder(ego_src, pos=ego_pos).permute(1, 0, 2)

        social_occ = mask.view(batch_size, mask.size(1) * mask.size(2), mask.size(3)).any(dim=-1)
        temporal_occ = temporal_mask.view(batch_size, temporal_mask.size(1) * temporal_mask.size(2), temporal_mask.size(3)).any(dim=-1)
        n_grid = social_occ.size(1)
        device = ego_state_norm.device

        if nbr_state_norm.size(0) > 0:
            nbr_embed = self.nbr_input_embedding(nbr_state_norm)
            nbr_src = nbr_embed.permute(1, 0, 2).contiguous()
            nbr_pos = self.position_encoding(nbr_state_norm.permute(1, 0, 2).contiguous())
            nbr_enc = self.nbr_encoder(nbr_src, pos=nbr_pos).permute(1, 0, 2)

            nbr_temporal_grid = self.scatterNeighbors(nbr_embed, temporal_occ)
            nbr_social_grid = self.scatterNeighbors(nbr_enc, social_occ)
            nbr_lane_grid = self.scatterNeighbors(nbr_lane, social_occ)
            if temporal_occ.equal(social_occ):
                nbr_lane_temporal_grid = nbr_lane_grid
            else:
                nbr_lane_temporal_grid = self.scatterNeighbors(nbr_lane, temporal_occ)
        else:
            nbr_temporal_grid = torch.zeros(batch_size, n_grid, seq_len, self.model_dim, device=device, dtype=ego_embed.dtype)
            nbr_social_grid = torch.zeros(batch_size, n_grid, seq_len, self.model_dim, device=device, dtype=ego_enc.dtype)
            nbr_lane_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_lane.dtype)
            nbr_lane_temporal_grid = torch.zeros(batch_size, n_grid, seq_len, 1, device=device, dtype=ego_lane.dtype)

        if self.use_relative_lane_delta:
            lane_delta_temporal = (nbr_lane_temporal_grid - ego_lane.unsqueeze(1)).squeeze(-1).round().long() + self.max_lane_index
            lane_delta_temporal = lane_delta_temporal.clamp_(0, self.max_lane_index * 2)
            lane_delta_social = (nbr_lane_grid - ego_lane.unsqueeze(1)).squeeze(-1).round().long() + self.max_lane_index
            lane_delta_social = lane_delta_social.clamp_(0, self.max_lane_index * 2)
            nbr_temporal_grid = nbr_temporal_grid + self.lane_delta_embedding(lane_delta_temporal)
            nbr_social_grid = nbr_social_grid + self.lane_delta_embedding(lane_delta_social)

        temporal_q = self.temporal_q(ego_embed)
        temporal_k = self.temporal_k(nbr_temporal_grid).permute(0, 2, 1, 3)
        temporal_v = self.temporal_v(nbr_temporal_grid).permute(0, 2, 1, 3)
        temporal_scores = (temporal_q.unsqueeze(2) * temporal_k).sum(dim=-1) / math.sqrt(self.model_dim)
        temporal_valid = temporal_occ.unsqueeze(1).expand(-1, seq_len, -1)
        temporal_scores = temporal_scores.masked_fill(~temporal_valid, -1e4)
        temporal_attn = torch.softmax(temporal_scores, dim=-1) * temporal_valid.to(temporal_scores.dtype)
        temporal_attn = temporal_attn / temporal_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        temporal_ctx = (temporal_attn.unsqueeze(-1) * temporal_v).sum(dim=2)
        temporal_ctx = self.temporal_residual(ego_embed, temporal_ctx)

        social_q = self.social_q(ego_enc)
        social_tokens = nbr_social_grid.permute(0, 2, 1, 3).contiguous()
        social_k = self.social_k(social_tokens)
        social_v = self.social_v(social_tokens)
        social_scores = (social_q.unsqueeze(2) * social_k).sum(dim=-1) / math.sqrt(self.model_dim)
        social_valid = social_occ.unsqueeze(1).expand(-1, seq_len, -1)
        social_scores = social_scores.masked_fill(~social_valid, -1e4)
        social_attn = torch.softmax(social_scores, dim=-1) * social_valid.to(social_scores.dtype)
        social_attn = social_attn / social_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        social_ctx = (social_attn.unsqueeze(-1) * social_v).sum(dim=2)

        fused = self.fusion_mlp(torch.cat((ego_enc, temporal_ctx, social_ctx), dim=-1))
        memory_tokens = fused
        memory_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

        intent_query = torch.cat(
            (
                self.lat_cls.expand(batch_size, -1, -1),
                self.lon_cls.expand(batch_size, -1, -1),
            ),
            dim=1,
        )
        intent_tokens = self.intent_attn(
            query=intent_query,
            key=fused,
            value=fused,
            key_padding_mask=memory_mask,
        )[0]
        intent_tokens = self.intent_cls_norm(intent_tokens + intent_query)
        lat_logits = self.lat_head(intent_tokens[:, 0, :])
        lon_logits = self.lon_head(intent_tokens[:, 1, :])
        return memory_tokens, memory_mask, lat_logits, lon_logits
