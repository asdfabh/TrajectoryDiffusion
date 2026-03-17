import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from method_diffusion.models.transformer import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer, Residual
from einops import repeat

from method_diffusion.utils.visualization import visualize_batch_trajectories


class PositionalEncodingSine(nn.Module):

    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 64
        self.temperature = temperature  # 10000
        self.scale = 2 * math.pi  # 2 * math.pi

    def forward(self, x):
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # [0, 1, 2, ..., 63]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)  # [10000^0, 10000^2/64, 10000^4/64, ...]
        x_embed = x[:, :, 0] * self.scale
        y_embed = x[:, :, 1] * self.scale
        pos_x = x_embed[:, :, None] / dim_t  # [seq_len, batch_size, 1]
        pos_y = y_embed[:, :, None] / dim_t  # [seq_len, batch_size, 1]
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # [seq_len, batch_size, 64]
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)  # [seq_len, batch_size, 64]
        pos = torch.cat((pos_y, pos_x), dim=2)  # [seq_len, batch_size, 128]
        return pos


def build_position_encoding(args):
    position_embedding = PositionalEncodingSine(args.encoder_input_dim // 2, temperature=10000)
    return position_embedding


class HistEncoder(nn.Module):
    def __init__(self, args):
        super(HistEncoder, self).__init__()
        self.args = args
        if int(args.feature_dim) != 4:
            raise ValueError(
                "HistEncoder in unified future branch expects feature_dim=4: "
                "[rel_x, rel_y, v, a]"
            )
        self.model_dim = int(args.encoder_input_dim)
        self.hidden_dim = int(getattr(args, "hidden_dim_fut", self.model_dim * 2))
        self.memory_topk = max(0, int(getattr(args, "hist_memory_topk", 4)))

        # Initalize embeddings and input projection.
        self.input_embedding = nn.Linear(args.feature_dim, args.encoder_input_dim)
        self.position_encoding = build_position_encoding(args)

        # Initialize encoder and decoder transformer layers.
        encoder_layer = TransformerEncoderLayer(args.encoder_input_dim, args.nheads,
                                                dim_feedforward=args.dim_feedforward, dropout=0.1,
                                                activation=args.activation)
        self.encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=args.enc_layers)

        # Intialize Temporal attention mechanism.
        self.temporal_q = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_k = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_v = nn.Linear(args.feature_dim, args.attn_nhead * args.attn_out)
        self.temporal_residual = Residual(args.encoder_input_dim)

        # Initialize social attention mechanism.
        self.qf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.kf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)
        self.vf = nn.Linear(args.encoder_input_dim, args.attn_nhead * args.attn_out)

        # Initialize activation and regularization functions.
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.model_dim * 2),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(self.model_dim)

        self.fused_to_hidden = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
        )
        self.nbr_summary_to_hidden = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.hidden_dim),
            nn.GELU(approximate="tanh"),
        )

        intent_heads = max(1, int(getattr(args, "heads_fut", 4)))
        if self.hidden_dim % intent_heads != 0:
            intent_heads = 1
        self.intent_cls = nn.Parameter(torch.randn(1, 2, self.hidden_dim) * 0.02)
        self.intent_memory_norm = nn.LayerNorm(self.hidden_dim)
        self.intent_attn = nn.MultiheadAttention(self.hidden_dim, intent_heads, dropout=0.1, batch_first=True)
        self.intent_cls_norm = nn.LayerNorm(self.hidden_dim)
        self.intent_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _topk_neighbor_tokens(self, soc_enc, social_occ):
        # soc_enc: [T, B, N_grid, D_enc], social_occ: [B, N_grid]
        bsz = soc_enc.size(1)
        n_grid = soc_enc.size(2)
        topk = min(self.memory_topk, n_grid)
        if topk <= 0:
            empty_tokens = soc_enc.new_zeros((bsz, 0, self.model_dim))
            empty_valid = social_occ.new_zeros((bsz, 0))
            return empty_tokens, empty_valid

        nbr_tokens = soc_enc.permute(1, 2, 0, 3)  # [B, N_grid, T, D_enc]
        nbr_summary = nbr_tokens.mean(dim=2)  # [B, N_grid, D_enc]
        nbr_score = nbr_tokens.pow(2).mean(dim=(2, 3))  # [B, N_grid]
        nbr_score = nbr_score.masked_fill(~social_occ, float("-inf"))

        topk_score, topk_idx = torch.topk(nbr_score, k=topk, dim=1)
        topk_valid = torch.isfinite(topk_score)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, nbr_summary.size(-1))
        topk_summary = torch.gather(nbr_summary, 1, gather_idx)
        topk_summary = topk_summary * topk_valid.unsqueeze(-1).to(topk_summary.dtype)
        return topk_summary, topk_valid

    # 输入：src, nbrs [B, T, D] and [N_total, T, D]
    def forward(self, src, nbrs, mask, temporal_mask):

        B, T, D_in = src.shape
        num_heads = int(self.args.attn_nhead)
        head_dim = int(self.args.attn_out)

        # [B, T, D] -> [T, B, D] [N_total, T, D] -> [T, N_total, D]
        src_t = src.permute(1, 0, 2).contiguous()
        nbrs_t = nbrs.permute(1, 0, 2).contiguous()

        # Temporal attention mechanism for temporal dependency.
        temporal_mask = temporal_mask.view(temporal_mask.size(0), temporal_mask.size(1) * temporal_mask.size(2), temporal_mask.size(3))
        temporal_occ = temporal_mask.any(dim=-1)  # [B, N_grid]
        temporal_mask = repeat(temporal_mask, 'b c n -> t b c n', t=T) # [T, B, N, D]
        temporal_grid = torch.zeros_like(temporal_mask, dtype=nbrs_t.dtype, device=nbrs_t.device)
        temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs_t)

        temporal_query = self.temporal_q(src_t) # [T, B, H * D_attn]
        # [T * H, B, 1, D_attn] -> [B, T * H, 1, D_attn]
        temporal_query = torch.cat(torch.split(torch.unsqueeze(temporal_query, dim=2), head_dim, dim=-1), dim=0).permute(1, 0, 2, 3)
        temporal_key = self.temporal_k(temporal_grid) # [T, B, N, H * D_attn]
        temporal_key = torch.cat(torch.split(temporal_key, head_dim, dim=-1), dim=0).permute(1, 0, 3, 2) #[B, T * H, D_attn, N_grid
        temporal_value = self.temporal_v(temporal_grid)
        temporal_value = torch.cat(torch.split(temporal_value, head_dim, dim=-1), dim=0).permute(1, 0, 2, 3) # [B, T * H, N_grid, D_attn]
        temporal_attn_mask = repeat(temporal_occ, 'b n -> b (t h) n', t=T, h=num_heads).unsqueeze(2)
        temporal_attn_weights = torch.matmul(temporal_query, temporal_key) # [B, T*H, 1, N_grid]
        temporal_attn_weights /= math.sqrt(head_dim)
        temporal_attn_weights = temporal_attn_weights.masked_fill(~temporal_attn_mask, -1e9)
        temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
        temporal_attn_weights = temporal_attn_weights * temporal_attn_mask.float()
        temporal_attn_weights = temporal_attn_weights / (temporal_attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        temporal_value = torch.matmul(temporal_attn_weights, temporal_value) # [B, T*H, 1, D_attn]
        temporal_value = torch.cat(torch.split(temporal_value, int(T), dim=1), dim=-1).squeeze(2) # [B, T, H * D_attn]
        temporal_value = self.temporal_residual(self.input_embedding(src_t).permute(1, 0, 2), temporal_value) # [B, T, H * D_attn]

        hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src_t)), pos=self.position_encoding(src_t)) #  [T, B, D_enc]
        hist_enc = hist_enc.permute(1, 0, 2) # [B, T, D_enc]

        # Social attention mechanism for interaction capture.
        nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs_t)), pos=self.position_encoding(nbrs_t)) # [T, N_total, D_enc]
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        social_occ = mask.any(dim=-1)  # [B, N_grid]
        mask = repeat(mask, 'b c n -> t b c n', t=T) # [T, B, N_grid, N]

        soc_enc = torch.zeros_like(mask, dtype=nbrs_enc.dtype, device=nbrs_enc.device)
        soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc) # [T, B, N_grid, D_enc]

        query = self.qf(hist_enc) # [B, T, H*d] [1024, 16, 64]
        _, _, embed_size = query.shape
        head_dim_social = int(embed_size // num_heads)
        query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), head_dim_social, dim=-1), dim=1)  # [B, T*H, 1, D_attn]
        key = torch.cat(torch.split(self.kf(soc_enc), head_dim_social, dim=-1), dim=0).permute(1, 0, 3, 2)  # [B, T*H, D_attn, N_grid]
        value = torch.cat(torch.split(self.vf(soc_enc), head_dim_social, dim=-1), dim=0).permute(1, 0, 2, 3) # [B, T*H, N_grid, D_attn]
        social_attn_mask = repeat(social_occ, 'b n -> b (t h) n', t=T, h=num_heads).unsqueeze(2)
        attn_weights = torch.matmul(query, key) # [B, T*H, 1, N_grid]
        attn_weights /= math.sqrt(head_dim_social)
        attn_weights = attn_weights.masked_fill(~social_attn_mask, -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights * social_attn_mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        value = torch.matmul(attn_weights, value) # [B, T*H, 1, D_attn]
        value = torch.cat(torch.split(value, int(T), dim=1), dim=-1).squeeze(2) # [B, T, H * D_attn]

        temporal_spatial_agg = self.leaky_relu(temporal_value + value) # [B, T, D_enc]

        # 共享主干融合：把时间建模与交互建模统一到同一语义 token。
        gate = self.fusion_gate(torch.cat((hist_enc, temporal_spatial_agg), dim=-1))
        fused_tokens = self.fusion_norm(hist_enc + gate * temporal_spatial_agg)  # [B, T, D_enc]

        # Memory Head: ego 时间 token + Top-K 邻居摘要 token。
        fused_hidden = self.fused_to_hidden(fused_tokens)  # [B, T, hidden]
        topk_summary, topk_valid = self._topk_neighbor_tokens(soc_enc, social_occ)
        if topk_summary.size(1) > 0:
            nbr_hidden = self.nbr_summary_to_hidden(topk_summary)
            memory_tokens = torch.cat((fused_hidden, nbr_hidden), dim=1)
        else:
            memory_tokens = fused_hidden

        ego_mask = torch.zeros((B, T), dtype=torch.bool, device=src.device)
        if topk_valid.size(1) > 0:
            nbr_mask = ~topk_valid
            memory_mask = torch.cat((ego_mask, nbr_mask), dim=1)
        else:
            memory_mask = ego_mask

        # Intent Head: 两个 CLS 从同一个 memory 池聚合全局行为意图。
        intent_query = self.intent_cls.expand(B, -1, -1)
        memory_norm = self.intent_memory_norm(memory_tokens)
        intent_tokens = self.intent_attn(
            query=intent_query,
            key=memory_norm,
            value=memory_norm,
            key_padding_mask=memory_mask
        )[0]
        intent_tokens = self.intent_cls_norm(intent_tokens + intent_query)
        z_motion, z_maneuver = intent_tokens[:, 0, :], intent_tokens[:, 1, :]
        z_global = self.intent_fusion(torch.cat((z_motion, z_maneuver), dim=-1))

        return memory_tokens, z_global, memory_mask

