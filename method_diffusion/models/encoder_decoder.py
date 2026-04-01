import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from method_diffusion.models.transformer import TransformerEncoder, TransformerEncoderLayer, Residual
from einops import repeat

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


class FutureDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, heads, dropout=0.1, mlp_ratio=2.0):
        super().__init__()
        self.norm_q_cross = nn.LayerNorm(hidden_dim)
        self.norm_mem = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)

        self.norm_q_self = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)

        self.norm_local = nn.LayerNorm(hidden_dim)
        self.local_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
        )

    def forward(self, query, memory):
        memory_norm = self.norm_mem(memory)
        query = query + self.cross_attn(
            self.norm_q_cross(query),
            memory_norm,
            memory_norm,
            need_weights=False,
        )[0]

        query_norm = self.norm_q_self(query)
        query = query + self.self_attn(query_norm, query_norm, query_norm, need_weights=False)[0]

        local_input = self.norm_local(query).transpose(1, 2)
        query = query + self.local_mixer(local_input).transpose(1, 2)
        query = query + self.ffn(self.norm_ffn(query))
        return query


class FutureContextDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.future_steps = int(args.T_f)
        self.hidden_dim = int(args.hidden_dim_fut)
        self.memory_dim = int(args.encoder_input_dim) * 2
        self.num_layers = 2
        self.num_heads = int(getattr(args, "heads_fut", 4))
        self.dropout = float(getattr(args, "dropout_fut", 0.1))

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(self.memory_dim),
            nn.Linear(self.memory_dim, self.hidden_dim),
        )
        self.summary_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.future_queries = nn.Parameter(torch.randn(1, self.future_steps, self.hidden_dim) * 0.02)
        self.future_pos = nn.Parameter(torch.randn(1, self.future_steps, self.hidden_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [FutureDecoderBlock(self.hidden_dim, self.num_heads, dropout=self.dropout, mlp_ratio=2.0) for _ in range(self.num_layers)]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, memory):
        batch_size = memory.size(0)
        memory = self.memory_proj(memory)
        summary = self.summary_proj(memory.mean(dim=1, keepdim=True))
        future_queries = self.future_queries + self.future_pos
        future_queries = future_queries.expand(batch_size, -1, -1) + summary

        for block in self.blocks:
            future_queries = block(future_queries, memory)

        return self.output_norm(future_queries)

class HistEncoder(nn.Module):
    def __init__(self, args):
        super(HistEncoder, self).__init__()
        self.args = args

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
        enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1) # [B, T, D_enc]

        return enc, hist_enc # [B, T, D_enc]


class HistEncoderDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hist_encoder = HistEncoder(args)
        self.future_decoder = FutureContextDecoder(args)

    def forward(self, src, nbrs, mask, temporal_mask):
        hist_context, hist_enc = self.hist_encoder(src, nbrs, mask, temporal_mask)
        aux = {
            "hist_context": hist_context,
            "hist_enc": hist_enc,
        }
        # 未来分支当前直接消费 hist encoder 输出，decoder 保留但暂不接入主路径。
        return hist_context, aux
