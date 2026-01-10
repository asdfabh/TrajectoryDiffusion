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

    """
    src: [B, T, 4]
    nbrs: [N_t, T, 4]
    mask: [B, 3, 13, h]
    temporal_mask: [B, 3, 13, 4]
    """
    # def forward(self, src, nbrs, mask, temporal_mask):
    #
    #     # Temporal attention mechanism for temporal dependency.
    #     temporal_mask = temporal_mask.view(temporal_mask.size(0), temporal_mask.size(1) * temporal_mask.size(2), temporal_mask.size(3))
    #     temporal_mask = repeat(temporal_mask, 'b c n -> t b c n', t=self.args.T)  # [T, B, N, D]
    #     temporal_grid = torch.zeros_like(temporal_mask).float()
    #     temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs) # [T, B, N, D]
    #
    #     temporal_query = self.temporal_q(src)  # [B, T, attn_nhead*attn_out]
    #     # [B, T, 1, attn_out*attn_nhead] -> [B*attn_nhead, T, 1, attn_out] -> [T, B*attn_nhead, 1, attn_out]
    #     temporal_query = torch.cat(torch.split(torch.unsqueeze(temporal_query, dim=2), int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 2, 3) # [T, B * H, 1, D]
    #     temporal_key = self.temporal_k(temporal_grid) # [T, B, N, attn_nhead*attn_out]
    #     temporal_key = torch.cat(torch.split(temporal_key, int(self.args.attn_out), dim=-1), dim=1).permute(0, 1, 3, 2) # [T, B * H, D, N]
    #     temporal_value = self.temporal_v(temporal_grid)
    #     temporal_value = torch.cat(torch.split(temporal_value, int(self.args.attn_out), dim=-1), dim=1) # [T, B * H, N, D]
    #     temporal_attn_weights = torch.matmul(temporal_query, temporal_key) # [T, B * H, 1, N]
    #     temporal_attn_weights /= torch.math.sqrt(self.args.attn_out)
    #     temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
    #     temporal_value = torch.matmul(temporal_attn_weights, temporal_value) # [T, B * H, 1, D]
    #     temporal_value = torch.cat(torch.split(temporal_value, int(self.args.batch_size), dim=1), dim=-1).squeeze(2) # [T, B, D*H]
    #     imbedding_src_T = self.input_embedding(src).permute(1, 0, 2)  # [T, B, D*H]
    #     temporal_value = self.temporal_residual(imbedding_src_T, temporal_value)
    #
    #     hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src)), pos=self.position_encoding(src))
    #     hist_enc = hist_enc.permute(1, 0, 2) # [T, B, D]
    #
    #     # Social attention mechanism for interaction capture.
    #     nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs)), pos=self.position_encoding(nbrs))
    #     mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
    #     mask = repeat(mask, 'b c n -> t b c n', t=self.args.T)
    #
    #     soc_enc = torch.zeros_like(mask).float()
    #     soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)
    #
    #     query = self.qf(hist_enc)
    #     _, _, embed_size = query.shape
    #     query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), int(embed_size / self.args.attn_nhead), dim=-1),
    #                       dim=1)  # (batch_size, seq_len*attn_nhead, 1, att_out)
    #     key = torch.cat(torch.split(self.kf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(1, 0, 3, 2)  # (batch_size, seq_len*attn_nhead, att_out, 1)
    #     value = torch.cat(torch.split(self.vf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(
    #         1, 0, 2, 3)
    #     attn_weights = torch.matmul(query, key)
    #     attn_weights /= torch.math.sqrt(self.args.encoder_input_dim)
    #     attn_weights = F.softmax(attn_weights, dim=-1)
    #     value = torch.matmul(attn_weights, value)
    #     value = torch.cat(torch.split(value, int(src.shape[0]), dim=1), dim=-1).squeeze(2)
    #
    #     # Temporal and social aggregation.
    #     temporal_spatial_agg = self.leaky_relu(temporal_value + value)
    #
    #     enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1).permute(1, 0, 2)
    #     # 期望维度[B, T, dim=128]
    #     return enc


    """
        Inputs:
        src: [B, T, 4] -> 我们将在内部转换为 [T, B, 4]
        nbrs: [N_total, T, 4] -> 我们将在内部转换为 [T, N_total, 4]
        mask: [B, N_max, D] 或 [B, 3, 13, D] -> 用于构建 Grid
        temporal_mask: [B, 3, 13, 4] -> 用于构建 Grid
        """
    def forward(self, src, nbrs, mask, temporal_mask):
        # 记录原始 Batch 信息
        B, T, D_in = src.shape
        H = self.args.attn_nhead
        d = self.args.attn_out

        # [B, T, D] -> [T, B, D] [N_total, T, D] -> [T, N_total, D]
        src_t = src.permute(1, 0, 2).contiguous()
        nbrs_t = nbrs.permute(1, 0, 2).contiguous()

        # Ego、nbrs Embedding
        src_emb = self.input_embedding(src_t)  # [T, B, D_emb]
        src_pos = self.position_encoding(src_t)  # [T, B, D_emb]
        nbrs_emb = self.input_embedding(nbrs_t)  # [T, N_total, D_emb]
        nbrs_pos = self.position_encoding(nbrs_t)

        # Ego、nbrs Encoder (Transformer 期望 [T, B, D])
        hist_enc = self.encoder(self.leaky_relu(src_emb), pos=src_pos)  # [T, B, D_emb]
        nbrs_enc = self.encoder(self.leaky_relu(nbrs_emb), pos=nbrs_pos)  # [T, N_total, D_emb]

        # 基于每一帧 t，计算 Ego(src) 与 Nbrs(temporal_grid) 的原始物理关系
        # Mask 变换: [B, N_grid, D] -> [B, N_grid] (view) -> Repeat over Time -> [T, B, N_grid, D]
        t_mask_flat = temporal_mask.view(B, -1, temporal_mask.size(-1))
        t_mask_rep = repeat(t_mask_flat, 'b n d -> t b n d', t=T)  # [T, B, N, D]
        temporal_grid = torch.zeros_like(t_mask_rep).float()  # [T, B, N, D]
        temporal_grid = temporal_grid.masked_scatter_(t_mask_rep.bool(), nbrs_t)  # [T, B, N, D]
        # visualize_batch_trajectories(hist=temporal_grid.permute(1, 0, 2, 3), future=None, pred=None, batch_idx=0)

        # Query: [T, B, D_in] -> [T, B, H*d] -> [T, B, H, d]
        # Permute to [T, B, H, 1, d] -> [T, B*H, 1, d]
        t_q = self.temporal_q(src_t).view(T, B, H, d)
        t_q_in = t_q.permute(0, 1, 2, 3).contiguous().view(T, B * H, 1, d)

        # Key/Value: [T, B, N, D_in] -> [T, B, N, H*d]
        t_k = self.temporal_k(temporal_grid).view(T, B, -1, H, d)
        t_v = self.temporal_v(temporal_grid).view(T, B, -1, H, d)

        # Permute K to [T, B*H, d, N]
        # [T, B, N, H, d] -> [T, B, H, d, N] -> [T, B*H, d, N]、[T, B*H, N, d]
        t_k_in = t_k.permute(0, 1, 3, 4, 2).contiguous().view(T, B * H, d, -1)
        t_v_in = t_v.permute(0, 1, 3, 2, 4).contiguous().view(T, B * H, -1, d)

        # Attention: [T, B*H, 1, d] x [T, B*H, d, N] -> [T, B*H, 1, N]
        attn = torch.matmul(t_q_in, t_k_in)
        attn = attn / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)

        # Value Aggregation: [T, B*H, 1, N] x [T, B*H, N, d] -> [T, B*H, 1, d]
        t_out = torch.matmul(attn, t_v_in)
        # Restore: [T, B*H, 1, d] -> [T, B, H*d]
        t_out = t_out.view(T, B, H, d).contiguous().view(T, B, H * d)
        # Residual: Add to src_emb (which is [T, B, D_emb])
        t_res = self.temporal_residual(src_emb, t_out)  # [T, B, D_emb]

        # Scatter nbrs_enc to Grid [T, B, N, D]
        s_mask = mask.view(B, -1, mask.size(-1)) # Mask [B, N, D] -> [T, B, N, D]
        s_mask_rep = repeat(s_mask, 'b n d -> t b n d', t=T)

        soc_grid = torch.zeros_like(s_mask_rep).float()
        soc_grid = soc_grid.masked_scatter_(s_mask_rep.bool(), nbrs_enc)  # nbrs_enc 是 [T, N_total, D]，mask 是 [T, B, N, D] -> [T, B, N, D]

        # Query: [T, B, D] -> [T, B, H, d] -> [T*B, H, 1, d] (Flatten T and B)
        # 我们把所有时间步的所有Batch视为独立样本进行交互
        s_q = self.qf(hist_enc).view(T, B, H, d).reshape(T * B, H, 1, d)

        # Key/Value: [T, B, N, D] -> [T*B, H, d, N]
        s_k = self.kf(soc_grid).view(T, B, -1, H, d).permute(0, 1, 3, 4, 2).reshape(T * B, H, d, -1)
        s_v = self.vf(soc_grid).view(T, B, -1, H, d).permute(0, 1, 3, 2, 4).reshape(T * B, H, -1, d)

        # Attention
        s_attn = torch.matmul(s_q, s_k)  # [T*B, H, 1, N]
        s_attn = s_attn / math.sqrt(self.args.encoder_input_dim)
        s_attn = F.softmax(s_attn, dim=-1)
        s_out = torch.matmul(s_attn, s_v)  # [T*B, H, 1, d]
        # Restore: [T*B, H, 1, d] -> [T, B, H*d]
        s_out = s_out.reshape(T, B, H * d)

        temporal_spatial_agg = self.leaky_relu(t_res + s_out) # [T, B, D]

        enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1) # [T, B, 2D]
        enc = enc.permute(1, 0, 2).contiguous() # [B, T, 2D]
        # 期望维度[B, T, dim=128]
        return enc