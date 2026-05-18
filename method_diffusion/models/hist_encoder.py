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

class HistEncoder(nn.Module):
    def __init__(self, args):
        super(HistEncoder, self).__init__()
        self.args = args

        # Initalize embeddings and input projection.
        self.input_embedding = nn.Linear(args.feature_dim, args.encoder_input_dim)
        self.position_encoding = build_position_encoding(args)

        # Initialize encoder and decoder transformer layers.
        encoder_layer = TransformerEncoderLayer(args.encoder_input_dim, args.nheads,
                                                dim_feedforward=args.dim_feedforward, dropout=0.1, activation=args.activation)
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

    def forward(self, hist, nbrs, mask, temporal_mask):
        src = hist
        # Temporal attention mechanism for temporal dependency.
        temporal_mask = temporal_mask.view(temporal_mask.size(0), temporal_mask.size(1) * temporal_mask.size(2), temporal_mask.size(3))
        temporal_mask = repeat(temporal_mask, 'b c n -> b t c n', t=self.args.hist_length)
        temporal_grid = torch.zeros_like(temporal_mask).float()
        temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs)

        temporal_query = self.temporal_q(src)  # (batch_size, seq_len, attn_nhead*attn_out)
        temporal_query = torch.cat(torch.split(torch.unsqueeze(temporal_query, dim=2), int(self.args.attn_out), dim=-1), dim=1)  # (batch_size, seq_len*attn_nhead, 1, att_out)
        temporal_key = self.temporal_k(temporal_grid)
        temporal_key = torch.cat(torch.split(temporal_key, int(self.args.attn_out), dim=-1), dim=1).permute(0, 1, 3, 2)
        temporal_value = self.temporal_v(temporal_grid)
        temporal_value = torch.cat(torch.split(temporal_value, int(self.args.attn_out), dim=-1), dim=1)
        temporal_attn_weights = torch.matmul(temporal_query, temporal_key)
        temporal_attn_weights /= torch.math.sqrt(self.args.attn_out)
        temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
        temporal_value = torch.matmul(temporal_attn_weights, temporal_value)
        temporal_value = torch.cat(torch.split(temporal_value, int(self.args.hist_length), dim=1), dim=-1).squeeze(2)
        temporal_value = self.temporal_residual(self.input_embedding(src), temporal_value)

        hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src)), pos=self.position_encoding(src))

        # Social attention mechanism for interaction capture.
        nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs)), pos=self.position_encoding(nbrs))
        mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
        mask = repeat(mask, 'b c n -> b t c n', t=self.args.hist_length)

        soc_enc = torch.zeros_like(mask).float()
        soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)

        query = self.qf(hist_enc)
        _, _, embed_size = query.shape
        query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), int(embed_size / self.args.attn_nhead), dim=-1), dim=1)  # (batch_size, seq_len*attn_nhead, 1, att_out)
        key = torch.cat(torch.split(self.kf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=1).permute(0, 1, 3, 2)  # (batch_size, seq_len*attn_nhead, att_out, 1)
        value = torch.cat(torch.split(self.vf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=1)
        attn_weights = torch.matmul(query, key)
        attn_weights /= torch.math.sqrt(self.args.encoder_input_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        value = torch.matmul(attn_weights, value)
        value = torch.cat(torch.split(value, int(hist.shape[1]), dim=1), dim=-1).squeeze(2)

        temporal_spatial_agg = self.leaky_relu(temporal_value + value)
        enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1)
        return enc

    # def forward(self, src, nbrs, mask, temporal_mask):
    #
    #     src = src.contiguous()
    #     nbrs = nbrs.contiguous()
    #     B, T, _ = src.shape
    #
    #     # MaDiff HighwayNet temporal branch: batch-first [B, T, D].
    #     temporal_mask = temporal_mask.view(B, -1, self.feature_dim).bool()
    #     temporal_mask = repeat(temporal_mask, 'b c n -> b t c n', t=T)
    #     temporal_grid = torch.zeros_like(temporal_mask, dtype=nbrs.dtype, device=nbrs.device)
    #     temporal_grid = temporal_grid.masked_scatter_(temporal_mask, nbrs)
    #
    #     temporal_query = self.temporal_q(src)
    #     temporal_query = torch.cat(
    #         torch.split(temporal_query.unsqueeze(2), self.attn_out, dim=-1),
    #         dim=1,
    #     )
    #     temporal_key = self.temporal_k(temporal_grid)
    #     temporal_key = torch.cat(torch.split(temporal_key, self.attn_out, dim=-1), dim=1).permute(0, 1, 3, 2)
    #     temporal_value = self.temporal_v(temporal_grid)
    #     temporal_value = torch.cat(torch.split(temporal_value, self.attn_out, dim=-1), dim=1)
    #     temporal_attn_weights = torch.matmul(temporal_query, temporal_key)
    #     temporal_attn_weights /= math.sqrt(self.attn_out)
    #     temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
    #     temporal_value = torch.matmul(temporal_attn_weights, temporal_value)
    #     temporal_value = torch.cat(torch.split(temporal_value, T, dim=1), dim=-1).squeeze(2)
    #     temporal_value = self.temporal_residual(self.input_embedding(src), temporal_value)
    #
    #     hist_enc = self.encoder(
    #         self.leaky_relu(self.input_embedding(src)),
    #         pos=self.position_encoding(src),
    #     )
    #
    #     # MaDiff HighwayNet social branch: encode compressed neighbors, then scatter back to the grid.
    #     nbrs_enc = self.encoder(
    #         self.leaky_relu(self.input_embedding(nbrs)),
    #         pos=self.position_encoding(nbrs),
    #     )
    #     mask = mask.view(B, -1, self.encoder_input_dim).bool()
    #     mask = repeat(mask, 'b c n -> b t c n', t=T)
    #     soc_enc = torch.zeros_like(mask, dtype=nbrs_enc.dtype, device=nbrs_enc.device)
    #     soc_enc = soc_enc.masked_scatter_(mask, nbrs_enc)
    #
    #     query = self.qf(hist_enc)
    #     embed_size = query.size(-1)
    #     head_dim = int(embed_size // self.attn_nhead)
    #     query = torch.cat(torch.split(query.unsqueeze(2), head_dim, dim=-1), dim=1)
    #     key = torch.cat(torch.split(self.kf(soc_enc), head_dim, dim=-1), dim=1).permute(0, 1, 3, 2)
    #     value = torch.cat(torch.split(self.vf(soc_enc), head_dim, dim=-1), dim=1)
    #     attn_weights = torch.matmul(query, key)
    #     attn_weights /= math.sqrt(self.encoder_input_dim)
    #     attn_weights = F.softmax(attn_weights, dim=-1)
    #     value = torch.matmul(attn_weights, value)
    #     value = torch.cat(torch.split(value, T, dim=1), dim=-1).squeeze(2)
    #
    #     temporal_spatial_agg = self.leaky_relu(temporal_value + value)
    #     enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1)
    #     return enc
