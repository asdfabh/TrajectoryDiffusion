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
        self.cross_topk_nbr = max(0, int(getattr(args, "cross_topk_nbr", 12)))
        self.context_dim = int(args.encoder_input_dim) * 2

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

        t_hist = int(args.hist_length)
        d_enc = int(args.encoder_input_dim)
        self.nbr_token_proj = nn.Sequential(
            nn.LayerNorm(t_hist * d_enc),
            nn.Linear(t_hist * d_enc, self.context_dim),
        )

    def _build_topk_neighbor_tokens(self, soc_enc, temporal_grid, social_occ):
        batch_size, n_grid = soc_enc.size(1), soc_enc.size(2)
        topk = min(self.cross_topk_nbr, n_grid)

        if topk <= 0:
            empty_tokens = soc_enc.new_zeros((batch_size, 0, self.context_dim))
            empty_valid = torch.zeros((batch_size, 0), dtype=torch.bool, device=soc_enc.device)
            return empty_tokens, empty_valid

        nbr_feat = soc_enc.permute(1, 2, 0, 3).contiguous()  # [B, N_grid, T, D_enc]
        _, _, t_hist, d_enc = nbr_feat.shape
        nbr_feat = nbr_feat.view(batch_size, n_grid, t_hist * d_enc)  # [B, N_grid, T*D_enc]

        nbr_xy_last = temporal_grid[-1, :, :, :2]
        # We keep distance in normalized space on purpose.
        # With anisotropic pos std (x/y), this behaves like an elliptical safety field for highway scenarios.
        nbr_dist = torch.norm(nbr_xy_last, dim=-1)
        nbr_dist = nbr_dist.masked_fill(~social_occ, float("inf"))
        topk_idx = torch.topk(nbr_dist, k=topk, dim=1, largest=False).indices

        gather_feat_idx = topk_idx.unsqueeze(-1).expand(-1, -1, nbr_feat.size(-1))
        nbr_feat_topk = torch.gather(nbr_feat, dim=1, index=gather_feat_idx)
        nbr_valid = torch.gather(social_occ, dim=1, index=topk_idx)

        nbr_tokens = self.nbr_token_proj(nbr_feat_topk)
        nbr_tokens = nbr_tokens * nbr_valid.unsqueeze(-1).to(nbr_tokens.dtype)
        return nbr_tokens, nbr_valid


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
        temporal_attn_weights = temporal_attn_weights.masked_fill(~temporal_attn_mask, 0.0)
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
        attn_weights = attn_weights.masked_fill(~social_attn_mask, 0.0)
        value = torch.matmul(attn_weights, value) # [B, T*H, 1, D_attn]
        value = torch.cat(torch.split(value, int(T), dim=1), dim=-1).squeeze(2) # [B, T, H * D_attn]

        temporal_spatial_agg = self.leaky_relu(temporal_value + value) # [B, T, D_enc]
        enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1) # [B, T, 2*D_enc]

        nbr_tokens, nbr_valid = self._build_topk_neighbor_tokens(soc_enc, temporal_grid, social_occ)
        enc_valid = torch.ones((B, T), dtype=torch.bool, device=enc.device)
        context_tokens = torch.cat([enc, nbr_tokens], dim=1)
        context_valid = torch.cat([enc_valid, nbr_valid], dim=1)

        return context_tokens, enc, context_valid


# tame代码 维度转化说明
# def forward(self, hist, nbrs, mask, va, nbrsva, lane, nbrslane, cls, nbrscls, temporal_mask):
#     # 1. 特征拼接 (Feature Concatenation)
#     if self.args.input_dim == 2:
#         src = hist
#     elif self.args.input_dim == 5:
#         # src: [T, B, 5]
#         src = torch.cat((hist, cls, va), dim=-1)
#         # nbrs: [T, N_total, 5]
#         nbrs = torch.cat((nbrs, nbrscls, nbrsva), dim=-1)
#     else:
#         # src: [T, B, D_in] (例如 D_in=2+3+2+dims)
#         src = torch.cat((hist, cls, va, lane), dim=-1)
#         # nbrs: [T, N_total, D_in]
#         nbrs = torch.cat((nbrs, nbrscls, nbrsva, nbrslane), dim=-1)
#
#     # 2. 时间注意力机制 (Temporal Attention Mechanism)
#     # 将空间网格维度展平: [B, H, W, N] -> [B, H*W, N] -> [B, N_grid, N]
#     temporal_mask = temporal_mask.view(temporal_mask.size(0), temporal_mask.size(1) * temporal_mask.size(2),
#                                        temporal_mask.size(3))
#     # 在时间维度复制: [B, N_grid, N] -> [T, B, N_grid, N]
#     temporal_mask = repeat(temporal_mask, 'b c n -> t b c n', t=self.args.hist_length)
#
#     # 初始化网格容器，用于放置邻居特征
#     temporal_grid = torch.zeros_like(temporal_mask).float()
#     # 使用掩码将扁平化的邻居特征 scatter 到网格中
#     # nbrs: [T, N_total, D_in] -> temporal_grid: [T, B, N_grid, D_in] (此处维度变换隐含了 N 的处理，假设 N_grid 维包含了邻居最大数)
#     # 注意：这里的 scatter 操作根据 mask 将紧凑的 nbrs 映射回 [T, B, Grid_Locations, D_in] 的结构
#     temporal_grid = temporal_grid.masked_scatter_(temporal_mask.bool(), nbrs)
#
#     # 计算 Query (从自身历史轨迹)
#     # src: [T, B, D_in] -> temporal_query: [T, B, H * D_attn]
#     temporal_query = self.temporal_q(src)
#     # Multi-head 处理:
#     # 1. Unsqueeze: [T, B, 1, H * D_attn]
#     # 2. Split: List of H tensors, each [T, B, 1, D_attn]
#     # 3. Cat dim=0 (Stack Time): [T * H, B, 1, D_attn]
#     # 4. Permute: [B, T * H, 1, D_attn]
#     temporal_query = torch.cat(torch.split(torch.unsqueeze(temporal_query, dim=2), int(self.args.attn_out), dim=-1),
#                                dim=0).permute(1, 0, 2, 3)
#
#     # 计算 Key (从网格化邻居特征)
#     # temporal_grid: [T, B, N_grid, D_in] -> temporal_key: [T, B, N_grid, H * D_attn]
#     temporal_key = self.temporal_k(temporal_grid)
#     # Multi-head 处理:
#     # ... Split & Cat dim=0: [T * H, B, N_grid, D_attn]
#     # Permute: [B, T * H, D_attn, N_grid] (转置最后两维以便矩阵乘法)
#     temporal_key = torch.cat(torch.split(temporal_key, int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 3, 2)
#
#     # 计算 Value
#     # temporal_value: [T, B, N_grid, H * D_attn]
#     temporal_value = self.temporal_v(temporal_grid)
#     # Permute: [B, T * H, N_grid, D_attn]
#     temporal_value = torch.cat(torch.split(temporal_value, int(self.args.attn_out), dim=-1), dim=0).permute(1, 0, 2, 3)
#
#     # Attention Weights Calculation
#     # Q * K^T: [B, T*H, 1, D_attn] * [B, T*H, D_attn, N_grid] -> [B, T*H, 1, N_grid]
#     temporal_attn_weights = torch.matmul(temporal_query, temporal_key)
#     temporal_attn_weights /= torch.math.sqrt(self.args.attn_out)
#     temporal_attn_weights = F.softmax(temporal_attn_weights, dim=-1)
#
#     # Weighted Sum (Apply Attention)
#     # Weights * V: [B, T*H, 1, N_grid] * [B, T*H, N_grid, D_attn] -> [B, T*H, 1, D_attn]
#     temporal_value = torch.matmul(temporal_attn_weights, temporal_value)
#
#     # Merge Heads back
#     # 1. Split dim=1 by T: List of H tensors, each [B, T, 1, D_attn]
#     # 2. Cat dim=-1: [B, T, 1, H * D_attn]
#     # 3. Squeeze: [B, T, H * D_attn]
#     temporal_value = torch.cat(torch.split(temporal_value, int(self.args.hist_length), dim=1), dim=-1).squeeze(2)
#
#     # Residual Connection & Projection
#     # input_embedding(src): [T, B, D_enc] -> Permute: [B, T, D_enc]
#     # temporal_residual maps [B, T, H*D_attn] back to [B, T, D_enc] and adds input
#     temporal_value = self.temporal_residual(self.input_embedding(src).permute(1, 0, 2), temporal_value)
#
#     # 3. 历史轨迹编码 (Transformer Encoder)
#     # input_embedding(src): [T, B, D_in] -> [T, B, D_enc]
#     # encoder output hist_enc: [T, B, D_enc]
#     hist_enc = self.encoder(self.leaky_relu(self.input_embedding(src)), pos=self.position_encoding(src))
#     # Permute: [B, T, D_enc]
#     hist_enc = hist_enc.permute(1, 0, 2)
#
#     # 4. 社交注意力机制 (Social Attention Mechanism)
#     # Process neighbors through encoder
#     # nbrs: [T, N_total, D_in] -> nbrs_enc: [T, N_total, D_enc]
#     nbrs_enc = self.encoder(self.leaky_relu(self.input_embedding(nbrs)), pos=self.position_encoding(nbrs))
#
#     # Prepare Social Mask (similar to temporal mask logic)
#     mask = mask.view(mask.size(0), mask.size(1) * mask.size(2), mask.size(3))
#     mask = repeat(mask, 'b c n -> t b c n', t=self.args.hist_length)  # [T, B, N_grid, N]
#
#     # Scatter embedded neighbors back to grid structure
#     soc_enc = torch.zeros_like(mask).float()
#     soc_enc = soc_enc.masked_scatter_(mask.bool(), nbrs_enc)  # [T, B, N_grid, D_enc]
#
#     # 计算 Q, K, V
#     # Q (from history): hist_enc [B, T, D_enc] -> qf -> [B, T, H*D_attn]
#     query = self.qf(hist_enc)
#     _, _, embed_size = query.shape
#
#     # Split Heads for Q: [B, T, 1, H*D_attn] -> Split & Cat dim=1 -> [B, T*H, 1, D_attn]
#     query = torch.cat(torch.split(torch.unsqueeze(query, dim=2), int(embed_size / self.args.attn_nhead), dim=-1), dim=1)
#
#     # Process K from neighbors: soc_enc [T, B, N_grid, D_enc]
#     # kf(soc_enc) -> [T, B, N_grid, H*D_attn]
#     # Split & Cat dim=0 (Stack Time): [T*H, B, N_grid, D_attn]
#     # Permute: [B, T*H, D_attn, N_grid] (Transposed for K)
#     key = torch.cat(torch.split(self.kf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(1, 0,
#                                                                                                                   3, 2)
#
#     # Process V from neighbors:
#     # Permute: [B, T*H, N_grid, D_attn]
#     value = torch.cat(torch.split(self.vf(soc_enc), int(embed_size / self.args.attn_nhead), dim=-1), dim=0).permute(1,
#                                                                                                                     0,
#                                                                                                                     2,
#                                                                                                                     3)
#
#     # Attention
#     # [B, T*H, 1, N_grid]
#     attn_weights = torch.matmul(query, key)
#     attn_weights /= torch.math.sqrt(self.args.encoder_input_dim)
#     attn_weights = F.softmax(attn_weights, dim=-1)
#
#     # [B, T*H, 1, D_attn]
#     value = torch.matmul(attn_weights, value)
#
#     # Merge Heads
#     # Split T*H by T -> Cat dim=-1 -> [B, T, 1, H*D_attn] -> Squeeze -> [B, T, H*D_attn] (Assuming H*D_attn == D_enc here)
#     value = torch.cat(torch.split(value, int(hist.shape[0]), dim=1), dim=-1).squeeze(2)
#
#     # 5. 聚合与解码 (Aggregation & Decoding)
#     # Sum Temporal Attention + Social Attention: [B, T, D_enc]
#     temporal_spatial_agg = self.leaky_relu(temporal_value + value)
#
#     # Feature Concatenation:
#     # [B, T, D_enc] cat [B, T, D_enc] -> [B, T, 2*D_enc]
#     # Permute for Decoder: [T, B, 2*D_enc]
#     enc = torch.cat((temporal_spatial_agg, hist_enc), dim=-1).permute(1, 0, 2)
#
#     memory = enc
#     # Query Position for Decoder: [1, B, 2*D_enc] (Assuming single query vector broadcasted)
#     query_pos = self.query_position_ecoding.weight.unsqueeze(1).repeat(1, enc.shape[1], 1)
#     tgt = torch.zeros_like(query_pos)
#
#     # Decoder Output:
#     # hs: [Layer_Count/1, B, 2*D_enc] or [T_out, B, 2*D_enc] depending on config.
#     # Assuming return_intermediate=True, hs is [Num_Layers, B, 2*D_enc]
#     hs = self.decoder(tgt, memory, query_pos=query_pos)
#
#     # 6. 混合专家输出 (Mixture of Experts Output)
#     # Taking the last layer output: hs[-1] shape [B, 2*D_enc] (or hidden_dim)
#     # expert_output: [B, hidden_dim]
#     expert_output = self.expert_gate(hs[-1])
#
#     # 7. 多任务输出投影 (Multi-task Projection)
#     # output_lat: [B, lat_dim] (Softmax probabilities)
#     output_lat = F.softmax(self.output_lateral(expert_output), dim=-1)
#     # output_lon: [B, lon_dim] (Softmax probabilities)
#     output_lon = F.softmax(self.output_longitudinal(expert_output), dim=-1)
#
#     # Feature Fusion for Trajectory Prediction
#     # [B, hidden_dim + lat_dim + lon_dim]
#     dec = torch.cat((expert_output, output_lat, output_lon), dim=-1)
#
#     # Trajectory Projection:
#     # output_projection(dec) -> [B, L * 5]
#     # view -> [B, L, 5] (Assuming hs.shape[1] is Batch size B)
#     # Note: hs.shape[1:3] suggests hs might be [Layers, B, Hidden]. view(*hs.shape[1:3]...) might be aiming for [B, Hidden_Part, ...] but standard is [B, L, 5]
#     output = self.output_projection(dec).view(*hs.shape[1:3], self.args.pred_length, 5)
#
#     # Final Activation (e.g., tanh/sigmoid based on implementation)
#     output_traj = out_activation(output)
#
#     return output_traj, output_lat, output_lon
