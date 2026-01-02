import torch
import torch.nn as nn
from diffusers.schedulers import DDIMScheduler
from method_diffusion.models import dit
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding

"""
编码历史轨迹: [B, T, N, 4] -> [B, N, Hidden]
"""
class AgentEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.T = int(args.T)
        self.input_dim = int(args.feature_dim)
        self.hidden_dim = int(args.hidden_dim_fut)  # 注意这里用 hidden_dim_fut 统一维度
        self.Encoder_depth = int(args.depth)  # 使用 depth 参数

        self.pre_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.mixer_blocks = nn.ModuleList([
            dit.MixerBlock(tokens_dim=self.T, channels_dim=self.hidden_dim)
            for _ in range(self.Encoder_depth)
        ])

        self.pooling_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=4, batch_first=True)

        self.rel_pos_bias = dit.RelativePositionBias(num_heads=4)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

    def forward(self, x, current_pos, agent_mask=None):
        """
        x: [B, T, N, D] (预处理后的相对数据)
        current_pos: [B, N, 2] (用于计算相对位置矩阵)
        """
        B, T, N, D = x.shape

        # [B, T, N, D] -> [B*N, T, D]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, D)
        x = self.pre_proj(x) # [B*N, T, H]

        for block in self.mixer_blocks:
            x = block(x)  # [B*N, T, H]

        last_frame_feat = x[:, -1, :].unsqueeze(1)  # [B*N, 1, H]
        x_pooled, _ = self.pooling_attn(last_frame_feat, x, x)
        x = x_pooled.squeeze(1)  # [B*N, H]

        x = x.view(B, N, -1)  # [B, N, H]
        resid = x
        x_norm = self.fusion_norm(x)

        rel_pos_matrix = current_pos.unsqueeze(1) - current_pos.unsqueeze(2)
        spatial_bias = self.rel_pos_bias(rel_pos_matrix)

        key_padding_mask = None
        if agent_mask is not None:
            key_padding_mask = ~agent_mask.bool()

        attn_mask = spatial_bias.reshape(B * 4, N, N)
        x_interacted, _ = self.fusion_attn(x_norm, x_norm, x_norm,
                                           key_padding_mask=key_padding_mask,
                                           attn_mask=attn_mask)  # 注入相对位置信息

        x = resid + x_interacted
        x = x + self.fusion_mlp(x)

        return x  # [B, N, H]

class TrajectoryProcessor(nn.Module):
    def __init__(self, pos_scale=4.0, pos_abs_scale=80.0):
        super().__init__()
        # 定义 Buffer 避免 device 问题
        self.register_buffer('pos_scale', torch.tensor(pos_scale))
        self.register_buffer('pos_abs_scale', torch.tensor(pos_abs_scale))

    def hist_process(self, hist_abs, current_pos):
        # hist_abs: [B, T, N, 4] (x, y, v, a), current_pos: [B, 1, N, 2] (t=0 时刻的绝对坐标)
        vel = hist_abs[..., 2:4]  # 假设输入数据里直接有 v (即差分)
        rel_pos = hist_abs[..., :2] - current_pos

        vel_norm = vel / self.pos_scale
        rel_pos_norm = rel_pos / self.pos_abs_scale

        return torch.cat([vel_norm, rel_pos_norm], dim=-1)  # [B, T, N, 4]

    def fut_abs2rel(self, future_abs, current_pos):
        # future_abs: [B, T, N, 2]
        temp = torch.cat([current_pos, future_abs], dim=1)
        diff = temp[:, 1:] - temp[:, :-1]
        return diff / self.pos_scale # 【B, T, N, 2】 (Normalized Deltas)

    def fut_rel2abs(self, pred_rel, current_pos):
        # pred_rel: [B, T, N, 2] (Normalized Deltas)
        diff = pred_rel * self.pos_scale
        cumsum = torch.cumsum(diff, dim=1)
        return cumsum + current_pos # [B, T, N, 2]

class DiffusionFut(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = int(args.hidden_dim_fut)
        self.T_f = int(args.T_f)

        self.processor = TrajectoryProcessor()
        self.hist_encoder = AgentEncoder(args)

        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps_fut,
            beta_schedule="scaled_linear",
            prediction_type="sample",
            clip_sample=False,
        )

        self.dit = dit.DiTFut(
            hidden_dim=self.hidden_dim,
            heads=int(args.heads_fut),
            dropout=args.dropout_fut,
            depth=int(args.depth_fut),
            mlp_ratio=args.mlp_ratio_fut,
            N=int(args.N_f),
            T=self.T_f,
            time_embedder=self.timestep_embedder
        )

    def get_rel_pos_matrix(self, current_pos):
        # current_pos: [B, 1, N, 2]
        pos = current_pos.squeeze(1)  # [B, N, 2]
        # 计算 P_j - P_i (邻居相对于我)
        rel_matrix = pos.unsqueeze(1) - pos.unsqueeze(2)
        return rel_matrix

    def forward_train(self, future, hist, device, agent_mask=None):
        """
        future: [B, T_f, N, 2] Ground Truth Absolute
        hist: [B, T_h, N, 4] Ground Truth Absolute
        """
        B, T, N, _ = future.shape

        current_pos = hist[:, -1:, :, :2]  # [B, 1, N, 2] Anchor
        current_pos_sq = current_pos.squeeze(1)  # [B, N, 2]

        # 1. Encode History
        hist_input = self.processor.hist_process(hist, current_pos)
        hist_feat = self.hist_encoder(hist_input, current_pos_sq, agent_mask)  # [B, N, H]

        # 2. Prepare Target & Flatten
        x_start = self.processor.fut_abs2rel(future, current_pos)  # [B, T, N, 2]
        # Flatten: [B, T, N, 2] -> [B, N, T, 2] -> [B, N, T*2]
        x_start_flat = x_start.permute(0, 2, 1, 3).reshape(B, N, -1)

        # 3. Add Noise
        timesteps = torch.randint(0, self.diffusion_scheduler.config.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(x_start_flat)
        x_noisy_flat = self.diffusion_scheduler.add_noise(x_start_flat, noise, timesteps)

        if agent_mask is not None:
            # agent_mask: [B, N] -> [B, N, 1]
            mask_exp = agent_mask.unsqueeze(-1).float()
            x_noisy_flat = x_noisy_flat * mask_exp

        # 4. DiT Forward
        rel_pos_matrix = self.get_rel_pos_matrix(current_pos)
        # Input: [B, N, T*2] -> Output: [B, N, T*2]
        pred_flat = self.dit(x_noisy_flat, timesteps, hist_feat, rel_pos_matrix, agent_mask)

        # 5. Loss
        target = x_start_flat

        loss = torch.nn.functional.mse_loss(pred_flat, target, reduction='none')

        if agent_mask is not None:
            loss = (loss * mask_exp).sum() / (mask_exp.sum() * (T * 2) + 1e-6)
        else:
            loss = loss.mean()

        pred_reshaped = pred_flat.view(B, N, T, 2).permute(0, 2, 1, 3)
        pred_abs = self.processor.fut_rel2abs(pred_reshaped, current_pos)

        diff = torch.norm(pred_abs - future, dim=-1)  # [B, T, N]

        if agent_mask is not None:
            mask_bool = agent_mask.bool()
            ade = (diff * mask_bool.unsqueeze(1)).sum() / (mask_bool.sum() * T + 1e-6)
            fde = (diff[:, -1] * mask_bool).sum() / (mask_bool.sum() + 1e-6)
        else:
            ade = diff.mean()
            fde = diff[:, -1].mean()

        return loss, pred_abs, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, device, agent_mask=None):
        B, T_h, N, _ = hist.shape
        T_f = self.T_f

        current_pos = hist[:, -1:, :, :2]
        current_pos_sq = current_pos.squeeze(1)

        hist_input = self.processor.hist_process(hist, current_pos)
        hist_feat = self.hist_encoder(hist_input, current_pos_sq, agent_mask)
        rel_pos_matrix = self.get_rel_pos_matrix(current_pos)

        # Init Noise: Flattened [B, N, T*2]
        x_t_flat = torch.randn(B, N, T_f * 2, device=device)

        self.diffusion_scheduler.set_timesteps(self.args.num_inference_steps)

        for t in self.diffusion_scheduler.timesteps:
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            pred_x0_flat = self.dit(x_t_flat, timesteps, hist_feat, rel_pos_matrix, agent_mask)
            x_t_flat = self.diffusion_scheduler.step(pred_x0_flat, t, x_t_flat).prev_sample

        # Unflatten
        pred_reshaped = x_t_flat.view(B, N, T_f, 2).permute(0, 2, 1, 3)
        pred_abs = self.processor.fut_rel2abs(pred_reshaped, current_pos)
        return pred_abs

class TrajectoryModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.fut_model = DiffusionFut(args)

    def forward(self, hist, hist_masked, future, device, agent_mask=None):
        loss, pred, ade, fde = self.fut_model.forward_train(future, hist, device, agent_mask)
        return torch.tensor(0.0), loss, None, pred, 0.0, 0.0, ade, fde

    def inference(self, hist, hist_masked, future, device, agent_mask=None):
        pred_fut = self.fut_model.forward_eval(hist, device, agent_mask)
        return None, pred_fut


# class DiffusionFut(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.T_f = int(args.T_f)
#         self.hidden_dim = int(args.hidden_dim_fut)
#
#         self.processor = TrajectoryProcessor()
#         self.hist_encoder = AgentEncoder(args)
#
#         self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim)
#         self.diffusion_scheduler = DDIMScheduler(
#             num_train_timesteps=args.num_train_timesteps_fut,
#             beta_schedule="scaled_linear",
#             prediction_type="sample",
#             clip_sample=False,
#         )
#
#         self.dit = dit.DiTFut(
#             input_dim=2,  # dx, dy
#             hidden_dim=self.hidden_dim,
#             output_dim=2,
#             heads=int(args.heads_fut),
#             dropout=args.dropout_fut,
#             depth=int(args.depth_fut),
#             mlp_ratio=args.mlp_ratio_fut,
#             N=int(args.N_f),
#             T=self.T_f,
#             time_embedder=self.timestep_embedder
#         )
#
#     def get_rel_pos_matrix(self, current_pos):
#         # current_pos: [B, 1, N, 2]
#         pos = current_pos.squeeze(1)  # [B, N, 2]
#         # [B, 1, N, 2] - [B, N, 1, 2] -> [B, N, N, 2] (P_j - P_i)
#         rel_matrix = pos.unsqueeze(1) - pos.unsqueeze(2)
#         return rel_matrix
#
#     def forward_train(self, future, hist, device, agent_mask=None):
#         """
#         future: [B, T_f, N, 2] Ground Truth Absolute
#         hist: [B, T_h, N, 4] Ground Truth Absolute
#         """
#         B, T, N, _ = future.shape
#
#         current_pos = hist[:, -1:, :, :2]  # [B, 1, N, 2] Anchor
#         current_pos_sq = current_pos.squeeze(1)  # [B, N, 2]
#
#         hist_input = self.processor.hist_process(hist, current_pos)
#         hist_feat = self.hist_encoder(hist_input, current_pos_sq, agent_mask)  # [B, N, H]
#
#         x_start = self.processor.fut_abs2rel(future, current_pos)  # [B, T, N, 2]
#
#         rel_pos_matrix = self.get_rel_pos_matrix(current_pos)  # [B, N, N, 2]
#
#         B, T, N, D = x_start.shape
#         x_start_flat = x_start.permute(0, 2, 1, 3).reshape(B, N, -1)
#
#         timesteps = torch.randint(0, self.diffusion_scheduler.config.num_train_timesteps, (B,), device=device)
#         noise = torch.randn_like(x_start_flat)
#         x_noisy_flat = self.diffusion_scheduler.add_noise(x_start_flat, noise, timesteps)
#
#         if agent_mask is not None:
#             x_noisy_flat = x_noisy_flat * agent_mask.unsqueeze(-1)
#
#         pred_flat = self.dit(x_noisy_flat, timesteps, hist_feat, rel_pos_matrix, current_pos_sq, agent_mask)
#
#         loss = torch.nn.functional.mse_loss(pred_flat, x_start_flat, reduction='none')
#
#         if agent_mask is not None:
#             loss = (loss * mask_exp).sum() / (mask_exp.sum() + 1e-6)
#         else:
#             loss = loss.mean()
#
#         # Metrics (ADE/FDE in Absolute Coords)
#         pred_abs = self.processor.fut_rel2abs(pred_noise, current_pos)
#         diff = torch.norm(pred_abs - future, dim=-1)
#
#         if agent_mask is not None:
#             mask_bool = agent_mask.bool()
#             ade = (diff * mask_bool.unsqueeze(1)).sum() / (mask_bool.sum() * T + 1e-6)
#             fde = (diff[:, -1] * mask_bool).sum() / (mask_bool.sum() + 1e-6)
#         else:
#             ade = diff.mean()
#             fde = diff[:, -1].mean()
#
#         return loss, pred_abs, ade, fde
#
#     @torch.no_grad()
#     def forward_eval(self, hist, device, agent_mask=None):
#         B, T_h, N, _ = hist.shape
#         T_f = self.dit.T
#
#         current_pos = hist[:, -1:, :, :2]
#         current_pos_sq = current_pos.squeeze(1)
#
#         hist_input = self.processor.hist_process(hist, current_pos)
#         hist_feat = self.hist_encoder(hist_input, current_pos_sq, agent_mask)
#         rel_pos_matrix = self.get_rel_pos_matrix(current_pos)
#
#         x_t = torch.randn(B, T_f, N, 2, device=device)
#
#         self.diffusion_scheduler.set_timesteps(self.args.num_inference_steps)
#
#         for t in self.diffusion_scheduler.timesteps:
#             timesteps = torch.full((B,), t, device=device, dtype=torch.long)
#             pred_x0 = self.dit(x_t, timesteps, hist_feat, rel_pos_matrix, current_pos.squeeze(1), agent_mask)
#             x_t = self.diffusion_scheduler.step(pred_x0, t, x_t).prev_sample
#
#         pred_abs = self.processor.fut_rel2abs(x_t, current_pos)
#         return pred_abs















