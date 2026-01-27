from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
from method_diffusion.models.hist_encoder import HistEncoder
import torch.nn.functional as F
from pathlib import Path
from method_diffusion.utils.visualization import visualize_batch_trajectories, plot_traj_with_mask, plot_traj
import math

def gen_sineembed_for_position_adaptive(pos_tensor, hidden_dim=128, max_len_x=20.0, max_len_y=300.0):
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    device = pos_tensor.device

    dim_t_steps = torch.arange(half_hidden_dim, dtype=torch.float32, device=device)
    exponent = 2 * (dim_t_steps // 2) / half_hidden_dim

    dim_t_x = max_len_x ** exponent
    dim_t_y = max_len_y ** exponent

    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale

    pos_x = x_embed[..., None] / dim_t_x
    pos_y = y_embed[..., None] / dim_t_y

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        # Net parameters
        self.args = args
        self.feature_dim = int(args.feature_dim_fut) # 输入特征维度 default: 6 (x, y, v, a, laneID, class)
        self.input_dim = int(args.input_dim_fut)   # 输入到Dit的维度 default: 128
        self.hidden_dim = int(args.hidden_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.dropout = args.dropout_fut
        self.depth = int(args.depth_fut)
        self.mlp_ratio = args.mlp_ratio_fut
        self.num_train_timesteps = args.num_train_timesteps_fut
        self.time_embedding_size = args.time_embedding_size_fut
        self.num_inference_steps = args.num_inference_steps
        self.num_modes = args.num_modes  # 聚类数
        self.T = int(args.T_f)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.query_encoder = nn.Sequential(nn.Linear(self.input_dim, self.input_dim), nn.SiLU(),
            nn.Linear(self.input_dim, self.input_dim), nn.LayerNorm(self.input_dim))
        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)

        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.JointFinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            depth=self.depth,
            model_type="x_start"
        )

        self._init_anchors()

        nn.init.zeros_(self.final_layer.traj_head[-1].weight)
        nn.init.zeros_(self.final_layer.traj_head[-1].bias)

        self.register_buffer('pos_mean', torch.tensor([0.0, 0.0]).float(), persistent=False)
        self.register_buffer('pos_std', torch.tensor([8, 120]).float(), persistent=False)
        self.register_buffer('va_mean', torch.tensor([20, 0.01]).float(), persistent=False)
        self.register_buffer('va_std', torch.tensor([15, 5]).float(), persistent=False)

    def _init_anchors(self):
        # 使用 args 中的路径或根据 root 动态推导
        root_path = Path(__file__).resolve().parent.parent
        self.anchor_path = str(root_path / 'dataset/ngsim_anchors_k10.npy')

        if Path(self.anchor_path).exists():
            anchors_np = np.load(self.anchor_path)  # [K, T, 2]
            anchors_tensor = torch.from_numpy(anchors_np).float()
            self.anchors = anchors_tensor
            print(f"Loaded anchors from {self.anchor_path}, shape: {self.anchors.shape}")
        else:
            print(f"Warning: Anchor file not found at {self.anchor_path}, using zeros.")
            self.anchors = torch.zeros(self.num_modes, self.T, 2)
        # visualize_batch_trajectories(hist=None, hist_nbrs=self.anchors.unsqueeze(0).permute(0, 2, 1, 3))

    def get_closest_anchor(self, future):
        """
        计算 GT 与所有 Anchor 的距离，返回最近 Anchor 的索引
        future: [B, T, 2]
        anchors: [K, T, 2]
        return: [B] (indices)
        """
        # [B, 1, T, 2] - [1, K, T, 2] -> [B, K, T, 2]
        # 使用 denorm 后的距离或者 norm 后的距离都可以，这里用 norm 后的
        future_norm = self.norm(future)[..., :2]
        anchors_norm = self.norm(self.anchors)[..., :2]

        diff = future_norm.unsqueeze(1) - anchors_norm.unsqueeze(0)
        dist = torch.norm(diff, dim=-1).mean(dim=-1)  # [B, K]
        min_dist, min_idx = torch.min(dist, dim=1)  # [B]
        return min_idx

    def compute_loss(self, pred_x0, target_norm, cls_logits, target_mode_idx):
        """
        pred_x0: [B, T, 2] 预测的轨迹
        target_norm: [B, T, 2] GT
        cls_logits: [B, K] 分类 logits
        target_mode_idx: [B] GT 对应的最近 anchor 索引
        """
        reg_loss = F.l1_loss(pred_x0, target_norm)  # 或者 MSE
        cls_loss = F.cross_entropy(cls_logits, target_mode_idx)

        pred_pos = pred_x0[..., :2]
        target_pos = target_norm[..., :2]

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        vel_loss = torch.abs(pred_vel - target_vel).mean() # L1 Loss

        total_loss = 1.2 * reg_loss + 0.6 * vel_loss + 0.5 * cls_loss
        return total_loss, reg_loss, vel_loss, cls_loss

    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape
        K = self.num_modes
        self.anchors = self.anchors.to(device)

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]
        enc_emb = self.enc_embedding(global_context)  # [B, D]

        # Context/Condition: [B, ...] -> [B*K, ...]
        context_expanded = context.repeat_interleave(K, dim=0)  # [B*K, T_hist, D]
        enc_emb_expanded = enc_emb.repeat_interleave(K, dim=0)  # [B*K, D]

        # Anchors: [K, T, 2] -> [B, K, T, 2] -> [B*K, T, 2]
        batch_anchors = self.anchors.unsqueeze(0).repeat(B, 1, 1, 1)
        batch_anchors_norm = self.norm(batch_anchors)[..., :2]

        # Timesteps: 每个 Batch 采样一个 t，K 个 anchor 共享
        timesteps = torch.randint(0, 50, (B,), device=device).long()
        timesteps_expanded = timesteps.repeat_interleave(K)  # [B*K]

        noise = torch.randn_like(batch_anchors_norm)  # [B, K, T, 2]
        x_start_flat = batch_anchors_norm.view(B * K, T, 2)
        noise_flat = noise.view(B * K, T, 2)

        x_noisy_flat = self.diffusion_scheduler.add_noise(
            x_start_flat,
            noise_flat,
            timesteps_expanded
        )  # [B*K, T, 2]

        # Denorm -> Sine Embed
        x_noisy_phys = self.denorm(x_noisy_flat)
        x_sine = gen_sineembed_for_position_adaptive(
            x_noisy_phys, hidden_dim=self.input_dim,
            max_len_x=20.0, max_len_y=300.0
        )
        x_input = self.query_encoder(x_sine)  # [B*K, T, D]

        t_emb = self.timestep_embedder(timesteps_expanded)
        y = t_emb + enc_emb_expanded

        # pred_traj_delta: [B*K, T, 2],表示修正量
        # pred_scores_flat: [B*K, 1]
        pred_traj_delta, pred_scores_flat = self.dit(x=x_input, y=y, cross=context_expanded)
        pred_traj_delta = pred_traj_delta.view(B, K, T, 2)
        input_anchor_coords = x_noisy_flat.view(B, K, T, 2)
        pred_x0 = input_anchor_coords + pred_traj_delta
        cls_logits = pred_scores_flat.view(B, K)  # [B, K] <- 后验分数

        target_phys = future[..., :2]  # [B, T, 2]

        # 找 Label (哪个 Anchor 离 GT 最近)
        batch_anchors_phys = self.anchors.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, K, T, 2]
        dist = torch.norm(batch_anchors_phys - target_phys.unsqueeze(1), dim=-1).mean(dim=-1)  # [B, K]
        target_mode_idx = torch.argmin(dist, dim=1)  # [B]

        # 只计算 Best Anchor 对应生成的轨迹的 L1 Loss
        idx_gather = target_mode_idx.view(B, 1, 1, 1).repeat(1, 1, T, 2)
        best_mode_pred = torch.gather(pred_x0, 1, idx_gather).squeeze(1)  # [B, T, 2]
        target_norm = self.norm(target_phys)[..., :2]

        total_loss = self.compute_loss(best_mode_pred, target_norm, cls_logits, target_mode_idx)[0]

        # Metrics
        pred_phys = self.denorm(best_mode_pred)
        diff = pred_phys - target_phys
        ade = torch.norm(diff, dim=-1).mean()
        fde = torch.norm(diff, dim=-1)[:, -1].mean()

        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs_aligned = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # full_gt_hist = torch.cat([hist.unsqueeze(2), hist_nbrs_aligned], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist_norm, future=self.norm(future), pred=self.norm(self.anchors.unsqueeze(0).permute(0, 2, 1, 3)))
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=pred_phys)

        return total_loss, pred_phys, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape
        K = self.num_modes
        self.anchors = self.anchors.to(device)

        # 1. Encode History
        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]
        enc_emb = self.enc_embedding(global_context)

        # 2. Prepare Inputs (Parallel B*K)
        context_expanded = context.repeat_interleave(K, dim=0)
        enc_emb_expanded = enc_emb.repeat_interleave(K, dim=0)

        # 3. Initialize from Anchors + Small Noise
        batch_anchors = self.anchors.unsqueeze(0).repeat(B, 1, 1, 1)
        x_t_norm = self.norm(batch_anchors)[..., :2].view(B * K, T, 2)

        start_step = 20  # 推理时只加一点点噪声
        noise = torch.randn_like(x_t_norm)
        timesteps = torch.full((B * K,), start_step, device=device, dtype=torch.long)

        x_t_norm = self.diffusion_scheduler.add_noise(x_t_norm, noise, timesteps)

        # 4. Denoising Loop
        inference_steps = [20, 10, 0]
        final_scores = None
        self.diffusion_scheduler.set_timesteps(500, device)

        for i, t_val in enumerate(inference_steps):
            # A. Prepare Input
            current_ts = torch.full((B * K,), t_val, device=device, dtype=torch.long)

            x_t_phys = self.denorm(x_t_norm)
            x_sine = gen_sineembed_for_position_adaptive(
                x_t_phys, hidden_dim=self.input_dim,
                max_len_x=20.0, max_len_y=300.0
            )
            x_input = self.query_encoder(x_sine)

            t_emb = self.timestep_embedder(current_ts)
            y = t_emb + enc_emb_expanded

            # B. Predict x0 and Score
            pred_delta_norm, pred_scores = self.dit(x=x_input, y=y, cross=context_expanded)
            pred_x0_norm = x_t_norm + pred_delta_norm

            # C. Scheduler Step (Using predicted x0)
            if i < len(inference_steps) - 1:
                # DDIM Step needs model_output (x0) and sample (xt)
                output = self.diffusion_scheduler.step(
                    model_output=pred_x0_norm,
                    timestep=t_val,
                    sample=x_t_norm
                )
                x_t_norm = output.prev_sample
            else:
                # Last step
                x_t_norm = pred_x0_norm
                final_scores = pred_scores  # Save scores for selection

        # 5. Selection (Posterior)
        all_preds_phys = self.denorm(x_t_norm).view(B, K, T, 2)
        cls_logits = final_scores.view(B, K)
        cls_probs = F.softmax(cls_logits, dim=-1)

        # 策略: 选分数最高的
        best_mode_idx = cls_probs.argmax(dim=-1)  # [B]
        idx_gather = best_mode_idx.view(B, 1, 1, 1).repeat(1, 1, T, 2)
        final_traj = torch.gather(all_preds_phys, 1, idx_gather).squeeze(1)

        # Calc Metrics (Optional)
        gt = future[..., :2].unsqueeze(1)
        dist = torch.norm(all_preds_phys - gt, dim=-1).mean(dim=-1)
        min_ade = dist.min(dim=1)[0].mean()

        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs_aligned = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # full_gt_hist = torch.cat([hist.unsqueeze(2), hist_nbrs_aligned], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=final_traj)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=all_preds_phys.permute(0, 2, 1, 3))

        return torch.tensor(0.0), final_traj, min_ade, torch.tensor(0.0)

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_nbrs, mask, temporal_mask, future, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        C = x_norm.shape[-1]
        if C == 3:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        C = x.shape[-1]
        if C == 3:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean  # v, a
        return x_denorm

