from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
from method_diffusion.models.hist_encoder import HistEncoder
import torch.nn.functional as F
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
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
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)
        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)

        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="scaled_linear",
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
        self.anchor_path = str(root_path / 'dataset/anchors_ngsim.npy')

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

        total_loss = 2.0 * reg_loss + 1.2 * vel_loss + 1.0 * cls_loss
        return total_loss, reg_loss, vel_loss, cls_loss

    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape
        T_h = hist.shape[1]
        K = self.num_modes
        """重点：anchor的聚类需要仔细调整，加噪后anchor的分布也需要考虑"""
        self.anchors = self.anchors.to(device)

        # 1. add truncated noise to the plan anchor
        # Anchors: [K, T, 2] -> [B, K, T, 2]
        anchors_norm = self.norm(self.anchors)[..., :2]  # [K, T, 2]
        timesteps = torch.randint(0, 50, (B,), device=device).long()

        # 广播视图加噪 [B, K, T, 2]
        noise = torch.randn(B, K, T, 2, device=device)
        x_noisy = self.diffusion_scheduler.add_noise(
            anchors_norm.unsqueeze(0).expand(B, -1, -1, -1),  # 虚拟扩展
            noise,
            timesteps
        ).float()
        x_noisy = torch.clamp(x_noisy, -3, 3)
        x_noisy_denorm = self.denorm(x_noisy)

        # 2. proj noisy_traj_points to the query
        x_sine = gen_sineembed_for_position_adaptive(
            x_noisy_denorm, hidden_dim=self.input_dim,
            max_len_x=20.0, max_len_y=300.0
        )  # [B, K, T, input_dim]
        x_input = self.query_encoder(x_sine.view(B * K, T, -1)) + self.pos_embedding(x_sine.view(B * K, T, -1))

        # 3. embed the timesteps and history context
        t_emb = self.timestep_embedder(timesteps)  # [B, D]

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]
        enc_emb = self.enc_embedding(global_context)  # [B, D]

        y_per_batch = (t_emb + enc_emb).unsqueeze(1)  # [B, 1, D]
        y_expanded = y_per_batch.expand(-1, K, -1).reshape(B * K, -1)

        context_expanded = context.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T_h, -1)

        # pred_traj_delta: [B*K, T, 2]，直接预测轨迹 pred_scores_flat: [B*K, 1] 表示后验分数
        pred_flat, pred_scores_flat = self.dit(x=x_input, y=y_expanded, cross=context_expanded)
        pred_x0 = pred_flat.view(B, K, T, 2)  # [B, K, T, 2]
        cls_logits = pred_scores_flat.view(B, K)  # [B, K] 后验分数

        # Best Label (哪个 Anchor 离 GT 最近)
        target_phys = future[..., :2]  # [B, T, 2]

        """重点，选取最佳轨迹需要仔细调参，选取错误会导致严重的训练问题"""
        target_norm = self.norm(future)[..., :2]
        anchors_norm = self.norm(self.anchors)[..., :2]
        dist = torch.norm(anchors_norm.unsqueeze(0) - target_norm.unsqueeze(1), dim=-1).mean(dim=-1)
        target_mode_idx = torch.argmin(dist, dim=1)

        # 只计算 Best Anchor 对应生成的轨迹的 L1 Loss
        batch_idx = torch.arange(B, device=device)
        pred = pred_x0[batch_idx, target_mode_idx]  # [B, T, 2]

        target = self.norm(target_phys)[..., :2]
        total_loss = self.compute_loss(pred, target, cls_logits, target_mode_idx)[0]


        # Metrics
        pred_phys = self.denorm(pred)
        diff = pred_phys - target_phys
        ade = torch.norm(diff, dim=-1).mean()
        fde = torch.norm(diff, dim=-1)[:, -1].mean()

        """
        目前存在的问题：
        best曲线选择的真值不够准，导致回归和分类都出现问题
        anchor聚类效果不够好，导致生成的轨迹离实际轨迹较远
        生成的轨迹存在摆烂行为，趋于均值，这是很严重的问题
        生成的轨迹没有时间关系，导致不连贯
        """
        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs_aligned = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # full_gt_hist = torch.cat([hist.unsqueeze(2), hist_nbrs_aligned], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=self.anchors.unsqueeze(0).permute(0, 2, 1, 3), best_index=target_mode_idx)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=self.denorm(pred_x0).permute(0, 2, 1, 3), best_index=target_mode_idx)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=pred_phys)

        return total_loss, pred_phys, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape
        T_h = hist.shape[1]
        K = self.num_modes
        self.anchors = self.anchors.to(device)

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]
        enc_emb = self.enc_embedding(global_context)  # [B, D]

        context_expanded = context.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T_h, -1)

        start_step = 20
        anchors_norm = self.norm(self.anchors)[..., :2]  # [K, T, 2]
        noise = torch.randn(B, K, T, 2, device=device)

        x_t_norm = self.diffusion_scheduler.add_noise(
            anchors_norm.unsqueeze(0).expand(B, -1, -1, -1),
            noise,
            torch.tensor([start_step], device=device).long()
        )

        inference_steps = [20, 10, 0]
        final_scores = None
        self.diffusion_scheduler.set_timesteps(self.num_train_timesteps)

        for t_val in inference_steps:
            timesteps = torch.full((B,), t_val, device=device, dtype=torch.long)

            x_t_denorm = self.denorm(x_t_norm)
            x_sine = gen_sineembed_for_position_adaptive(
                x_t_denorm, hidden_dim=self.input_dim,
                max_len_x=20.0, max_len_y=300.0
            )
            x_input = self.query_encoder(x_sine.view(B * K, T, -1)) + self.pos_embedding(x_sine.view(B * K, T, -1))

            # 时间步嵌入和环境条件融合
            t_emb = self.timestep_embedder(timesteps)  # [B, D]
            y_per_batch = (t_emb.unsqueeze(1) + enc_emb.unsqueeze(1))  # [B, 1, D]
            y_expanded = y_per_batch.expand(-1, K, -1).reshape(B * K, -1)

            pred_flat, pred_scores_flat = self.dit(x=x_input, y=y_expanded, cross=context_expanded)
            pred_x0_norm = pred_flat.view(B, K, T, 2)

            if t_val > 0:
                x_t_norm_flat = x_t_norm.view(B * K, T, 2)
                pred_x0_norm_flat = pred_x0_norm.view(B * K, T, 2)

                step_out = self.diffusion_scheduler.step(
                    model_output=pred_x0_norm_flat,
                    timestep=t_val,
                    sample=x_t_norm_flat
                )
                x_t_norm = step_out.prev_sample.view(B, K, T, 2)
            else:
                x_t_norm = pred_x0_norm
                final_scores = pred_scores_flat.view(B, K)

        all_preds_phys = self.denorm(x_t_norm)  # [B, K, T, 2]

        best_mode_idx = final_scores.argmax(dim=-1)  # [B]
        batch_idx = torch.arange(B, device=device)
        final_traj = all_preds_phys[batch_idx, best_mode_idx]  # [B, T, 2]

        target_phys = future[..., :2]
        dist = torch.norm(final_traj - target_phys, dim=-1).mean(dim=-1)
        ade = dist.mean()

        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs_aligned = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # full_gt_hist = torch.cat([hist.unsqueeze(2), hist_nbrs_aligned], dim=2)  # [B, T, 1+N, D]
        # target_norm = self.norm(future)[..., :2]
        # anchors_norm = self.norm(self.anchors)[..., :2]
        # dist = torch.norm(anchors_norm.unsqueeze(0) - target_norm.unsqueeze(1), dim=-1).mean(dim=-1)
        # target_mode_idx = torch.argmin(dist, dim=1)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=self.anchors.unsqueeze(0).permute(0, 2, 1, 3), best_index=target_mode_idx)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=self.anchors.unsqueeze(0).permute(0, 2, 1, 3), best_index=best_mode_idx)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=self.denorm(x_t_norm).permute(0, 2, 1, 3), best_index=best_mode_idx)
        # visualize_batch_trajectories(hist=hist, hist_nbrs=full_gt_hist, future=future, pred=final_traj)

        return torch.tensor(0.0), final_traj, ade, torch.tensor(0.0)

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
