from torch.nn.init import xavier_normal

import method_diffusion.models.hist_encoder
from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch

from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from pathlib import Path
from method_diffusion.utils.visualization import visualize_batch_trajectories, plot_traj_with_mask, plot_traj

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
        self.T = int(args.T_f)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)
        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            depth=self.depth,
            model_type="x_start"
        )

        self.register_buffer('pos_mean', torch.tensor([0.0, 20.0]).float())
        self.register_buffer('pos_std', torch.tensor([10.0, 80.0]).float())

    def compute_motion_loss(self, pred, target):
        """
        pred: [B, T, D]
        target: [B, T, D]
        """
        loss_l1 = torch.abs(pred - target).mean() # L1 Loss

        pred_pos = pred[..., :2]
        target_pos = target[..., :2]

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        loss_vel = torch.abs(pred_vel - target_vel).mean() # L1 Loss

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        loss_acc = torch.abs(pred_acc - target_acc).mean()

        total_loss = loss_l1 + 0.7 * loss_vel + 0.2 * loss_acc
        return total_loss

    # hist: [B, T, dim], hist_masked: [B, T, dim+1]
    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, dim = future.shape
        future_norm = self.norm(future) # [B, T, 2]
        x_start = future_norm
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        x_noisy = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)
        # visualize_batch_trajectories(hist=hist.unsqueeze(2), future=future.unsqueeze(2), pred=self.denorm(x_noisy.unsqueeze(2)), batch_idx=0)
        model_input = x_noisy  # [B, T, 2]

        context, hist_enc = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)  # [B, T, hidden_dim]
        t_emb = self.timestep_embedder(timesteps)
        enc_emb = self.enc_embedding(hist_enc).permute(1, 0, 2).mean(dim=1)  # [B, D]
        y = t_emb + enc_emb

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, y=y, cross=context)

        pred = self.denorm(pred_x0)
        loss = self.compute_motion_loss(pred, future)

        diff = pred[..., :2] - future[..., :2]
        dist = torch.norm(diff, dim=-1) # [B, T]

        ade = dist.mean()
        fde = dist[:, -1].mean()

        # Visualize
        # hist = hist.unsqueeze(2)  # [B, T, 1, D]
        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # hist = torch.cat([hist, hist_nbrs], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist, future=future, pred=pred, batch_idx=0)

        return loss, pred, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, dim = future.shape
        x_start = torch.randn((B, T, dim), device=device)
        x_t = x_start
        # visualize_batch_trajectories(hist=hist.unsqueeze(2), future=future.unsqueeze(2), pred=self.denorm(x_start.unsqueeze(2)), batch_idx=0)
        context, hist_enc = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)  # [B, T, hidden_dim]
        enc_emb = self.enc_embedding(hist_enc).permute(1, 0, 2).mean(dim=1)  # [B, D]

        self.diffusion_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.diffusion_scheduler.timesteps:
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            y = t_emb + enc_emb
            input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
            pred_x0_norm = self.dit(x=input_embedded, y=y, cross=context)
            x_t = self.diffusion_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        pred = self.denorm(x_t)
        loss = torch.nn.functional.mse_loss(pred, future)

        diff = pred[..., :2] - future[..., :2]
        dist = torch.norm(diff, dim=-1) # [B, T]

        ade = dist.mean()
        fde = dist[:, -1].mean()

        # visualize
        # hist = hist.unsqueeze(2)  # [B, T, 1, D]
        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # hist = torch.cat([hist, hist_nbrs], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist, future=future, pred=pred, batch_idx=0)
        return loss, pred, ade, fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_nbrs, mask, temporal_mask, future, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        return x_denorm

