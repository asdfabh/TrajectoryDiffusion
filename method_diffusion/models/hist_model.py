from method_diffusion.models import dit_hist as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from pathlib import Path
from method_diffusion.utils.visualization import visualize_batch_trajectories, plot_traj_with_mask

class DiffusionPast(nn.Module):

    def __init__(self, args):
        super(DiffusionPast, self).__init__()
        # Net parameters
        self.args = args
        self.feature_dim = 2 * int(args.feature_dim) + 1# 输入特征维度 default: 6 (x, y, v, a, laneID, class)
        self.input_dim = int(args.input_dim)   # 输入到Dit的维度 default: 128
        self.hidden_dim = int(args.hidden_dim)
        self.output_dim = int(args.output_dim)
        self.heads = int(args.heads)
        self.dropout = args.dropout
        self.depth = int(args.depth)
        self.mlp_ratio = args.mlp_ratio
        self.num_train_timesteps = args.num_train_timesteps
        self.time_embedding_size = args.time_embedding_size
        self.num_inference_steps = args.num_inference_steps
        self.T = int(args.T)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            time_embedder=self.timestep_embedder,
            depth=self.depth,
            model_type="x_start"
        )

        self.norm_config_path = str(Path(__file__).resolve().parent.parent / 'dataset/ngsim_stats.npz')
        self.norm_config = np.load(self.norm_config_path)
        for key, value in self.norm_config.items():
            self.register_buffer(key, torch.from_numpy(value).float())

    def compute_motion_loss(self, pred, target, mask):
        """
        pred: [B, T, D]
        target: [B, T, D]
        mask: [B, T, 1] (1 for known/observed, 0 for unknown/masked)
        """
        B, T, D = pred.shape

        mask = mask.view(B, T, -1)
        if mask.shape[-1] == 1:
            mask = mask.expand(B, T, D)
        mask = mask.to(pred.device).float()

        loss_mse = (pred - target) ** 2

        # Known (Observed) Loss
        loss_known = (loss_mse * mask).sum() / (mask.sum() + 1e-6)
        # Unknown (Masked) Loss
        mask_unknown = 1.0 - mask
        loss_unknown = (loss_mse * mask_unknown).sum() / (mask_unknown.sum() + 1e-6)

        pred_pos = pred[..., :2]
        target_pos = target[..., :2]

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        loss_vel = ((pred_vel - target_vel) ** 2).mean()

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        loss_acc = ((pred_acc - target_acc) ** 2).mean()

        total_loss = loss_known + 1.5 * loss_unknown + 0.7 * loss_vel
        return total_loss

    # hist: [B, T, dim], hist_masked: [B, T, dim+1]
    def forward_train(self, hist, hist_masked, device):
        B, T,  _ = hist_masked.shape

        hist_mask = hist_masked[..., -1:]  # [B, T, 1]
        hist_masked_value = hist_masked[..., :-1]  # [B, T, dim]
        hist_masked_norm = self.norm(hist_masked_value) # [B, T, dim]
        x_start = hist_masked_norm # [B, T, dim]
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)
        model_input = torch.cat([x_noisy, hist_masked], dim=-1)  # [B, T, 2*dim+1]

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, t=timesteps)

        mask = hist_mask.unsqueeze(-1)
        loss = self.compute_motion_loss(pred_x0, self.norm(hist), mask)
        # loss = torch.nn.functional.mse_loss(pred_x0, self.norm(hist))

        pred = self.denorm(pred_x0)
        diff = pred[..., :2] - hist[..., :2]
        dist = torch.norm(diff, dim=-1) # [B, T]

        ade = dist.mean()
        fde = dist[:, -1].mean()

        # hist = hist[0, :, :2].detach().cpu().numpy()
        # hist_masked = hist_masked[0, :, :2].detach().cpu().numpy()
        # pred_ego = pred[0, :, :2].detach().cpu().numpy()
        # plot_traj_with_mask(
        #     hist_original=[hist],
        #     hist_masked=[hist_masked],
        #     hist_pred=[pred_ego],
        #     fig_num1=1,
        #     fig_num2=1,
        # )

        return loss, pred, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_masked, device):
        B, T, _ = hist_masked.shape

        hist_masked_norm = self.norm(hist_masked)
        hist_mask = hist_masked[..., -1:]
        x_start = hist_masked_norm[..., :4]  # [B, T, dim]
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_t = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)

        self.diffusion_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.diffusion_scheduler.timesteps:
            model_input = torch.cat((x_t, hist_masked), dim=-1)
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
            pred_x0_norm = self.dit(x=input_embedded, t=timesteps)
            x_t = self.diffusion_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        final_pred = self.denorm(x_t)
        loss = torch.nn.functional.mse_loss(final_pred, hist)
        diff = final_pred[..., :2] - hist[..., :2]
        dist = torch.norm(diff, dim=-1)  # [B, T]
        ade = dist.mean()  # Scalar
        fde = dist[:, -1].mean() # Scalar

        # hist = hist[0, :, :2].detach().cpu().numpy()
        # hist_masked = hist_masked[0, :, :2].detach().cpu().numpy()
        # pred_ego = final_pred[0, :, :2].detach().cpu().numpy()
        #
        # plot_traj_with_mask(
        #     hist_original=[hist],
        #     hist_masked=[hist_masked],
        #     hist_pred=[pred_ego],
        #     fig_num1=1,
        #     fig_num2=1,
        # )

        return loss, final_pred, ade, fde

    def forward(self, hist, hist_masked, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_masked, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean  # v, a
        return x_denorm
