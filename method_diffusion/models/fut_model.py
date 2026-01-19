from torch.nn.init import xavier_normal
import method_diffusion.models.hist_encoder
from method_diffusion.models import dit_fut as dit
from torch import nn
# from diffusers.schedulers import DDIMScheduler # EDM 不需要 DDIMScheduler
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
        self.feature_dim = int(args.feature_dim_fut)
        self.input_dim = int(args.input_dim_fut)
        self.hidden_dim = int(args.hidden_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.dropout = args.dropout_fut
        self.depth = int(args.depth_fut)
        self.mlp_ratio = args.mlp_ratio_fut
        # self.num_train_timesteps = args.num_train_timesteps_fut # EDM 不再依赖离散步数
        self.time_embedding_size = args.time_embedding_size_fut
        self.num_inference_steps = args.num_inference_steps
        self.T = int(args.T_f)

        self.sigma_min = 0.002  # 最小噪声 (对应 clean data)
        self.sigma_max = 80.0  # 最大噪声 (对应纯噪声，解决鸿沟的关键)
        self.sigma_data = 0.5  # 数据集的期望标准差 (Norm之后通常设为0.5)
        self.P_mean = -0.5  # Log-Normal 采样的均值 -1.2
        self.P_std = 1.4  # Log-Normal 采样的方差 1.2
        self.rho = 7  # 推理时的采样曲线参数

        # 输入嵌入层和位置编码
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)
        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            depth=self.depth,
            model_type="x_start"
        )

        self.register_buffer('pos_mean', torch.tensor([0.0, 0.0]).float(), persistent=False)
        self.register_buffer('pos_std', torch.tensor([10, 150]).float(), persistent=False)
        self.register_buffer('va_mean', torch.tensor([20, 0.01]).float(), persistent=False)
        self.register_buffer('va_std', torch.tensor([15, 5]).float(), persistent=False)

    def get_edm_scalings(self, sigma):
        """
        根据 EDM 理论计算 Preconditioning 系数
        Input: sigma [B]
        Output: c_skip, c_out, c_in, c_noise
        EDM 论文建议将 ln(sigma) / 4 作为输入给 timestep embedding
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_noise = sigma.log() / 4.0
        return c_skip, c_out, c_in, c_noise

    # def compute_motion_loss(self, pred, target):
    #     """
    #     pred: [B, T, D]
    #     target: [B, T, D]
    #     """
    #     loss_l1 = torch.abs(pred[..., :2] - target[..., :2]).mean()
    #
    #     pred_pos = pred[..., :2]
    #     target_pos = target[..., :2]
    #
    #     pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
    #     target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
    #     loss_vel = torch.abs(pred_vel - target_vel).mean()
    #
    #     pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
    #     target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
    #     loss_acc = torch.abs(pred_acc - target_acc).mean()
    #
    #     total_loss = 1.2 * loss_l1 + 0.5 * loss_vel + 0.15 * loss_acc
    #     return total_loss

    def compute_motion_loss(self, pred, target):
        """
        pred: [B, T, D]
        target: [B, T, D]
        """
        # loss_pos = torch.nn.functional.mse_loss(pred[..., :2], target[..., :2])
        loss_pos = torch.nn.functional.smooth_l1_loss(pred[..., :2], target[..., :2], beta=0.1)

        pred_pos = pred[..., :2]
        target_pos = target[..., :2]

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]

        # 辅助损失考虑保持 L1 或改为 Smooth L1 (Huber Loss)
        # 纯 MSE 对异常值太敏感，L1 能让轨迹动态更稳健
        loss_vel = torch.abs(pred_vel - target_vel).mean()
        loss_acc = torch.abs(pred_acc - target_acc).mean()

        total_loss = 5.0 * loss_pos + 1.0 * loss_vel + 0.5 * loss_acc

        return total_loss

    # 使用 Log-Normal 采样 + Preconditioning
    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, dim = future.shape
        future_norm = self.norm(future)  # Ground Truth
        x_start = future_norm

        # 这种分布会让训练集中在 loss 梯度最大的中间区域，同时 sigma_max=80 保证了模型见过纯噪声
        rnd_normal = torch.randn([B, 1, 1], device=device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        noise = torch.randn_like(x_start)
        x_noisy = x_start + noise * sigma

        c_skip, c_out, c_in, c_noise = self.get_edm_scalings(sigma)

        model_input = c_in * x_noisy

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.enc_embedding(hist_enc[:, -1, :])

        t_emb = self.timestep_embedder(c_noise.view(B))
        y = t_emb + enc_emb

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        F_x = self.dit(x=input_embedded, y=y, cross=context)  # 网络输出 F_theta

        # D(x) = c_skip * input + c_out * F(input),这一步保证了边界条件：当 sigma 很大时，D(x) 趋向于 0 (均值)
        pred_x0 = c_skip * x_noisy + c_out * F_x

        loss = self.compute_motion_loss(pred_x0, future_norm)

        pred = self.denorm(pred_x0.detach())
        diff = pred[..., :2] - future[..., :2]
        dist = torch.norm(diff, dim=-1)
        ade = dist.mean()
        fde = dist[:, -1].mean()

        # hist = hist.unsqueeze(2)  # [B, T, 1, D]
        # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
        # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist.size(1), -1)
        # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
        # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
        # hist_nbrs = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
        # hist = torch.cat([hist, hist_nbrs], dim=2)  # [B, T, 1+N, D]
        # visualize_batch_trajectories(hist=hist, future=future, pred=pred, batch_idx=0)

        return loss, pred, ade, fde

    # 使用 EDM 确定性采样 (Euler Step)
    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, dim = future.shape

        # 在 EDM 中，纯噪声是 N(0, sigma_max^2)，所以要乘以 sigma_max, 这保证了推理起点和训练时的 "纯噪样本" 分布一致
        x_t = torch.randn((B, T, dim), device=device) * self.sigma_max

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.enc_embedding(hist_enc[:, -1, :])

        # 使用 Karras 的多项式 Schedule，步长在高噪区大，低噪区小
        num_steps = self.num_inference_steps
        step_indices = torch.arange(num_steps, device=device, dtype=torch.float64)

        # 公式: (sigma_max^(1/rho) + t * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # 最后加个 0

        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]

            sigma_cur = torch.full((B, 1, 1), t_cur, device=device)
            c_skip, c_out, c_in, c_noise = self.get_edm_scalings(sigma_cur)
            model_input = c_in * x_t
            t_emb = self.timestep_embedder(c_noise.view(B))
            y = t_emb + enc_emb

            input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
            F_x = self.dit(x=input_embedded, y=y, cross=context)

            # 得到去噪预测 D(x)
            denoised = c_skip * x_t + c_out * F_x

            # d = (x - D(x)) / sigma, 求解OED
            d_cur = (x_t - denoised) / t_cur

            # x_next = x_cur + (t_next - t_cur) * d
            x_t = x_t + (t_next - t_cur) * d_cur

        pred = self.denorm(x_t)
        loss = torch.nn.functional.mse_loss(pred, future)  # Eval 时的简单 MSE

        diff = pred[..., :2] - future[..., :2]
        dist = torch.norm(diff, dim=-1)

        ade = dist.mean()
        fde = dist[:, -1].mean()

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