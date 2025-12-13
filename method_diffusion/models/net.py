from method_diffusion.models import dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
from method_diffusion.loss import DiffusionLoss
import numpy as np
import torch
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from pathlib import Path

"""
反向预测历史轨迹的扩散模型
输入自车，周车的历史轨迹以及掩码，输出自车和周车历史轨迹
hist:[B, T, dim]  hist_mask:[B, T, 1] 内部生成掩码
nbrs:[B'=B*N', T, dim]  nbrs_mask:[B'=N（3*13）, T, dim]内部生成掩码
nbrs每一个批量的数量N堆叠在Batch维度，输入前构建稀疏表达（3*13）
"""

class DiffusionPast(nn.Module):

    def __init__(self, args):
        super(DiffusionPast, self).__init__()
        # Net parameters
        self.args = args
        self.feature_dim = 2 * int(args.feature_dim) + 1 # 输入特征维度 default: 6 (x, y, class, v, a, laneID)
        self.input_dim = int(args.input_dim)   # 输入到Dit的维度 default: 128
        self.hidden_dim = int(args.hidden_dim)
        self.output_dim = int(args.output_dim)
        self.heads = int(args.heads)
        self.dropout = args.dropout
        self.depth = int(args.depth)
        self.mlp_ratio = args.mlp_ratio
        self.num_train_timesteps = args.num_train_timesteps
        self.time_embedding_size = args.time_embedding_size
        self.training = True
        self.T = int(args.T)
        self.N = int(args.N)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        self.dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.N, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=self.dit_block,
            final_layer=self.final_layer,
            time_embedder=self.timestep_embedder,
            depth=self.depth,
            model_type="x_start"
        )

        self.loss = DiffusionLoss(reduction="mean")
        self.norm_config_path = str(Path(__file__).resolve().parent.parent / 'dataset/ngsim_stats.npz')
        self.norm_config = np.load(self.norm_config_path)
        for key, value in self.norm_config.items():
            self.register_buffer(key, torch.from_numpy(value).float())

    def forward_train(self, hist, hist_masked, device):
        hist_norm = self.norm(hist)

        B, T, N, dim = hist.shape
        x_start = hist_norm
        hist_masked = hist_masked.view(B, N * T, -1)

        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)
        x_noisy = self.denorm(x_noisy).view(B, N * T, -1)

        # (Input = Noisy_GT + Hist_Masked)
        model_input = torch.cat([x_noisy, hist_masked], dim=-1)

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, t=timesteps)

        loss = self.loss(pred_x0, hist, None)

        return loss, pred_x0

    def forward_eval(self, hist, hist_masked, device):
        B, T, N, dim = hist.shape

        hist_mask = hist_masked[..., -1:]  # [B, T, N, 1]
        hist_values = hist_masked[..., :-1]  # [B, T, N, dim]
        hist_norm = self.norm(hist_values)
        hist_masked_norm = hist_norm * hist_mask # [B, T, N, dim]

        timesteps = torch.full((B,), self.num_train_timesteps - 1, device=device, dtype=torch.long)
        noise = torch.randn_like(hist_masked_norm)
        x_t = self.diffusion_scheduler.add_noise(hist_masked_norm, noise, timesteps)
        x_t = x_t.view(B, T, N, -1)
        hist_masked = hist_masked.view(B, N * T, -1)

        self.diffusion_scheduler.set_timesteps(4)

        for t in self.diffusion_scheduler.timesteps:
            x_input = self.denorm(x_t).view(B, N * T, -1)
            model_input = torch.cat([x_input, hist_masked], dim=-1)

            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
            pred_x0 = self.dit(x=input_embedded, t=timesteps) # [B, T, N, dim]

            pred_x0_norm = self.norm(pred_x0)
            x_t = pred_x0_norm

        final_pred = self.denorm(x_t)
        loss = self.loss(final_pred, hist, None)

        return loss, final_pred

    def forward_test(self, hist_masked, device):
        mask_indicator = hist_masked[..., -1:]  # [B, T, 1, 1]
        hist_val_masked = self.norm(hist_masked[..., :-1])  # 归一化值
        cond_input = hist_val_masked * mask_indicator  # 确保未观测区域绝对为 0

        B, T, N, dim_plus_one = hist_masked.shape
        x_t = torch.randn((B, N * T, self.output_dim), device=device)
        dim_gap = dim_plus_one - self.output_dim - 1
        zeros_padding = torch.zeros((B, N * T, dim_gap), device=device)
        x_t_expanded = torch.cat([x_t, zeros_padding], dim=-1)

        cond_flat = cond_input.view(B, N * T, -1)
        mask_flat = mask_indicator.view(B, N * T, 1)
        model_input = torch.cat([x_t_expanded, cond_flat, mask_flat], dim=-1)
        timesteps = torch.full((B,), self.num_train_timesteps - 1, device=device, dtype=torch.long)

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, t=timesteps)

        # loss = self.loss(pred_x0, x_start.view(B, T, N, -1)[..., :2], None)
        pred_denorm = self.denorm(pred_x0.view(B, T, N, -1))

        return self.denorm(pred_denorm)  # self.denorm(pred_x0.view(B, T, N, -1))

    def forward_test1(self, hist, hist_masked, device):
        B, T, N, dim = hist.shape
        hist_masked = hist_masked.view(B, N * T, -1)

        x_noisy = torch.randn((B, T, N, dim), device=device)
        x_noisy = self.denorm(x_noisy).view(B, N * T, -1)
        timesteps = torch.full((B,), self.num_train_timesteps, device=device, dtype=torch.long)

        # (Input = Noisy_GT + Hist_Masked)
        model_input = torch.cat([x_noisy, hist_masked], dim=-1)

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, t=timesteps)

        loss = self.loss(pred_x0, hist, None)

        return loss, pred_x0

    def forward_test2(self, hist, hist_masked, device):
        device = device
        B, T, N, dim_plus_one = hist_masked.shape

        hist_masked = hist_masked.view(B, N * T, -1)

        x_t = torch.randn((B, N * T, self.output_dim), device=device)
        x_input = self.denorm_eval(x_t)

        self.diffusion_scheduler.set_timesteps(5)

        dim_gap = dim_plus_one - 1 - self.output_dim
        zeros_padding = torch.randn((B, N * T, dim_gap), device=device)

        for t in self.diffusion_scheduler.timesteps:
            x_t_expanded = torch.cat([x_input, zeros_padding], dim=-1)
            model_input = torch.cat([x_t_expanded, hist_masked], dim=-1)
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
            pred_x0 = self.dit(x=input_embedded, t=t_batch).view(B, N * T, self.output_dim)
            pred_x0_norm = self.norm_eval(pred_x0)
            output = self.diffusion_scheduler.step(pred_x0_norm, t, x_t)
            x_t_minus_1 = output.prev_sample
            x_t = x_t_minus_1

            # if t > 0:
            #     cond_pos = cond_val[..., :self.output_dim]
            #     t_prev = t - self.diffusion_scheduler.config.num_train_timesteps // self.diffusion_scheduler.num_inference_steps
            #     noise = torch.randn_like(cond_pos)
            #     known_part_noisy = self.diffusion_scheduler.add_noise(
            #         cond_pos, noise, torch.tensor([t], device=device)  # 理论上应为 t_prev
            #     )
            #
            #     x_t = x_t_minus_1 * (1 - mask_indicator) + known_part_noisy * mask_indicator
            # else:
            #     x_t = x_t_minus_1

        final_pred = x_t
        # cond_pos = cond_val[..., :self.output_dim]

        # 强制将已知部分替换为完美的 GT (无噪声)
        # final_pred = final_pred * (1 - mask_indicator) + cond_pos * mask_indicator
        final_pred = final_pred.view(B, T, N, self.output_dim)
        final_pred = self.denorm_eval(final_pred)
        loss = self.loss(final_pred, hist[..., :2], None)

        return loss, final_pred

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm[..., 4] = (x[..., 4] - self.lane_mean) / self.lane_std  # laneID
        x_norm[..., 5] = (x[..., 5] - self.class_mean) / self.class_std  # class
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def norm_eval(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean  # v, a
        x_denorm[..., 4] = x[..., 4] * self.lane_std + self.lane_mean  # laneID
        x_denorm[..., 5] = x[..., 5] * self.class_std + self.class_mean  # class
        return x_denorm


    def denorm_eval(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        return x_denorm
