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
        self.feature_dim = int(args.feature_dim) + 1 # 输入特征维度 default: 6 (x, y, class, v, a, laneID)
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
        self.final_layer = dit.FinalLayer(self.hidden_dim, 40, 16, 2)
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

    def forward_train(self, hist, nbrs, nbrs_num, src, nbrs_src):
        hist_norm = self.norm(hist)
        nbrs_norm = self.norm(nbrs)
        src_norm = self.norm(src)
        nbrs_src_norm = self.norm(nbrs_src)
        inputs = torch.cat([hist_norm, nbrs_norm], dim=2)  # [B, T, N_total, dim]
        B, N, T, dim = inputs.shape  # N includes ego and nbrs, N = 40
        device = hist.device
        inputs = inputs.view(B, N * T, dim)

        timestpes = torch.randint(0, self.num_train_timesteps, (B,), device=inputs.device)
        noise = torch.randn(inputs.shape, device=device)
        noisy_inputs = self.diffusion_scheduler.add_noise(
            original_samples=inputs,
            noise=noise,
            timesteps=timestpes,
        ).float()
        noisy_inputs = torch.clamp(noisy_inputs, -5, 5)

        input_embedded = self.input_embedding(noisy_inputs) + self.pos_embedding(noisy_inputs) # [B, N*T, input_dim]
        pred = self.dit(x=input_embedded, t=timestpes, neighbor_current_mask=None) # [B, T, N, 2]

        pred_ego = pred[:, :, 0:1, :]  # [B, T， 1, 2]
        pred_nbrs_dense = pred[:, :, 1:, :]  # [B, T, 39, 2]
        # print(f"shape of pred in train: {pred.shape}")

        loss_ego = self.loss(pred_ego, src_norm[..., 0:2], None)
        loss_nbrs = self.loss(pred_nbrs_dense, nbrs_src_norm[..., 0:2], None)
        loss = loss_ego + loss_nbrs

        pred_nbrs_sparse = []
        for b in range(B):
            count = int(nbrs_num[b])
            if count > 0:
                pred_nbrs_batch = pred_nbrs_dense[b, :, 0:count, :]
                pred_nbrs_batch = pred_nbrs_batch.permute(1, 0, 2)  # [count, T, 2]
                pred_nbrs_sparse.append(pred_nbrs_batch)

        if pred_nbrs_sparse:
            pred_nbrs = torch.cat(pred_nbrs_sparse, dim=0)  # [N_total, T, 2]
        else:
            pred_nbrs = torch.empty(0, T, 2, device=device)

        pred_ego = self.denorm(pred_ego)
        pred_nbrs = self.denorm(pred_nbrs)

        return loss, pred_ego, pred_nbrs

    def forward_eval(self, hist, nbrs, nbrs_num, num_inference_steps=10):
        """
        推理模式: 使用 DDIM 采样基于掩码输入生成完整历史轨迹

        Args:
            hist: [B, T, 1, dim] 掩码后的自车历史 (最后一维是观测标记)
            nbrs: [B, T, 39, dim] 掩码后的邻车历史 (最后一维是观测标记)
            nbrs_num: [B] 每个样本的实际邻车数量
            num_inference_steps: DDIM 推理步数
        """
        device = hist.device
        B, T, _, _ = hist.shape
        N = 40  # 1 ego + 39 nbrs

        # 1. 提取观测掩码 (最后一维)
        obs_mask_hist = hist[..., -1:]  # [B, T, 1, 1]
        obs_mask_nbrs = nbrs[..., -1:]  # [B, T, 39, 1]
        obs_mask = torch.cat([obs_mask_hist, obs_mask_nbrs], dim=2)  # [B, T, 40, 1]

        # 2. 归一化输入 (去掉观测标记)
        hist_features = hist[..., :-1]  # [B, T, 1, dim-1]
        nbrs_features = nbrs[..., :-1]  # [B, T, 39, dim-1]

        hist_norm = self.norm(hist_features)
        nbrs_norm = self.norm(nbrs_features)

        inputs_norm = torch.cat([hist_norm, nbrs_norm], dim=2)  # [B, T, 40, dim-1]

        # 3. 展平为 [B, N*T, dim-1] 并拼接观测掩码
        obs_mask_flat = obs_mask.permute(0, 2, 1, 3).reshape(B, N * T, 1)
        inputs_flat = inputs_norm.permute(0, 2, 1, 3).reshape(B, N * T, self.feature_dim - 1)
        inputs_with_mask = torch.cat([inputs_flat, obs_mask_flat], dim=-1)  # [B, N*T, dim]

        # 4. 初始化: 从截断噪声开始
        noisy_inputs = torch.randn(B, N * T, self.feature_dim, device=device)

        # 对已观测的点使用真实值
        noisy_inputs = noisy_inputs * (1 - obs_mask_flat) + inputs_with_mask * obs_mask_flat

        # 5. 设置 DDIM 调度器
        self.diffusion_scheduler.set_timesteps(num_inference_steps, device)

        # 使用截断扩散 (参考 Transfuser)
        step_ratio = self.num_train_timesteps / num_inference_steps
        roll_timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)

        # 6. DDIM 逐步去噪
        for timestep in roll_timesteps:
            # 将时间步扩展到 batch
            t = timestep if torch.is_tensor(timestep) else torch.tensor([timestep], device=device)
            t = t.expand(B)

            # 嵌入当前噪声样本
            input_embedded = self.input_embedding(noisy_inputs) + self.pos_embedding(noisy_inputs)

            # 模型预测去噪后的轨迹 x_0 [B, T, N, output_dim]
            pred_x0 = self.dit(x=input_embedded, t=t, neighbor_current_mask=None)

            # 调整维度: [B, T, N, 2] -> [B, N*T, 2]
            pred_x0_flat = pred_x0.permute(0, 2, 1, 3).reshape(B, N * T, self.output_dim)

            # 拼接其他特征 (保持输入值不变)
            pred_x0_full = torch.cat([
                pred_x0_flat,  # 预测的 (x, y)
                inputs_flat[..., 2:]  # 保留 (class, v, a, lane)
            ], dim=-1)

            # 拼接观测掩码
            pred_x0_with_mask = torch.cat([pred_x0_full, obs_mask_flat], dim=-1)

            # DDIM step: 更新噪声样本
            noisy_inputs = self.diffusion_scheduler.step(
                model_output=pred_x0_with_mask,
                timestep=timestep,
                sample=noisy_inputs
            ).prev_sample

            # 强制已观测点保持不变
            noisy_inputs = noisy_inputs * (1 - obs_mask_flat) + inputs_with_mask * obs_mask_flat

        # 7. 获取最终预测 (去掉观测掩码)
        final_pred_flat = noisy_inputs[..., :-1]  # [B, N*T, dim-1]

        # 8. 调整维度并拆分
        final_pred = final_pred_flat.view(B, N, T, self.feature_dim - 1).permute(0, 2, 1, 3)  # [B, T, N, dim-1]
        pred_ego = final_pred[:, :, 0:1, :]  # [B, T, 1, dim-1]
        pred_nbrs_dense = final_pred[:, :, 1:, :]  # [B, T, 39, dim-1]

        # 9. 转换为稀疏邻车格式
        pred_nbrs_sparse = []
        for b in range(B):
            count = int(nbrs_num[b])
            if count > 0:
                pred_nbrs_batch = pred_nbrs_dense[b, :, :count, :].permute(1, 0, 2)  # [count, T, dim-1]
                pred_nbrs_sparse.append(pred_nbrs_batch)

        pred_nbrs = torch.cat(pred_nbrs_sparse, dim=0) if pred_nbrs_sparse else torch.empty(0, T, self.feature_dim - 1,
                                                                                            device=device)
        # 10. 反归一化
        pred_ego = self.denorm(pred_ego)
        pred_nbrs = self.denorm(pred_nbrs)

        return pred_ego, pred_nbrs, pred_nbrs_dense

    def forward_test(self, hist, nbrs, nbrs_num, src, nbrs_src):
        hist_norm = self.norm(hist)
        nbrs_norm = self.norm(nbrs)
        src_norm = self.norm(src)
        nbrs_src_norm = self.norm(nbrs_src)
        inputs = torch.cat([hist_norm, nbrs_norm], dim=2)  # [B, T, N_total, dim]
        B, N, T, dim = inputs.shape  # N includes ego and nbrs, N = 40
        device = hist.device
        inputs = inputs.view(B, N * T, dim)

        timestpes = torch.randint(0, self.num_train_timesteps, (B,), device=inputs.device)
        noise = torch.randn(inputs.shape, device=device)
        noisy_inputs = self.diffusion_scheduler.add_noise(
            original_samples=inputs,
            noise=noise,
            timesteps=timestpes,
        ).float()
        noisy_inputs = torch.clamp(noisy_inputs, -5, 5)

        input_embedded = self.input_embedding(noisy_inputs) + self.pos_embedding(noisy_inputs) # [B, N*T, input_dim]
        pred = self.dit(x=input_embedded, t=timestpes, neighbor_current_mask=None) # [B, T, N, 2]

        pred_ego = pred[:, :, 0:1, :]  # [B, T， 1, 2]
        pred_nbrs_dense = pred[:, :, 1:, :]  # [B, T, 39, 2]
        # print(f"shape of pred in train: {pred.shape}")

        loss_ego = self.loss(pred_ego, src_norm[..., 0:2], None)
        loss_nbrs = self.loss(pred_nbrs_dense, nbrs_src_norm[..., 0:2], None)
        loss = loss_ego + loss_nbrs

        pred_nbrs_sparse = []
        for b in range(B):
            count = int(nbrs_num[b])
            if count > 0:
                pred_nbrs_batch = pred_nbrs_dense[b, :, 0:count, :]
                pred_nbrs_batch = pred_nbrs_batch.permute(1, 0, 2)  # [count, T, 2]
                pred_nbrs_sparse.append(pred_nbrs_batch)

        if pred_nbrs_sparse:
            pred_nbrs = torch.cat(pred_nbrs_sparse, dim=0)  # [N_total, T, 2]
        else:
            pred_nbrs = torch.empty(0, T, 2, device=device)

        pred_ego_denorm = self.denorm(pred_ego)
        pred_nbrs_denorm = self.denorm(pred_nbrs)

        return loss, pred_ego, pred_nbrs, pred_ego_denorm, pred_nbrs_denorm, hist_norm, nbrs_norm, self.norm(src)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm[..., 3:5] = (x[..., 3:5] - self.va_mean) / self.va_std  # v, a
        x_norm[..., 4] = (x[..., 4] - self.lane_mean) / self.lane_std  # laneID
        x_norm[..., 5] = (x[..., 5] - self.class_mean) / self.class_std  # class
        x_norm = torch.clamp(x_norm, -5.0, 5.0)

        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        return x_denorm



