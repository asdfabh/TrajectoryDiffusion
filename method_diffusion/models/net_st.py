from method_diffusion.models import dit
from method_diffusion.models.dit_spatiotemporal import SpatioTemporalDiTBlock
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
        self.args = args
        self.feature_dim = int(args.feature_dim)
        self.input_dim = int(args.input_dim)
        self.hidden_dim = int(args.hidden_dim)
        self.output_dim = int(args.output_dim)
        self.heads = int(args.heads)
        self.dropout = args.dropout
        self.depth = int(args.depth)
        self.mlp_ratio = args.mlp_ratio
        self.num_train_timesteps = args.num_train_timesteps
        self.time_embedding_size = args.time_embedding_size
        self.training = True

        # 固定维度
        self.N = 40  # 车辆数量 (1 ego + 39 nbrs)
        self.T = 16  # 时间步数

        # 输入嵌入和位置编码
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)

        # 时间步嵌入
        self.timestep_embedder = dit.TimestepEmbedder(
            self.input_dim, self.time_embedding_size
        )

        # Diffusion 调度器
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # 时空分离 DiT Blocks
        self.dit_blocks = nn.ModuleList([
            SpatioTemporalDiTBlock(
                dim=self.input_dim,
                heads=self.heads,
                dropout=self.dropout,
                mlp_ratio=self.mlp_ratio,
                N=self.N,
                T=self.T
            )
            for _ in range(self.depth)
        ])

        # 最终输出层
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.output_dim)

        # 损失函数
        self.loss = DiffusionLoss(reduction="mean")

        # 归一化配置
        self.norm_config_path = str(
            Path(__file__).resolve().parent.parent / 'dataset/ngsim_stats2.npz'
        )
        self.norm_config = np.load(self.norm_config_path)
        for key, value in self.norm_config.items():
            self.register_buffer(key, torch.from_numpy(value).float())

    def padding(self, x, mask, fill_value=0.0):
        x = x.numpy() if isinstance(x, torch.Tensor) else x
        mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask
        mask = mask.astype(bool)
        masked_traj = x.copy()
        masked_traj[~mask] = fill_value
        return masked_traj

    """
    首先进行归一化处理，将从ngsim中拿到的数据归一化，包括自车和周车的历史轨迹，速度加速度信息，历史轨迹以及车辆类型
    然后将稠密堆叠的周车数据转换为稀疏表达形式，得到Ego和nbrs的矩阵表达
    构建hist和nbrs的掩码矩阵，表示是否有信息，以及进行非完整轨迹掩码
    将hist和nbrs进行输入嵌入和位置编码相加，得到Dit的输入
    hist: [B, T, dim]  nbrs:[N_total, T, dim]
    """
    def forward(self, hist, hist_mask, nbrs, nbrs_mask, nbrs_num, timestep):
        if self.training:
            return self.forward_train(hist, hist_mask, nbrs, nbrs_mask, nbrs_num)
        else:
            return self.forward_test(hist, hist_mask, nbrs, nbrs_mask, nbrs_num)

    def preprossess_input(self, hist, hist_mask, nbrs, nbrs_mask, nbrs_num):
        # 归一化
        hist_norm = self.norm(hist * hist_mask.unsqueeze(-1))
        nbrs_norm = self.norm(nbrs * nbrs_mask.unsqueeze(-1))

        B, T, dim = hist_norm.shape
        device = hist.device

        # 向量化掩码: [B, 39]
        nbrs_exists = torch.arange(39, device=device).unsqueeze(0) < nbrs_num.unsqueeze(1)
        nbrs_mask_grid = nbrs_exists.unsqueeze(-1).expand(-1, -1, T)  # [B, 39, T]

        # 创建并填充网格
        nbrs_sparse = torch.zeros(B, 39, T, dim, device=device)
        nbrs_sparse_mask = torch.zeros(B, 39, T, device=device)

        nbrs_sparse.view(-1, dim).masked_scatter_(
            nbrs_mask_grid.reshape(-1, 1).expand(-1, dim),
            nbrs_norm.view(-1, dim)
        )
        nbrs_sparse_mask.view(-1).masked_scatter_(
            nbrs_mask_grid.reshape(-1),
            nbrs_mask.view(-1)
        )

        # 拼接
        return (
            torch.cat([hist_norm.unsqueeze(1), nbrs_sparse], dim=1),
            torch.cat([hist_mask.unsqueeze(1), nbrs_sparse_mask], dim=1)
        )

    def forward_train(self, hist, hist_mask, nbrs, nbrs_mask, nbrs_num):
        inputs, masks = self.preprossess_input(
            hist, hist_mask, nbrs, nbrs_mask, nbrs_num
        )
        B, N, T, dim = inputs.shape
        device = hist.device

        # 验证维度
        assert N == self.N and T == self.T, \
            f"输入维度不匹配: N={N} (期望{self.N}), T={T} (期望{self.T})"

        inputs = inputs.view(B, N * T, dim)
        masks = masks.view(B, N * T).unsqueeze(-1)

        # 加噪
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=inputs.device
        )
        noise = torch.randn(inputs.shape, device=device)
        noisy_inputs = self.diffusion_scheduler.add_noise(
            original_samples=inputs,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_inputs = torch.clamp(noisy_inputs, -5, 5)

        # 嵌入 + 位置编码
        input_embedded = self.input_embedding(noisy_inputs) + \
                         self.pos_embedding(noisy_inputs)
        attn_mask = (masks.squeeze(-1) == 0)  # [B, N*T]

        # 时间步嵌入
        y = self.timestep_embedder(timesteps)  # [B, dim]

        # 前向传播 (时空分离 DiT)
        x = input_embedded
        for block in self.dit_blocks:
            x = block(x, y, attn_mask)

        # 最终层
        pred = self.final_layer(x, y)

        # 后处理
        pred = pred.view(B, N, T, 2)
        pred_ego = pred[:, 0, :, :]
        pred_nbrs_dense = pred[:, 1:, :, :]

        # 转换为稀疏表示
        pred_nbrs_sparse = []
        for b in range(B):
            count = int(nbrs_num[b])
            if count > 0:
                pred_nbrs_batch = pred_nbrs_dense[b, 0:count, :, :]
                pred_nbrs_sparse.append(pred_nbrs_batch)

        if pred_nbrs_sparse:
            pred_nbrs = torch.cat(pred_nbrs_sparse, dim=0)
        else:
            pred_nbrs = torch.empty(0, T, 2, device=device)

        # 计算损失
        loss_ego = self.loss(pred_ego, hist[..., 0:2], hist_mask.unsqueeze(-1))
        loss_nbrs = self.loss(pred_nbrs, nbrs[..., 0:2], nbrs_mask.unsqueeze(-1))
        loss = loss_ego + loss_nbrs

        return loss, pred_ego, pred_nbrs

    def forward_test(self, hist, hist_mask, nbrs, nbrs_mask, nbrs_num, num_inference_steps=50):
        inputs, masks = self.preprossess_input(hist, hist_mask, nbrs, nbrs_mask, nbrs_num)
        B, N, T, dim = inputs.shape
        device = inputs.device

        inputs = inputs.view(B, N * T, dim)
        masks = masks.view(B, N * T).unsqueeze(-1)

        # 设置推理步数
        self.diffusion_scheduler.set_timesteps(num_inference_steps, device=device)

        # 从纯噪声开始
        noisy_inputs = torch.randn_like(inputs)

        # 反向去噪循环
        for t in self.diffusion_scheduler.timesteps:
            timestep = torch.full((B,), t, device=device, dtype=torch.long)

            # 嵌入并预测
            input_embedded = self.input_embedding(noisy_inputs) + self.pos_embedding(noisy_inputs)
            attn_mask = masks.squeeze(-1) == 0
            pred = self.dit(x=input_embedded, t=timestep, neighbor_current_mask=attn_mask)

            # 拼接预测的 x,y 和原始的其他特征
            pred_full = noisy_inputs.clone()
            pred_full[..., 0:2] = pred

            # 调度器单步去噪
            noisy_inputs = self.diffusion_scheduler.step(
                model_output=pred_full,
                timestep=t,
                sample=noisy_inputs
            ).prev_sample

            noisy_inputs = torch.clamp(noisy_inputs, -5, 5)

        final_pred = noisy_inputs[..., 0:2]  # [B, N*T, 2]

        pred = final_pred.view(B, N, T, 2)

        pred_ego = pred[:, 0, :, :]  # [B, T, 2]
        pred_nbrs_dense = pred[:, 1:, :, :]  # [B, 39, T, 2]

        pred_nbrs_sparse = []
        for b in range(B):
            count = int(nbrs_num[b])
            if count > 0:
                pred_nbrs_batch = pred_nbrs_dense[b, 0:count, :, :]  # [count, T, 2]
                pred_nbrs_sparse.append(pred_nbrs_batch)

        if pred_nbrs_sparse:
            pred_nbrs = torch.cat(pred_nbrs_sparse, dim=0)  # [N_total, T, 2]
        else:
            pred_nbrs = torch.empty(0, T, 2, device=device)

        loss_ego = self.loss(pred_ego, hist[..., 0:2], hist_mask.unsqueeze(-1))
        loss_nbrs = self.loss(pred_nbrs, nbrs[..., 0:2], nbrs_mask.unsqueeze(-1)) if pred_nbrs.numel() > 0 else 0.0
        loss = loss_ego + loss_nbrs

        pred = torch.cat([pred_ego, pred_nbrs], dim=0)  # [B + N_total, T, 2]
        print(f"shape of pred in test: {pred.shape}")
        return loss, pred

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        x_norm[..., 3:5] = (x[..., 3:5] - self.va_mean) / self.va_std  # v, a
        # x_norm[..., 5] = (x[..., 5] - self.lane_mean) / self.lane_std  # laneID
        # x_norm[..., 6] = (x[..., 6] - self.class_mean) / self.class_std  # class
        x_norm = torch.clamp(x_norm, -5.0, 5.0)

        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        x_denorm[..., 3:5] = x[..., 3:5] * self.va_std + self.va_mean  # v, a
        # x_denorm[..., 5] = x[..., 5] * self.lane_std + self.lane_mean  # laneID
        # x_denorm[..., 6] = x[..., 6] * self.class_std + self.class_mean  # class

        return x_denorm



