import torch
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import maybe_visualize_future_prediction


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()

        # 模型结构参数：控制 DiT 主干维度、层数和 future 序列长度。
        self.hidden_dim = int(args.hidden_dim_fut)
        self.input_dim = int(args.input_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.depth = int(args.depth_fut)
        self.dropout = float(args.dropout_fut)
        self.mlp_ratio = int(args.mlp_ratio_fut)
        self.time_embedding_size = int(args.time_embedding_size_fut)
        self.T = int(args.T_f)
        self.gaussian_param_dim = int(args.gaussian_param_dim_fut)

        # 扩散与推理参数：控制训练时间步、推理步数和 DDIM 采样行为。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.x0_clip = float(args.x0_clip) if float(args.x0_clip) > 0 else None

        # 训练策略与损失参数：控制自条件训练和损失项权重。
        self.self_condition_prob = min(max(float(args.self_condition_prob), 0.0), 1.0)
        self.pos_loss_weight = max(0.0, float(args.fut_pos_loss_weight))

        # 可视化参数：仅决定训练和评估阶段是否绘图。
        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048

        # 输入编码模块：分别处理 future 噪声序列和 history context。
        self.input_embedding = nn.Linear(self.input_dim * 2, self.hidden_dim)
        self.context_embedding = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)

        # DiT 主干与扩散调度器：负责时间嵌入、去噪建模和 DDIM 调度。
        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.hidden_dim, self.heads, self.dropout, self.mlp_ratio)
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.gaussian_param_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=self.depth, model_type="x_start")

        # 双空间归一化参数
        # 物理坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        self.register_buffer("pos_mean", torch.tensor([0.0330, -15.9150]).float(), persistent=False)
        self.register_buffer("pos_std", torch.tensor([8.8866, 68.8105]).float(), persistent=False)
        self.register_buffer("va_mean", torch.tensor([21.1503, 0.0060]).float(), persistent=False)
        self.register_buffer("va_std", torch.tensor([13.5983, 4.5057]).float(), persistent=False)
        # 帧间相对位移归一化参数 (Velocity)
        self.register_buffer("vel_mean", torch.tensor([-0.004182, 5.041937], dtype=torch.float32).float(), persistent=False)
        self.register_buffer("vel_std", torch.tensor([0.150222, 2.951254], dtype=torch.float32).float(), persistent=False)

    # 将输出通道掩码转换为按时间步的有效帧掩码。
    @staticmethod
    def toValidMask(op_mask, device):
        return (op_mask[..., 0] > 0.5).float().to(device)

    # 基于当前噪声状态与历史条件预测二维高斯参数。
    def predictX0(self, x_t, timesteps, context_aligned, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        context_encoded = self.context_embedding(context_aligned)
        gaussian_raw = self.dit(
            x=input_embedded,
            t_cond=t_emb,
            cross=context_encoded,
        )
        mu = gaussian_raw[..., 0:2]
        prec = torch.exp(gaussian_raw[..., 2:4])
        rho = torch.tanh(gaussian_raw[..., 4:5])
        return torch.cat([mu, prec, rho], dim=-1)

    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, _ = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        return context

    @staticmethod
    def gaussianNllWithPrecision(pred_gaussian, target_xy, add_const_term=True):
        mu_x = pred_gaussian[..., 0]
        mu_y = pred_gaussian[..., 1]
        prec_x = torch.clamp(pred_gaussian[..., 2], min=1e-6)
        prec_y = torch.clamp(pred_gaussian[..., 3], min=1e-6)
        rho = torch.clamp(pred_gaussian[..., 4], min=-0.99, max=0.99)
        x = target_xy[..., 0]
        y = target_xy[..., 1]

        ohr = torch.rsqrt(1.0 - rho.pow(2) + 1e-6)
        loss_nll = 0.5 * ohr.pow(2) * (
            prec_x.pow(2) * (x - mu_x).pow(2)
            + prec_y.pow(2) * (y - mu_y).pow(2)
            - 2.0 * rho * prec_x * prec_y * (x - mu_x) * (y - mu_y)
        ) - torch.log(prec_x * prec_y * ohr + 1e-6)
        if add_const_term:
            loss_nll = loss_nll - 1.8379
        return loss_nll

    def integrateGaussianToAbsolute(self, pred_gaussian, anchor_phys):
        std_vel = self.vel_std.view(1, 1, 2).to(pred_gaussian.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_gaussian.device)

        mu_vel_phys = pred_gaussian[..., 0:2] * std_vel + mean_vel
        prec_vel = torch.clamp(pred_gaussian[..., 2:4], min=1e-6)
        rho_vel = torch.clamp(pred_gaussian[..., 4:5], min=-0.99, max=0.99)

        sigma_vel = std_vel / prec_vel
        var_vel = sigma_vel.pow(2)
        cov_vel = rho_vel * sigma_vel[..., 0:1] * sigma_vel[..., 1:2]

        # 在当前实现里，各时间步增量只显式建模各自的二维边缘协方差；
        # 积分到绝对位置时默认忽略跨时间协方差，只做前缀累加。
        mu_pos_phys = torch.cumsum(mu_vel_phys, dim=1) + anchor_phys[..., :2]
        var_pos_x = torch.cumsum(var_vel[..., 0:1], dim=1)
        var_pos_y = torch.cumsum(var_vel[..., 1:2], dim=1)
        cov_pos = torch.cumsum(cov_vel, dim=1)

        sigma_pos_x = torch.sqrt(torch.clamp(var_pos_x, min=1e-6))
        sigma_pos_y = torch.sqrt(torch.clamp(var_pos_y, min=1e-6))
        rho_pos = cov_pos / torch.clamp(sigma_pos_x * sigma_pos_y, min=1e-6)
        rho_pos = torch.clamp(rho_pos, min=-0.99, max=0.99)
        prec_pos_x = 1.0 / sigma_pos_x
        prec_pos_y = 1.0 / sigma_pos_y

        pred_abs_gaussian = torch.cat(
            [mu_pos_phys, prec_pos_x, prec_pos_y, rho_pos],
            dim=-1,
        )
        return pred_abs_gaussian

    # 计算训练使用的绝对坐标高斯 NLL；相对空间 NLL 仅作为诊断项保留。
    def computeLoss(self, pred_gaussian, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=False):
        rel_nll = self.gaussianNllWithPrecision(
            pred_gaussian,
            target_vel_norm,
            add_const_term=False,
        )
        pred_abs_gaussian = self.integrateGaussianToAbsolute(pred_gaussian, anchor_phys)
        abs_nll = self.gaussianNllWithPrecision(
            pred_abs_gaussian,
            future_phys[..., :2],
            add_const_term=True,
        )

        valid = valid_mask.float()
        loss_rel_mean = (rel_nll * valid).sum() / (valid.sum() + 1e-6)
        loss_abs_mean = (abs_nll * valid).sum() / (valid.sum() + 1e-6)
        total_mean = loss_abs_mean

        if not return_parts:
            return total_mean

        parts = {
            "loss_total": total_mean.detach(),
            "loss_diffusion": loss_abs_mean.detach(),
            "loss_vel": loss_rel_mean.detach(),
            "loss_pos": loss_abs_mean.detach(),
            "loss_rel_nll": loss_rel_mean.detach(),
            "loss_abs_nll": loss_abs_mean.detach(),
        }
        return total_mean, parts

    # 计算 batch 级别的 ADE 与 FDE 指标。
    @staticmethod
    def computeAdeFde(pred, target, valid_mask):
        diff = pred[..., :2] - target[..., :2]
        dist = torch.norm(diff, dim=-1)
        ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        valid_counts = valid_mask.sum(dim=1).long()
        has_valid = valid_counts > 0
        last_idx = torch.clamp(valid_counts - 1, min=0)
        final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
        return ade, fde

    # 执行推理阶段的 DDIM 采样并输出最终预测结果（相对坐标系）。
    def sampleFromXt(self, x_t, context_aligned, infer_scheduler):
        bsz, t_len, _ = x_t.shape
        pred_vel_cond = torch.zeros((bsz, t_len, self.input_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_gaussian = self.predictX0(x_t, timesteps, context_aligned, pred_vel_cond)
            pred_vel_mean = pred_gaussian[..., 0:2]
            if self.x0_clip is not None:
                pred_vel_mean = torch.clamp(pred_vel_mean, -self.x0_clip, self.x0_clip)

            pred_vel_cond = pred_vel_mean.detach()
            try:
                x_t = infer_scheduler.step(pred_vel_mean, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_mean, t, x_t).prev_sample
        return pred_vel_cond

    # 统一准备评估阶段所需的条件编码、掩码和调度器参数。
    def prepareEvalInputs(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, device)
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        context = self.encodeContext(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
        )
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        return bsz, t_len, valid_mask, anchor_phys, future_phys, context, infer_scheduler, std_vel, mean_vel

    # 执行单个 batch 的 fut 训练前向与损失计算。
    def forwardTrain(
        self,
        hist,
        hist_nbrs,
        mask,
        temporal_mask,
        future,
        op_mask,
        device,
        return_components=False,
    ):
        bsz, t_len, _ = future.shape  # [B, T, D]
        valid_mask = self.toValidMask(op_mask, device)  # [B, T]

        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        # 坐标空间转换：计算 GT 的真实物理帧间位移 (Velocity)
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys

        # 归一化：将物理速度变为正态分布
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -5.0, 5.0)

        # 加噪过程
        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        context = self.encodeContext(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
        )

        # 自条件：以一定概率将前一步的预测结果作为下一步的输入条件，增强模型稳定性和一致性
        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_gaussian = self.predictX0(x_t, timesteps, context, pred_vel_cond)
                pred_vel_cond = prev_pred_gaussian[..., 0:2].detach() * use_sc

        # 网络输出：预测的二维高斯分布参数
        pred_gaussian_t = self.predictX0(x_t, timesteps, context, pred_vel_cond)
        pred_vel_mean_t = pred_gaussian_t[..., 0:2]

        diffusion_loss, loss_parts = self.computeLoss(
            pred_gaussian_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            return_parts=True,
        )
        loss = diffusion_loss
        loss_parts.update(
            {
                "loss_total": loss.detach(),
                "loss_diffusion": diffusion_loss.detach(),
            }
        )

        if self.fut_enable_train_vis:
            pred_vel_phys_t = pred_vel_mean_t * std_vel + mean_vel
            pred_pos_phys = torch.cumsum(pred_vel_phys_t, dim=1) + anchor_phys[..., :2]

            # 兼容多维度输出拼接
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            maybe_visualize_future_prediction(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=pred_phys_abs,
                valid_mask=valid_mask,
                stage="train",
                enable_train_vis=self.fut_enable_train_vis,
                enable_eval_vis=self.fut_enable_eval_vis,
                meter_per_foot=self.meter_per_foot,
            )

        if return_components:
            return loss, loss_parts
        return loss

    @torch.no_grad()
    # 执行单模态推理评估并返回轨迹与指标。
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        (
            bsz,
            t_len,
            valid_mask,
            anchor_phys,
            future_phys,
            context,
            infer_scheduler,
            std_vel,
            mean_vel,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        x_t = torch.randn((bsz, t_len, self.input_dim), device=device)
        pred_vel_norm = self.sampleFromXt(x_t, context, infer_scheduler)

        # 积分还原：解归一化 -> 累加 -> 拼接绝对锚点
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        pred_phys_abs = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        maybe_visualize_future_prediction(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=pred_phys_abs,
            valid_mask=valid_mask,
            stage="eval",
            enable_train_vis=self.fut_enable_train_vis,
            enable_eval_vis=self.fut_enable_eval_vis,
            meter_per_foot=self.meter_per_foot,
        )
        return pred_phys_abs, ade, fde

    @torch.no_grad()
    # 执行并行多模态推理评估并返回 minADE 对应结果。
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5):
        (
            bsz,
            t_len,
            valid_mask,
            anchor_phys,
            future_phys,
            context,
            infer_scheduler,
            std_vel,
            mean_vel,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍
        context_k = context.repeat_interleave(K, dim=0)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程
        x_t_k = torch.randn((bsz * K, t_len, self.input_dim), device=device)
        pred_vel_norm_k = self.sampleFromXt(x_t_k, context_k, infer_scheduler)

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)
        std_vel_k = std_vel.unsqueeze(1)
        mean_vel_k = mean_vel.unsqueeze(1)
        pred_vel_phys = pred_vel_norm * std_vel_k + mean_vel_k

        anchor_phys_k = anchor_phys[..., :2].unsqueeze(1)
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys_k

        # 兼容多维度输出，前两维写入还原后的物理坐标
        all_preds = future_phys.unsqueeze(1).repeat(1, K, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys

        # 计算 minADE_K
        target_phys = future_phys[..., :2].unsqueeze(1)
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)

        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        _, best_k_idx = torch.min(ade_k, dim=1)

        # 提取 K 条轨迹中 ADE 最小的那一条作为最终预测结果
        best_k_idx_exp = best_k_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, self.output_dim)
        best_pred_phys = all_preds.gather(1, best_k_idx_exp).squeeze(1)

        # 缓存多模态采样结果，供外部评估/可视化直接使用
        self.last_minade_all_preds = all_preds.detach()
        self.last_minade_best_idx = best_k_idx.detach()

        ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
        maybe_visualize_future_prediction(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=best_pred_phys,
            valid_mask=valid_mask,
            stage="eval",
            enable_train_vis=self.fut_enable_train_vis,
            enable_eval_vis=self.fut_enable_eval_vis,
            pred_all=all_preds,
            pred_best_idx=best_k_idx,
            meter_per_foot=self.meter_per_foot,
        )

        return best_pred_phys, ade_batch, fde_batch

    # 统一前向入口，默认复用训练路径。
    def forward(
        self,
        hist,
        hist_nbrs,
        mask,
        temporal_mask,
        future,
        op_mask,
        device,
        return_components=False,
    ):
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            device,
            return_components=return_components,
        )

    # 对历史输入的坐标与运动学特征做归一化。
    def normalize(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -10.0, 10.0)
        channels = x_norm.shape[-1]
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -10.0, 10.0)
        return x_norm
