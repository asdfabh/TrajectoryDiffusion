import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import maybe_visualize_future_prediction


_FUT_NORMALIZATION_PRESETS = {
    "ngsim": {
        "pos_mean": [0.05076411229651117, -31.318518632454474],
        "pos_std": [9.67614343193339, 59.53730335210165],
        "va_mean": [21.150308365503957, 0.006041414014469039],
        "va_std": [13.598306447881924, 4.505736504111998],
        "vel_mean": [-0.004181504611623526, 5.041936610524995],
        "vel_std": [0.1502223350250087, 2.951254134709027],
    },
    # highD 先保留占位值，收到统计参数后直接替换这一组常量即可。
    "highd": {
        "pos_mean": [-0.39106148272179536, -115.63853904936501],
        "pos_std": [9.266579303046143, 98.49671326349531],
        "va_mean": [78.09292302772707, -0.04991240184019581],
        "va_std": [29.215909315170098, 1.1700240860076556],
        "vel_mean": [0.004845835373614644, 17.01558226555126],
        "vel_std": [0.10621210903901461, 4.838376260255577],
    },
}


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()

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
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=self.depth, model_type="x_start")

        norm_params = self.loadNormalizationParams()
        # 双空间归一化参数
        # 物理坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        self.register_buffer("pos_mean", norm_params["pos_mean"], persistent=False)
        self.register_buffer("pos_std", norm_params["pos_std"], persistent=False)
        self.register_buffer("va_mean", norm_params["va_mean"], persistent=False)
        self.register_buffer("va_std", norm_params["va_std"], persistent=False)
        # 帧间相对位移归一化参数 (Velocity)
        self.register_buffer("vel_mean", norm_params["vel_mean"], persistent=False)
        self.register_buffer("vel_std", norm_params["vel_std"], persistent=False)

    def loadNormalizationParams(self):
        params = _FUT_NORMALIZATION_PRESETS.get(self.dataset_name)
        if params is None:
            supported = ", ".join(sorted(_FUT_NORMALIZATION_PRESETS.keys()))
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}' for fut normalization. Supported: {supported}"
            )
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in params.items()
        }

    # 将输出通道掩码转换为按时间步的有效帧掩码。
    @staticmethod
    def toValidMask(op_mask, device):
        return (op_mask[..., 0] > 0.5).float().to(device)

    # 基于当前噪声状态与历史条件预测归一化速度。
    def predictX0(self, x_t, timesteps, context_aligned, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        context_encoded = self.context_embedding(context_aligned)
        return self.dit(
            x=input_embedded,
            t_cond=t_emb,
            cross=context_encoded,
        )

    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, _ = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        return context

    # 计算训练使用的速度损失与物理位置损失。
    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=False):
        loss_vel = F.l1_loss(pred_vel_norm, target_vel_norm, reduction="none")

        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        loss_pos = F.l1_loss(pred_pos_phys, future_phys[..., :2], reduction="none")
        total_loss = loss_vel + self.pos_loss_weight * loss_pos

        valid = valid_mask.unsqueeze(-1)
        numer = (total_loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        total_mean = (numer / denom).mean()

        if not return_parts:
            return total_mean

        vel_mean = ((loss_vel * valid).sum(dim=(1, 2)) / denom).mean()
        pos_mean = ((loss_pos * valid).sum(dim=(1, 2)) / denom).mean()
        parts = {
            "loss_total": total_mean.detach(),
            "loss_diffusion": total_mean.detach(),
            "loss_vel": vel_mean.detach(),
            "loss_pos": pos_mean.detach(),
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
        pred_vel_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_vel_norm = self.predictX0(x_t, timesteps, context_aligned, pred_vel_cond)
            if self.x0_clip is not None:
                pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)

            pred_vel_cond = pred_vel_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t).prev_sample
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
                    prev_pred_vel = self.predictX0(x_t, timesteps, context, pred_vel_cond)
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        # 网络输出：预测的归一化速度
        pred_vel_norm_t = self.predictX0(x_t, timesteps, context, pred_vel_cond)

        diffusion_loss, loss_parts = self.computeLoss(
            pred_vel_norm_t,
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
            pred_vel_phys_t = pred_vel_norm_t * std_vel + mean_vel
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
