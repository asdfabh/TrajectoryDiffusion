import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import maybe_visualize_future_prediction


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

        # 可视化参数：仅决定训练和评估阶段是否绘图。
        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048

        # 输入编码模块：分别处理 future 噪声序列和 history context。
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.context_embedding = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)

        # DiT 主干与扩散调度器：负责时间嵌入、去噪建模和 DDIM 调度。
        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.hidden_dim, self.heads, self.dropout, self.mlp_ratio)
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=self.depth, model_type="x_start")

        if self.dataset_name == "ngsim":
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.1502223350250087, 2.951254134709027], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("vel_mean", torch.tensor([0.004845835373614644, 17.01558226555126], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.10621210903901461, 4.838376260255577], dtype=torch.float32), persistent=False)
        else:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}' for fut normalization. Supported: highd, ngsim"
            )

    # 将输出通道掩码转换为按时间步的有效帧掩码。
    @staticmethod
    def toValidMask(op_mask, device):
        return (op_mask[..., 0] > 0.5).float().to(device)

    # 基于当前噪声状态与 history context 预测 epsilon。
    def predictNoise(self, x_t, timesteps, context_tokens):
        t_emb = self.timestep_embedder(timesteps)
        input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
        cross_encoded = self.context_embedding(context_tokens)
        return self.dit(
            x=input_embedded,
            t_cond=t_emb,
            cross=cross_encoded,
        )

    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        context, _ = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        return context

    def computeLoss(self, pred_eps, noise, valid_mask, return_parts=False):
        loss_noise = F.mse_loss(pred_eps, noise, reduction="none")
        valid = valid_mask.unsqueeze(-1)
        denom = valid.sum() * noise.size(-1) + 1e-6
        loss_total = (loss_noise * valid).sum() / denom

        if not return_parts:
            return loss_total

        parts = {
            "loss_total": loss_total.detach(),
            "loss_diffusion": loss_total.detach(),
            "loss_noise": loss_total.detach(),
        }
        return loss_total, parts

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

    def decodeVelocityToTrajectory(self, pred_vel_norm, anchor_phys):
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        return pred_vel_phys, pred_pos_phys

    # 执行推理阶段的 DDIM 采样并输出最终预测结果（归一化速度）。
    def sampleFromXt(self, x_t, context_tokens, infer_scheduler):
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((x_t.size(0),), t_scalar, device=x_t.device, dtype=torch.long)
            pred_eps = self.predictNoise(x_t, timesteps, context_tokens)

            try:
                x_t = infer_scheduler.step(pred_eps, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_eps, t, x_t).prev_sample

        if self.x0_clip is not None:
            x_t = torch.clamp(x_t, -self.x0_clip, self.x0_clip)
        return x_t

    # 统一准备评估阶段所需的条件编码、掩码和调度器参数。
    def prepareEvalInputs(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, device)
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        context_tokens = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        return bsz, t_len, valid_mask, anchor_phys, future_phys, context_tokens, infer_scheduler

    # 执行单个 batch 的 fut 训练前向与损失计算。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        bsz, t_len, _ = future.shape  # [B, T_f, D]
        valid_mask = self.toValidMask(op_mask, device)  # [B, T_f]

        anchor_phys = hist[:, -1:, :self.output_dim]  # [B, 1, 2]
        future_phys = future[..., :self.output_dim]  # [B, T_f, 2]

        # 坐标空间转换：计算 GT 的真实物理帧间位移 (Velocity)。
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys  # [B, T_f, 2]

        # 归一化：将物理速度变为正态分布。
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -5.0, 5.0)

        # 加噪过程。
        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        context_tokens = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)

        # 网络输出：预测的噪声 epsilon。
        pred_eps = self.predictNoise(x_t, timesteps, context_tokens)

        loss, loss_parts = self.computeLoss(
            pred_eps,
            noise,
            valid_mask,
            return_parts=True,
        )

        if self.fut_enable_train_vis:
            infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
            infer_scheduler.set_timesteps(self.num_inference_steps)
            x_t_vis = torch.randn((bsz, t_len, self.input_dim), device=device)
            pred_vel_norm = self.sampleFromXt(x_t_vis, context_tokens, infer_scheduler)
            _, pred_pos_phys = self.decodeVelocityToTrajectory(pred_vel_norm, anchor_phys)
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
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_aux=False):
        (
            bsz,
            t_len,
            valid_mask,
            anchor_phys,
            future_phys,
            infer_scheduler,
            context_tokens,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        x_t = torch.randn((bsz, t_len, self.input_dim), device=device)
        pred_vel_norm = self.sampleFromXt(x_t, context_tokens, infer_scheduler)

        # 积分还原：解归一化 -> 累加 -> 拼接绝对锚点。
        _, pred_pos_phys = self.decodeVelocityToTrajectory(pred_vel_norm, anchor_phys)
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
        if return_aux:
            return pred_phys_abs, ade, fde, None
        return pred_phys_abs, ade, fde

    @torch.no_grad()
    # 执行并行多模态推理评估并返回 minADE 对应结果。
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5, return_aux=False):
        (
            bsz,
            t_len,
            valid_mask,
            anchor_phys,
            future_phys,
            infer_scheduler,
            context_tokens,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍。
        context_tokens_k = context_tokens.repeat_interleave(K, dim=0)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程。
        x_t_k = torch.randn((bsz * K, t_len, self.input_dim), device=device)
        pred_vel_norm_k = self.sampleFromXt(x_t_k, context_tokens_k, infer_scheduler)

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]。
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)。
        std_vel = self.vel_std.view(1, 1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        anchor_phys_k = anchor_phys[..., :2].unsqueeze(1)
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys_k

        # 兼容多维度输出，前两维写入还原后的物理坐标。
        all_preds = future_phys.unsqueeze(1).repeat(1, K, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys

        # 计算 minADE_K。
        target_phys = future_phys[..., :2].unsqueeze(1)
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)

        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        _, best_k_idx = torch.min(ade_k, dim=1)

        # 提取 K 条轨迹中 ADE 最小的那一条作为最终预测结果。
        best_k_idx_exp = best_k_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, self.output_dim)
        best_pred_phys = all_preds.gather(1, best_k_idx_exp).squeeze(1)

        # 缓存多模态采样结果，供外部评估/可视化直接使用。
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

        if return_aux:
            return best_pred_phys, ade_batch, fde_batch, None
        return best_pred_phys, ade_batch, fde_batch

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
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
