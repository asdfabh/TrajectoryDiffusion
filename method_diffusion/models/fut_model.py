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

        # 模型结构参数：控制 DiT 主干维度、层数和 future 序列长度。
        self.hidden_dim = int(args.hidden_dim_fut)
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
        self.huber_delta = max(1e-4, float(args.fut_huber_delta))
        self.pos_loss_weight = max(0.0, float(args.fut_pos_loss_weight))
        self.use_intent_cond = int(getattr(args, "fut_use_intent_cond", 1)) > 0
        self.use_residual_diffusion = int(getattr(args, "fut_use_residual_diffusion", 1)) > 0
        self.intent_loss_weight = float(getattr(args, "intent_loss_weight", 0.2))
        self.coarse_loss_weight = float(getattr(args, "coarse_loss_weight", 0.5))
        self.lat_intent_weight = float(getattr(args, "lat_intent_weight", 1.0))
        self.lon_intent_weight = float(getattr(args, "lon_intent_weight", 0.5))
        self.intent_teacher_forcing_prob = float(getattr(args, "intent_teacher_forcing_prob", 0.7))
        self.intent_cond_drop_prob = float(getattr(args, "intent_cond_drop_prob", 0.1))
        self.semantic_fuse_hidden_dim = int(getattr(args, "semantic_fuse_hidden_dim", self.hidden_dim))

        # 可视化参数：仅决定训练和评估阶段是否绘图。
        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048
        self.use_attention_pooling = int(getattr(args, "fut_use_attention_pooling", 1)) > 0
        self.use_split_cond_adaln = int(getattr(args, "fut_use_split_cond_adaln", 1)) > 0

        # 输入编码模块：分别处理 future 噪声序列和 history context。
        self.input_embedding = nn.Linear(self.output_dim * 3, self.hidden_dim)
        self.context_embedding = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)
        self.scene_pool_score = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.scene_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.lat_intent_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.lon_intent_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        self.intent_embed = nn.Sequential(
            nn.Linear(6, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.semantic_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.semantic_fuse_hidden_dim),
            nn.SiLU(),
            nn.Linear(self.semantic_fuse_hidden_dim, self.hidden_dim),
        )
        self.coarse_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.T * self.output_dim),
        )

        # DiT 主干与扩散调度器：负责时间嵌入、去噪建模和 DDIM 调度。
        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(
            self.hidden_dim,
            self.heads,
            self.dropout,
            self.mlp_ratio,
            use_split_cond=self.use_split_cond_adaln,
        )
        final_layer = dit.FinalLayer(
            self.hidden_dim,
            self.T,
            self.output_dim,
            use_split_cond=self.use_split_cond_adaln,
        )
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

    # 从完整 history context 中提取 scene-level 全局条件，默认使用 attention pooling。
    def buildSceneEmbedding(self, context):
        if not self.use_attention_pooling:
            return context[:, -1, :]

        attn_scores = self.scene_pool_score(context).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        scene_vec = torch.sum(context * attn_weights.unsqueeze(-1), dim=1)
        return self.scene_proj(scene_vec)

    # 统一编码 history side 条件，输出 cross-attn context 和 scene summary。
    def encodeHistoryCondition(self, hist_norm, hist_nbrs_norm, mask, temporal_mask):
        context, _ = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        scene_summary = self.buildSceneEmbedding(context)
        return context, scene_summary

    # 基于场景摘要预测横纵向意图分布。
    def predictIntent(self, scene_summary):
        lat_logits = self.lat_intent_head(scene_summary)
        lon_logits = self.lon_intent_head(scene_summary)
        return lat_logits, lon_logits

    # 构造意图条件向量，训练阶段支持 teacher forcing 与条件 dropout。
    def buildIntentCondition(self, lat_logits, lon_logits, lat_enc=None, lon_enc=None):
        lat_prob = torch.softmax(lat_logits, dim=-1)
        lon_prob = torch.softmax(lon_logits, dim=-1)

        if self.training and lat_enc is not None and lon_enc is not None:
            use_gt = (
                torch.rand(lat_prob.size(0), 1, device=lat_prob.device)
                < self.intent_teacher_forcing_prob
            ).float()
            lat_prob = use_gt * lat_enc + (1.0 - use_gt) * lat_prob
            lon_prob = use_gt * lon_enc + (1.0 - use_gt) * lon_prob

        intent_prob = torch.cat([lat_prob, lon_prob], dim=-1)

        if self.training and self.intent_cond_drop_prob > 0:
            drop_mask = (
                torch.rand(intent_prob.size(0), 1, device=intent_prob.device)
                < self.intent_cond_drop_prob
            ).float()
            intent_prob = intent_prob * (1.0 - drop_mask)

        intent_emb = self.intent_embed(intent_prob)
        if not self.use_intent_cond:
            intent_emb = torch.zeros_like(intent_emb)
        return intent_emb, lat_prob, lon_prob

    # 组合场景摘要与意图嵌入，生成 AdaLN 使用的语义条件。
    def buildSemanticCondition(self, scene_summary, intent_emb):
        if not self.use_intent_cond:
            return scene_summary
        return self.semantic_fuse(torch.cat([scene_summary, intent_emb], dim=-1))

    # 预测 coarse future 轨迹，输出未来相对位置序列。
    def predictCoarseTrajectory(self, scene_summary, intent_emb):
        coarse_input = torch.cat([scene_summary, intent_emb], dim=-1)
        coarse_delta = self.coarse_head(coarse_input).view(-1, self.T, self.output_dim)
        coarse_pos = torch.cumsum(coarse_delta, dim=1)
        return coarse_pos

    # 对速度张量做无原地操作的归一化，避免 autograd 版本冲突。
    def normalizeVelocity(self, vel_phys, device):
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        vel_xy_norm = torch.clamp((vel_phys[..., :2] - mean_vel) / std_vel, -5.0, 5.0)
        if vel_phys.size(-1) <= 2:
            return vel_xy_norm
        return torch.cat([vel_xy_norm, vel_phys[..., 2:]], dim=-1)

    # 将未来相对位置轨迹转换为归一化速度序列。
    def trajectoryToVelocityNorm(self, traj_phys, anchor_phys, device):
        shifted = torch.cat([anchor_phys, traj_phys[:, :-1, :]], dim=1)
        vel_phys = traj_phys - shifted
        return self.normalizeVelocity(vel_phys, device)

    # 将归一化速度序列还原为物理速度与相对位置。
    def velocityNormToPosition(self, vel_norm, anchor_phys, device):
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        vel_phys = vel_norm[..., :2] * std_vel + mean_vel
        pos_phys = torch.cumsum(vel_phys, dim=1) + anchor_phys[..., :2]
        return vel_phys, pos_phys

    # 在有效帧范围内计算 SmoothL1，供 coarse 轨迹监督使用。
    def maskedSmoothL1(self, pred, target, valid_mask):
        loss = F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_delta)
        valid = valid_mask.unsqueeze(-1)
        numer = (loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

    # 基于当前噪声状态与历史条件预测归一化残差。
    def predictX0(self, x_t, timesteps, context_aligned, semantic_cond, pred_x0_cond, coarse_vel_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond, coarse_vel_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        context_with_pos = self.context_embedding(context_aligned) + self.pos_embedding(context_aligned)
        return self.dit(x=input_embedded, t_cond=t_emb, semantic_cond=semantic_cond, cross=context_with_pos)

    # 计算训练使用的速度损失与物理位置损失。
    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=False):
        # 速度域微观约束 (Huber Loss)：在归一化空间算，梯度稳定，防高频抖动
        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)

        # 积分回物理位置域 (宏观绝对约束)
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)

        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]

        # 物理空间位置损失
        loss_pos = F.smooth_l1_loss(pred_pos_phys, future_phys[..., :2], reduction="none", beta=self.huber_delta)
        total_loss = loss_vel + self.pos_loss_weight * loss_pos

        # 损失函数指标计算：仅在有效帧上平均，避免无效帧稀释梯度
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

    # 执行推理阶段的 DDIM 采样并输出归一化残差预测。
    def sampleFromXt(self, x_t, context_aligned, semantic_cond, coarse_vel_norm, infer_scheduler):
        bsz, t_len, _ = x_t.shape
        pred_residual_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_residual = self.predictX0(
                x_t,
                timesteps,
                context_aligned,
                semantic_cond,
                pred_residual_cond,
                coarse_vel_norm,
            )
            if self.x0_clip is not None:
                pred_residual = torch.clamp(pred_residual, -self.x0_clip, self.x0_clip)

            pred_residual_cond = pred_residual.detach()
            try:
                x_t = infer_scheduler.step(pred_residual, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_residual, t, x_t).prev_sample
        return pred_residual_cond

    # 统一准备评估阶段所需的条件编码、掩码和调度器参数。
    def prepareEvalInputs(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, device)
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, scene_summary = self.encodeHistoryCondition(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        lat_logits, lon_logits = self.predictIntent(scene_summary)
        intent_emb, _, _ = self.buildIntentCondition(lat_logits, lon_logits, None, None)
        semantic_cond = self.buildSemanticCondition(scene_summary, intent_emb)
        coarse_pos = self.predictCoarseTrajectory(scene_summary, intent_emb)
        coarse_vel_norm = self.trajectoryToVelocityNorm(coarse_pos, anchor_phys, device)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        return (
            bsz,
            t_len,
            valid_mask,
            anchor_phys,
            future_phys,
            context,
            scene_summary,
            semantic_cond,
            coarse_vel_norm,
            coarse_pos,
            infer_scheduler,
            std_vel,
            mean_vel,
            lat_logits,
            lon_logits,
        )

    # 执行单个 batch 的 fut 训练前向与损失计算。
    def forwardTrain(
        self,
        hist,
        hist_nbrs,
        mask,
        temporal_mask,
        future,
        op_mask,
        lat_enc=None,
        lon_enc=None,
        device=None,
        return_components=False,
    ):
        if device is None:
            device = future.device

        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, device)

        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        # 坐标空间转换：计算 GT 的真实物理帧间位移 (Velocity)
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys

        # 归一化：将物理速度变为正态分布
        target_vel_norm = self.normalizeVelocity(target_vel_phys, device)

        # 加噪过程
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, scene_summary = self.encodeHistoryCondition(hist_norm, hist_nbrs_norm, mask, temporal_mask)

        lat_logits, lon_logits = self.predictIntent(scene_summary)
        intent_emb, _, _ = self.buildIntentCondition(lat_logits, lon_logits, lat_enc, lon_enc)
        semantic_cond = self.buildSemanticCondition(scene_summary, intent_emb)

        coarse_pos = self.predictCoarseTrajectory(scene_summary, intent_emb)
        coarse_vel_norm = self.trajectoryToVelocityNorm(coarse_pos, anchor_phys, device)

        if self.use_residual_diffusion:
            target_residual = target_vel_norm - coarse_vel_norm
        else:
            target_residual = target_vel_norm

        noise = torch.randn_like(target_residual)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_residual, noise, timesteps)

        # 自条件：以一定概率将前一步的预测结果作为下一步的输入条件，增强模型稳定性和一致性
        pred_residual_cond = torch.zeros_like(target_residual)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_residual = self.predictX0(
                        x_t,
                        timesteps,
                        context,
                        semantic_cond,
                        pred_residual_cond,
                        coarse_vel_norm,
                    )
                pred_residual_cond = prev_pred_residual.detach() * use_sc

        pred_residual = self.predictX0(
            x_t,
            timesteps,
            context,
            semantic_cond,
            pred_residual_cond,
            coarse_vel_norm,
        )

        if self.use_residual_diffusion:
            pred_vel_norm_t = coarse_vel_norm + pred_residual
        else:
            pred_vel_norm_t = pred_residual

        # 传入双轨 Loss 损失函数
        motion_loss, motion_parts = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            return_parts=True,
        )

        if lat_enc is not None and lon_enc is not None:
            lat_gt = lat_enc.argmax(dim=-1)
            lon_gt = lon_enc.argmax(dim=-1)
            loss_int = (
                self.lat_intent_weight * F.cross_entropy(lat_logits, lat_gt) +
                self.lon_intent_weight * F.cross_entropy(lon_logits, lon_gt)
            )
        else:
            loss_int = motion_loss.new_zeros(())

        loss_coarse = self.maskedSmoothL1(coarse_pos, future_phys[..., :2], valid_mask)
        loss = motion_loss + self.intent_loss_weight * loss_int + self.coarse_loss_weight * loss_coarse

        if self.fut_enable_train_vis:
            _, pred_pos_phys = self.velocityNormToPosition(pred_vel_norm_t, anchor_phys, device)

            # 兼容多维度输出拼接
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            maybe_visualize_future_prediction(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=pred_phys_abs,
                coarse_pred=coarse_pos,
                valid_mask=valid_mask,
                stage="train",
                enable_train_vis=self.fut_enable_train_vis,
                enable_eval_vis=self.fut_enable_eval_vis,
                meter_per_foot=self.meter_per_foot,
            )

        if return_components:
            parts = {
                "loss_total": loss.detach(),
                "loss_vel": motion_parts["loss_vel"].detach(),
                "loss_pos": motion_parts["loss_pos"].detach(),
                "loss_int": loss_int.detach(),
                "loss_coarse": loss_coarse.detach(),
            }
            return loss, parts
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
            scene_summary,
            semantic_cond,
            coarse_vel_norm,
            coarse_pos,
            infer_scheduler,
            std_vel,
            mean_vel,
            lat_logits,
            lon_logits,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        x_t = torch.randn((bsz, t_len, self.output_dim), device=device)
        pred_residual_norm = self.sampleFromXt(x_t, context, semantic_cond, coarse_vel_norm, infer_scheduler)

        if self.use_residual_diffusion:
            pred_vel_norm = coarse_vel_norm + pred_residual_norm
        else:
            pred_vel_norm = pred_residual_norm

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
            coarse_pred=coarse_pos,
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
            scene_summary,
            semantic_cond,
            coarse_vel_norm,
            coarse_pos,
            infer_scheduler,
            std_vel,
            mean_vel,
            lat_logits,
            lon_logits,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍
        context_k = context.repeat_interleave(K, dim=0)
        semantic_cond_k = semantic_cond.repeat_interleave(K, dim=0)
        coarse_vel_norm_k = coarse_vel_norm.repeat_interleave(K, dim=0)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程
        x_t_k = torch.randn((bsz * K, t_len, self.output_dim), device=device)
        pred_residual_norm_k = self.sampleFromXt(
            x_t_k,
            context_k,
            semantic_cond_k,
            coarse_vel_norm_k,
            infer_scheduler,
        )

        if self.use_residual_diffusion:
            pred_vel_norm_k = coarse_vel_norm_k + pred_residual_norm_k
        else:
            pred_vel_norm_k = pred_residual_norm_k

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)
        std_vel_k = std_vel.unsqueeze(1)  # [1, 1, 1, 2]
        mean_vel_k = mean_vel.unsqueeze(1)
        pred_vel_phys = pred_vel_norm * std_vel_k + mean_vel_k

        anchor_phys_k = anchor_phys[..., :2].unsqueeze(1)  # [bsz, 1, 1, 2]
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys_k

        # 兼容多维度输出，前两维写入还原后的物理坐标
        all_preds = future_phys.unsqueeze(1).repeat(1, K, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys

        # 计算 minADE_K
        target_phys = future_phys[..., :2].unsqueeze(1)  # [bsz, 1, t_len, 2]
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)

        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        min_ade, best_k_idx = torch.min(ade_k, dim=1)

        # 提取 K 条轨迹中ADE最小的那一条作为最终预测结果
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
            coarse_pred=coarse_pos,
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
        lat_enc=None,
        lon_enc=None,
        device=None,
        return_components=False,
    ):
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            lat_enc,
            lon_enc,
            device,
            return_components=return_components,
        )

    # 对历史输入的坐标与运动学特征做归一化。
    def normalize(self, x):
        pos_norm = torch.clamp((x[..., 0:2] - self.pos_mean) / self.pos_std, -10.0, 10.0)
        channels = x.shape[-1]
        if channels <= 2:
            return pos_norm

        if channels >= 4:
            va_norm = torch.clamp((x[..., 2:4] - self.va_mean) / self.va_std, -10.0, 10.0)
            if channels == 4:
                return torch.cat([pos_norm, va_norm], dim=-1)
            return torch.cat([pos_norm, va_norm, x[..., 4:]], dim=-1)

        return torch.cat([pos_norm, x[..., 2:]], dim=-1)
