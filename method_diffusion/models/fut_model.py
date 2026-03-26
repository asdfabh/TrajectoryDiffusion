import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.models.planning_heads import BridgeHead, ContextPooler, IntentHead, MotionSummaryHead
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

        # planning 分支参数：瞬时预测帧数、采用的hist 尾部帧数
        self.bridge_tau = int(getattr(args, "bridge_tau", 5))
        self.intent_tail_k = int(getattr(args, "intent_tail_k", 4))

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

        # planning 分支：从 history context 提取意图、桥接与运动摘要 token。
        self.context_pooler = ContextPooler(self.hidden_dim)
        self.intent_head = IntentHead(self.hidden_dim, self.intent_tail_k, self.context_pooler)
        self.bridge_head = BridgeHead(self.hidden_dim, self.intent_tail_k, self.bridge_tau, self.context_pooler)
        self.motion_head = MotionSummaryHead(self.hidden_dim)

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

        # 双空间归一化参数
        # 物理坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        if self.dataset_name == "ngsim":
            self.register_buffer("pos_mean", torch.tensor([0.05076411229651117, -31.318518632454474], dtype=torch.float32), persistent=False)
            self.register_buffer("pos_std", torch.tensor([9.67614343193339, 59.53730335210165], dtype=torch.float32), persistent=False)
            self.register_buffer("va_mean", torch.tensor([21.150308365503957, 0.006041414014469039], dtype=torch.float32), persistent=False)
            self.register_buffer("va_std", torch.tensor([13.598306447881924, 4.505736504111998], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.1502223350250087, 2.951254134709027], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("pos_mean", torch.tensor([-0.39106148272179536, -115.63853904936501], dtype=torch.float32), persistent=False)
            self.register_buffer("pos_std", torch.tensor([9.266579303046143, 98.49671326349531], dtype=torch.float32), persistent=False)
            self.register_buffer("va_mean", torch.tensor([78.09292302772707, -0.04991240184019581], dtype=torch.float32), persistent=False)
            self.register_buffer("va_std", torch.tensor([29.215909315170098, 1.1700240860076556], dtype=torch.float32), persistent=False)
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

    # 基于当前噪声状态与 planning-aware cross 条件预测归一化速度。
    def predictX0(self, x_t, timesteps, cross_tokens, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        cross_encoded = self.context_embedding(cross_tokens)
        return self.dit(
            x=input_embedded,
            t_cond=t_emb,
            cross=cross_encoded,
        )

    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, _ = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        return context

    # 构造送入 future DiT cross-attn 的 planning-aware cross tokens。
    def buildCrossTokens(self, hist, hist_nbrs, mask, temporal_mask):
        context = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)  # [B, T_ctx, H]
        hist_feat = hist[..., :4]  # [B, T_hist, 4]，future 主路径当前固定为 xyva 四维输入。
        intent_token, intent_aux = self.intent_head(context, hist_feat)  # [B, 1, H]
        anchor_pos = hist[:, -1:, :2]  # [B, 1, 2]
        bridge_tokens, bridge_aux = self.bridge_head(
            context=context,
            hist_feat=hist_feat,
            anchor_pos=anchor_pos,
        )  # bridge_tokens: [B, tau, H]
        motion_token = self.motion_head(bridge_tokens, bridge_aux)  # [B, 1, H]
        cross_tokens = torch.cat([context, intent_token, motion_token, bridge_tokens], dim=1)  # [B, T_ctx+tau+2, H]

        planning_aux = {
            "context": context,
            "hist_feat": hist_feat,
            "intent_aux": intent_aux,
            "intent_token": intent_token,
            "bridge_aux": bridge_aux,
            "bridge_tokens": bridge_tokens,
            "motion_token": motion_token,
            "cross_tokens": cross_tokens,
        }
        return cross_tokens, planning_aux

    # 统一计算 fut 分支损失。
    # - planning_aux: planning 分支的辅助输出；为 None 时只计算 diffusion 主损失
    # - lat_enc / lon_enc: [B, 3]，横向/纵向意图 one-hot 标签
    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask, planning_aux=None, lat_enc=None, lon_enc=None, return_parts=False):
        # diffusion 主损失第一部分：直接约束归一化速度预测。
        loss_vel = F.l1_loss(pred_vel_norm, target_vel_norm, reduction="none")

        # 将预测速度还原到物理空间，并从历史锚点开始积分为 future 位置。
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]

        # diffusion 主损失第二部分：约束积分后的物理位置轨迹。
        loss_pos = F.l1_loss(pred_pos_phys, future_phys[..., :2], reduction="none")
        total_loss = loss_vel + self.pos_loss_weight * loss_pos

        # 只在有效 future 帧上聚合 diffusion 主损失。
        valid = valid_mask.unsqueeze(-1)
        numer = (total_loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        total_mean = (numer / denom).mean()

        loss_total = total_mean
        loss_intent = total_mean.new_zeros(())  # 意图分类损失
        loss_bridge = total_mean.new_zeros(())  # bridge 短时轨迹监督损失
        loss_cons = total_mean.new_zeros(())    # diffusion 与 bridge 的一致性损失

        if planning_aux is not None:
            # bridge 分支：监督前 tau 帧的短时连接轨迹。
            bridge_aux = planning_aux["bridge_aux"]
            bridge_pos = bridge_aux["bridge_pos"]  # [B, tau, 2]
            bridge_vel = bridge_aux["bridge_vel"]  # [B, tau, 2]
            tau = min(self.bridge_tau, future_phys.size(1), bridge_pos.size(1))
            if tau > 0:
                gt_bridge_pos = future_phys[:, :tau, :2]
                gt_bridge_vel = gt_bridge_pos - torch.cat([anchor_phys[..., :2], gt_bridge_pos[:, :-1, :]], dim=1)
                valid_bridge = valid_mask[:, :tau].unsqueeze(-1)
                denom_bridge = valid_bridge.sum() * 2.0 + 1e-6
                loss_bridge_vel = (torch.abs(bridge_vel[:, :tau, :] - gt_bridge_vel) * valid_bridge).sum() / denom_bridge
                loss_bridge_pos = (torch.abs(bridge_pos[:, :tau, :] - gt_bridge_pos) * valid_bridge).sum() / denom_bridge
                loss_bridge = loss_bridge_vel + 1.5 * loss_bridge_pos
                loss_cons = (torch.abs(pred_pos_phys[:, :tau, :] - bridge_pos[:, :tau, :].detach()) * valid_bridge).sum() / denom_bridge

            # intent 分支：使用 lat/lon one-hot 标签做横向/纵向意图分类监督。
            intent_aux = planning_aux["intent_aux"]
            loss_intent_lat = total_mean.new_zeros(())
            loss_intent_lon = total_mean.new_zeros(())
            if lat_enc is not None:
                loss_intent_lat = F.cross_entropy(intent_aux["logits_lat"], lat_enc.argmax(dim=-1).long())
            if lon_enc is not None:
                loss_intent_lon = F.cross_entropy(intent_aux["logits_lon"], lon_enc.argmax(dim=-1).long())
            loss_intent = loss_intent_lat + loss_intent_lon

            # 总损失 = diffusion 主损失 + planning 辅助损失。
            loss_total = total_mean + 0.4 * loss_bridge + 0.2 * loss_intent + 0.3 * loss_cons

        if not return_parts:
            return loss_total

        vel_mean = ((loss_vel * valid).sum(dim=(1, 2)) / denom).mean()
        pos_mean = ((loss_pos * valid).sum(dim=(1, 2)) / denom).mean()
        parts = {
            "loss_total": loss_total.detach(),
            "loss_diffusion": total_mean.detach(),
            "loss_vel": vel_mean.detach(),
            "loss_pos": pos_mean.detach(),
            "loss_intent": loss_intent.detach(),
            "loss_bridge": loss_bridge.detach(),
            "loss_cons": loss_cons.detach(),
        }
        return loss_total, parts, pred_pos_phys

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
    def sampleFromXt(self, x_t, cross_tokens, infer_scheduler):
        bsz, t_len, _ = x_t.shape
        pred_vel_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_vel_norm = self.predictX0(x_t, timesteps, cross_tokens, pred_vel_cond)
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
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        cross_tokens, planning_aux = self.buildCrossTokens(hist, hist_nbrs, mask, temporal_mask)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        return bsz, t_len, valid_mask, anchor_phys, future_phys, cross_tokens, planning_aux, infer_scheduler, std_vel, mean_vel

    # 执行单个 batch 的 fut 训练前向与损失计算。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, lat_enc=None, lon_enc=None, return_components=False):
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

        cross_tokens, planning_aux = self.buildCrossTokens(hist, hist_nbrs, mask, temporal_mask)

        # 自条件：以一定概率将前一步的预测结果作为下一步的输入条件，增强模型稳定性和一致性。
        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(x_t, timesteps, cross_tokens, pred_vel_cond)
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        # 网络输出：预测的归一化速度。
        pred_vel_norm_t = self.predictX0(x_t, timesteps, cross_tokens, pred_vel_cond)

        loss, loss_parts, pred_pos_phys = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            planning_aux=planning_aux,
            lat_enc=lat_enc,
            lon_enc=lon_enc,
            return_parts=True,
        )

        if self.fut_enable_train_vis:
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            vis_payload = self.buildVisualizationPayload(planning_aux)
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
                pred_instant=None if vis_payload is None else vis_payload["pred_instant"],
                intent_probs=None if vis_payload is None else vis_payload["intent_probs"],
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
            cross_tokens,
            planning_aux,
            infer_scheduler,
            std_vel,
            mean_vel,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        x_t = torch.randn((bsz, t_len, self.input_dim), device=device)
        pred_vel_norm = self.sampleFromXt(x_t, cross_tokens, infer_scheduler)

        # 积分还原：解归一化 -> 累加 -> 拼接绝对锚点。
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        pred_phys_abs = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        vis_payload = self.buildVisualizationPayload(planning_aux)
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
            pred_instant=None if vis_payload is None else vis_payload["pred_instant"],
            intent_probs=None if vis_payload is None else vis_payload["intent_probs"],
            meter_per_foot=self.meter_per_foot,
        )
        if return_aux:
            return pred_phys_abs, ade, fde, vis_payload
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
            cross_tokens,
            planning_aux,
            infer_scheduler,
            std_vel,
            mean_vel,
        ) = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍。
        cross_tokens_k = cross_tokens.repeat_interleave(K, dim=0)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程。
        x_t_k = torch.randn((bsz * K, t_len, self.input_dim), device=device)
        pred_vel_norm_k = self.sampleFromXt(x_t_k, cross_tokens_k, infer_scheduler)

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]。
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)。
        std_vel_k = std_vel.unsqueeze(1)
        mean_vel_k = mean_vel.unsqueeze(1)
        pred_vel_phys = pred_vel_norm * std_vel_k + mean_vel_k

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
        vis_payload = self.buildVisualizationPayload(planning_aux)
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
            pred_instant=None if vis_payload is None else vis_payload["pred_instant"],
            intent_probs=None if vis_payload is None else vis_payload["intent_probs"],
            meter_per_foot=self.meter_per_foot,
        )

        if return_aux:
            return best_pred_phys, ade_batch, fde_batch, vis_payload
        return best_pred_phys, ade_batch, fde_batch

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, lat_enc=None, lon_enc=None, return_components=False):
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            device,
            lat_enc=lat_enc,
            lon_enc=lon_enc,
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

    @staticmethod
    def buildVisualizationPayload(planning_aux):
        if planning_aux is None:
            return None

        intent_aux = planning_aux.get("intent_aux", {})
        bridge_aux = planning_aux.get("bridge_aux", {})
        return {
            "pred_instant": bridge_aux.get("bridge_pos"),
            "intent_probs": {
                "lat": intent_aux.get("p_lat"),
                "lon": intent_aux.get("p_lon"),
            },
        }
