import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.encoder_decoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import maybe_visualize_future_prediction
from method_diffusion.utils.fut_utils import (
    to_valid_mask,
    gather_by_index,
    gather_last_by_valid,
    compute_ade_fde,
    compute_per_mode_distance,
)


def smooth_one_hot(targets, smoothing):
    if smoothing <= 0.0:
        return targets
    num_classes = targets.size(-1)
    return targets * (1.0 - smoothing) + float(smoothing) / float(num_classes)


def weighted_soft_cross_entropy(logits, soft_targets, class_weights=None):
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is None:
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    weights = class_weights.view(1, -1).to(logits.device)
    weighted_targets = soft_targets * weights
    denom = weighted_targets.sum(dim=-1).clamp(min=1e-6)
    return -((weighted_targets * log_probs).sum(dim=-1) / denom).mean()


def build_balanced_weights(freq_values):
    freq = torch.tensor(freq_values, dtype=torch.float32)
    weights = torch.rsqrt(freq.clamp(min=1e-6))
    return weights / weights.mean().clamp(min=1e-6)


class AnchorDecoder(nn.Module):
    # 输入 mode_tokens[...,D] → 输出归一化速度序列[...,T_f,2]
    # 每个模态 token 加上可学习时序查询，经 MLP 解码为逐帧速度预测
    def __init__(self, mode_dim, hidden_dim, future_steps, output_dim=2):
        super().__init__()
        self.future_queries = nn.Parameter(torch.randn(1, future_steps, mode_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.LayerNorm(mode_dim),
            nn.Linear(mode_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, mode_tokens):
        query_shape = [1] * (mode_tokens.dim() - 1) + [self.future_queries.size(1), self.future_queries.size(2)]
        future_queries = self.future_queries.view(*query_shape)
        query = mode_tokens.unsqueeze(-2) + future_queries
        return self.decoder(query)


class IntentConditionedModePrior(nn.Module):
    # 输入 global_token[B,D] → 输出横向/纵向意图概率与结构化 anchor
    # mode = lat_i × lon_j × submode_s，每个 mode 都有显式语义索引
    def __init__(self, hidden_dim, mode_dim, num_lat_classes, num_lon_classes, num_submodes, future_steps, output_dim):
        super().__init__()
        self.num_lat_classes = int(num_lat_classes)
        self.num_lon_classes = int(num_lon_classes)
        self.num_submodes = int(num_submodes)
        self.num_modes = self.num_lat_classes * self.num_lon_classes * self.num_submodes
        self.mode_dim = int(mode_dim)

        self.global_proj = nn.Linear(hidden_dim, mode_dim)
        self.lat_head = nn.Linear(hidden_dim, self.num_lat_classes)
        self.lon_head = nn.Linear(hidden_dim, self.num_lon_classes)

        self.lat_queries = nn.Parameter(torch.randn(self.num_lat_classes, self.mode_dim) * 0.02)
        self.lon_queries = nn.Parameter(torch.randn(self.num_lon_classes, self.mode_dim) * 0.02)
        self.submode_queries = nn.Parameter(torch.randn(self.num_submodes, self.mode_dim) * 0.02)

        self.mode_norm = nn.LayerNorm(self.mode_dim)
        self.submode_logit_head = nn.Linear(self.mode_dim, 1)
        self.anchor_decoder = AnchorDecoder(self.mode_dim, hidden_dim, future_steps, output_dim)

    def forward(self, global_token):
        bsz = global_token.size(0)
        base_ctx = self.global_proj(global_token).view(bsz, 1, 1, 1, self.mode_dim)
        mode_seed = (
            base_ctx
            + self.lat_queries.view(1, self.num_lat_classes, 1, 1, self.mode_dim)
            + self.lon_queries.view(1, 1, self.num_lon_classes, 1, self.mode_dim)
            + self.submode_queries.view(1, 1, 1, self.num_submodes, self.mode_dim)
        )
        mode_tokens = self.mode_norm(mode_seed)
        submode_logits = self.submode_logit_head(mode_tokens).squeeze(-1)
        submode_log_probs = F.log_softmax(submode_logits, dim=-1)

        lat_logits = self.lat_head(global_token)
        lon_logits = self.lon_head(global_token)
        lat_log_probs = F.log_softmax(lat_logits, dim=-1).view(bsz, self.num_lat_classes, 1, 1)
        lon_log_probs = F.log_softmax(lon_logits, dim=-1).view(bsz, 1, self.num_lon_classes, 1)
        mode_log_probs = lat_log_probs + lon_log_probs + submode_log_probs

        anchor_vel_norm = torch.clamp(self.anchor_decoder(mode_tokens), -5.0, 5.0)
        return {
            "lat_logits": lat_logits,
            "lon_logits": lon_logits,
            "lat_probs": torch.softmax(lat_logits, dim=-1),
            "lon_probs": torch.softmax(lon_logits, dim=-1),
            "submode_logits": submode_logits,
            "mode_log_probs": mode_log_probs,
            "mode_probs": torch.softmax(mode_log_probs.view(bsz, self.num_modes), dim=-1),
            "mode_tokens": mode_tokens,
            "anchor_vel_norm": anchor_vel_norm,
        }


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()

        # 模型结构超参
        self.hidden_dim = int(args.hidden_dim_fut)
        self.input_dim = int(args.input_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.T = int(args.T_f)
        self.num_lat_classes = int(getattr(args, "num_lat_classes", 3))
        self.num_lon_classes = int(getattr(args, "num_lon_classes", 3))
        self.num_submodes = int(getattr(args, "num_submodes", 2))
        self.num_modes = self.num_lat_classes * self.num_lon_classes * self.num_submodes
        self.mode_dim = int(args.mode_dim)

        # 损失权重与选模策略
        self.lambda_intent = float(getattr(args, "lambda_intent", 1.0))
        self.lambda_mode = float(args.lambda_mode)
        self.lambda_anchor = float(args.lambda_anchor)
        self.lambda_div = float(args.lambda_div)
        self.lambda_x0 = float(args.lambda_x0)
        self.lambda_end = float(args.lambda_end)
        self.use_hard_assignment = int(args.use_hard_assignment) > 0
        self.intent_label_smoothing = max(0.0, min(0.2, float(getattr(args, "intent_label_smoothing", 0.05))))
        self.submode_temperature = max(0.1, float(getattr(args, "submode_temperature", 1.0)))
        self.anchor_div_margin = max(0.0, float(getattr(args, "anchor_div_margin", 2.0)))

        # 扩散推理参数
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.x0_clip = float(args.x0_clip) if float(args.x0_clip) > 0 else None

        # 可视化开关
        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048

        # ── 子模块 ────────────────────────────────────────────────────────────
        self.hist_encoder = HistEncoder(args)
        self.mode_prior = IntentConditionedModePrior(
            hidden_dim=self.hidden_dim,
            mode_dim=self.mode_dim,
            num_lat_classes=self.num_lat_classes,
            num_lon_classes=self.num_lon_classes,
            num_submodes=self.num_submodes,
            future_steps=self.T,
            output_dim=self.output_dim,
        )

        # future residual 嵌入 + 位置编码 + mode 条件投影
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.mode_condition_proj = nn.Linear(self.mode_dim, self.hidden_dim)

        # DiT 扩散主干
        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, int(args.time_embedding_size_fut))
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        dit_block = dit.DiTBlock(self.hidden_dim, int(args.heads_fut), float(args.dropout_fut), int(args.mlp_ratio_fut))
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=int(args.depth_fut), model_type="score")

        # 速度归一化统计量（数据集相关，不参与梯度）
        if self.dataset_name == "ngsim":
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.1502223350250087, 2.951254134709027], dtype=torch.float32), persistent=False)
            self.register_buffer("lat_class_weights", build_balanced_weights([0.957336, 0.033197, 0.009467]), persistent=False)
            self.register_buffer("lon_class_weights", build_balanced_weights([0.614005, 0.179100, 0.206895]), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("vel_mean", torch.tensor([0.004845835373614644, 17.01558226555126], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.10621210903901461, 4.838376260255577], dtype=torch.float32), persistent=False)
            self.register_buffer("lat_class_weights", torch.ones(self.num_lat_classes, dtype=torch.float32), persistent=False)
            self.register_buffer("lon_class_weights", torch.ones(self.num_lon_classes, dtype=torch.float32), persistent=False)
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'. Supported: ngsim, highd")

    # ── 坐标变换 ──────────────────────────────────────────────────────────────

    def buildTargetVelNorm(self, hist, future, device):
        # hist[B,T,D]+future[B,T_f,D] → (anchor_phys[B,1,2], future_phys[B,T_f,2], target_vel_norm[B,T_f,2])
        # 取历史末帧为锚点，计算逐帧位移并标准化为归一化速度目标
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        shifted = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        vel_phys = future_phys - shifted
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        target_vel_norm = torch.clamp((vel_phys - mean_vel) / std_vel, -5.0, 5.0)
        return anchor_phys, future_phys, target_vel_norm

    def decodeVelocityToTrajectory(self, pred_vel_norm, anchor_phys):
        # pred_vel_norm[...,T_f,2]+anchor_phys[...,1,2] → (vel_phys, pos_phys) 同形状
        # 反归一化速度后累积求和得轨迹坐标
        shape_prefix = [1] * (pred_vel_norm.dim() - 1)
        std_vel = self.vel_std.view(*shape_prefix, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(*shape_prefix, 2).to(pred_vel_norm.device)
        vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pos_phys = torch.cumsum(vel_phys, dim=-2) + anchor_phys[..., :2]
        return vel_phys, pos_phys

    def decodeAnchorToPosition(self, anchor_vel_norm, anchor_phys):
        # anchor_vel_norm[...,T_f,2]+anchor_phys[...,1,2] → (vel_phys, pos_phys)
        # 自动补齐 mode 维，兼容单模态/多模态/结构化模态
        while anchor_phys.dim() < anchor_vel_norm.dim():
            anchor_phys = anchor_phys.unsqueeze(1)
        return self.decodeVelocityToTrajectory(anchor_vel_norm, anchor_phys)

    # ── 结构化模态工具 ────────────────────────────────────────────────────────

    def flattenStructuredModes(self, tensor):
        if tensor is None:
            return None
        return tensor.view(tensor.size(0), self.num_modes, *tensor.shape[4:])

    def selectIntentGroup(self, tensor, lat_idx, lon_idx):
        batch_idx = torch.arange(tensor.size(0), device=tensor.device)
        return tensor[batch_idx, lat_idx, lon_idx]

    def buildIntentLosses(self, lat_logits, lon_logits, lat_targets=None, lon_targets=None):
        lat_probs = torch.softmax(lat_logits, dim=-1)
        lon_probs = torch.softmax(lon_logits, dim=-1)

        if lat_targets is None:
            lat_targets = lat_probs.detach()
        if lon_targets is None:
            lon_targets = lon_probs.detach()

        lat_targets = smooth_one_hot(lat_targets.float(), self.intent_label_smoothing)
        lon_targets = smooth_one_hot(lon_targets.float(), self.intent_label_smoothing)
        lat_idx = torch.argmax(lat_targets, dim=-1)
        lon_idx = torch.argmax(lon_targets, dim=-1)

        loss_lat = weighted_soft_cross_entropy(lat_logits, lat_targets, self.lat_class_weights)
        loss_lon = weighted_soft_cross_entropy(lon_logits, lon_targets, self.lon_class_weights)
        return {
            "loss_intent_lat": loss_lat,
            "loss_intent_lon": loss_lon,
            "lat_probs": lat_probs,
            "lon_probs": lon_probs,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
        }

    def selectAssignedMode(self, anchor_vel_norm_group, mode_tokens_group, dist_group):
        # 仅在 GT 联合意图组内选 anchor；hard 取最佳，soft 用组内距离加权
        if self.use_hard_assignment:
            best_idx = torch.argmin(dist_group, dim=1)
            return (
                gather_by_index(anchor_vel_norm_group, best_idx),
                gather_by_index(mode_tokens_group, best_idx),
                best_idx,
            )

        soft_weights = torch.softmax(-dist_group.detach() / self.submode_temperature, dim=1)
        sel_anchor = torch.sum(anchor_vel_norm_group * soft_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        sel_token = torch.sum(mode_tokens_group * soft_weights.unsqueeze(-1), dim=1)
        best_idx = torch.argmax(soft_weights, dim=1)
        return sel_anchor, sel_token, best_idx

    def computeAnchorGroupLosses(self, group_submode_logits, group_anchor_pos_phys, future_phys, valid_mask):
        # 仅监督 GT 联合意图组内的 submode，多样性也只在组内约束
        ade_group, fde_group = compute_per_mode_distance(group_anchor_pos_phys, future_phys, valid_mask)
        dist_group = ade_group + 0.5 * fde_group
        assign_probs = torch.softmax(-dist_group.detach() / self.submode_temperature, dim=1)

        loss_mode = weighted_soft_cross_entropy(group_submode_logits, assign_probs, None)
        loss_anchor = (assign_probs * dist_group).sum(dim=1).mean()

        if self.num_submodes > 1:
            checkpoints = sorted({min(self.T - 1, idx) for idx in [4, 9, 14, 19, self.T - 1]})
            sampled = group_anchor_pos_phys[:, :, checkpoints, :2].reshape(group_anchor_pos_phys.size(0), self.num_submodes, -1)
            pairwise_dist = torch.cdist(sampled, sampled, p=2)
            off_diag = ~torch.eye(self.num_submodes, device=sampled.device, dtype=torch.bool).unsqueeze(0).expand(sampled.size(0), -1, -1)
            loss_div = F.relu(self.anchor_div_margin - pairwise_dist)[off_diag].mean()
        else:
            loss_div = group_anchor_pos_phys.new_tensor(0.0)

        return {
            "loss_mode": loss_mode,
            "loss_anchor": loss_anchor,
            "loss_div": loss_div,
            "assign_probs": assign_probs,
            "dist_group": dist_group,
            "ade_group": ade_group,
            "fde_group": fde_group,
        }

    # ── 扩散去噪 ──────────────────────────────────────────────────────────────

    def predictNoise(self, x_t, timesteps, ctx, mode_token):
        # x_t[B,T_f,2]+t[B]+ctx+mode_token[B,D] → pred_eps[B,T_f,2]
        # 嵌入带噪残差序列，以时间步+模态为 adaLN 条件，历史 context 为 cross-attn KV
        t_emb = self.timestep_embedder(timesteps) + self.mode_condition_proj(mode_token)
        x_emb = self.input_embedding(x_t) + self.pos_embedding(x_t)
        return self.dit(x=x_emb, t_cond=t_emb, cross=ctx["cross_tokens"])

    def predictResidualX0(self, x_t, pred_eps, timesteps):
        # x_t[B,T,2]+pred_eps[B,T,2]+t[B] → pred_r0[B,T,2]
        alpha_cumprod = self.diffusion_scheduler.alphas_cumprod.to(x_t.device)[timesteps].view(-1, 1, 1)
        pred_r0 = (x_t - torch.sqrt(1.0 - alpha_cumprod) * pred_eps) / (torch.sqrt(alpha_cumprod) + 1e-6)
        if self.x0_clip is not None:
            pred_r0 = torch.clamp(pred_r0, -self.x0_clip, self.x0_clip)
        return pred_r0

    def sampleFromXt(self, x_t, ctx, mode_token, infer_scheduler):
        # x_t[B,T,2](噪声) → x_t[B,T,2](干净残差 r0)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((x_t.size(0),), t_scalar, device=x_t.device, dtype=torch.long)
            pred_eps = self.predictNoise(x_t, timesteps, ctx, mode_token)
            try:
                x_t = infer_scheduler.step(pred_eps, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_eps, t, x_t).prev_sample
        if self.x0_clip is not None:
            x_t = torch.clamp(x_t, -self.x0_clip, self.x0_clip)
        return x_t

    # ── 损失计算 ──────────────────────────────────────────────────────────────

    def computeGlobalMetrics(self, mode_logits, anchor_pos_phys, future_phys, valid_mask):
        ade_per_mode, fde_per_mode = compute_per_mode_distance(anchor_pos_phys, future_phys, valid_mask)
        mode_probs = torch.softmax(mode_logits, dim=1)
        top1_idx = torch.argmax(mode_probs, dim=1)
        best_idx = torch.argmin(ade_per_mode, dim=1)

        return {
            "mode_probs": mode_probs,
            "top1_idx": top1_idx,
            "best_idx": best_idx,
            "mode_nll": F.cross_entropy(mode_logits, best_idx).detach(),
            "top1_ade": ade_per_mode.gather(1, top1_idx.unsqueeze(1)).squeeze(1).mean().detach(),
            "top1_fde": fde_per_mode.gather(1, top1_idx.unsqueeze(1)).squeeze(1).mean().detach(),
            "minade_m": ade_per_mode.min(dim=1).values.mean().detach(),
            "minfde_m": fde_per_mode.gather(1, best_idx.unsqueeze(1)).squeeze(1).mean().detach(),
            "ade_per_mode": ade_per_mode.detach(),
            "fde_per_mode": fde_per_mode.detach(),
        }

    def computeRefinementLosses(self, pred_eps, noise, valid_mask, x_t, timesteps,
                                best_anchor_vel_norm, anchor_phys, future_phys):
        # loss_eps=噪声MSE，loss_x0=反推轨迹SmoothL1，loss_end=终点L1
        valid = valid_mask.unsqueeze(-1)
        denom = valid.sum() * noise.size(-1) + 1e-6
        loss_eps = ((pred_eps - noise) ** 2 * valid).sum() / denom

        pred_r0 = self.predictResidualX0(x_t, pred_eps, timesteps)
        pred_vel_norm = best_anchor_vel_norm + pred_r0
        _, pred_pos_phys = self.decodeVelocityToTrajectory(pred_vel_norm, anchor_phys)

        loss_x0 = (F.smooth_l1_loss(pred_pos_phys, future_phys, reduction="none") * valid).sum() / denom

        pred_end, has_valid = gather_last_by_valid(pred_pos_phys, valid_mask)
        target_end, _ = gather_last_by_valid(future_phys, valid_mask)
        loss_end = (torch.abs(pred_end - target_end).sum(dim=-1) * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)

        return {"loss_eps": loss_eps, "loss_x0": loss_x0, "loss_end": loss_end}

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def getAllModePredictions(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, select_topk=None):
        # 全量输入 → 所有 K 模态的完整预测结果字典（含 anchor、轨迹、意图概率）
        bsz, t_len, _ = future.shape
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, _ = self.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])

        mode_logits = prior["mode_log_probs"].view(bsz, self.num_modes)
        mode_tokens = self.flattenStructuredModes(prior["mode_tokens"])
        anchor_vel_norm = self.flattenStructuredModes(prior["anchor_vel_norm"])

        if select_topk is not None and select_topk < self.num_modes:
            topk = max(1, int(select_topk))
            topk_idx = torch.topk(mode_logits, k=topk, dim=1).indices
            mode_logits = mode_logits.gather(1, topk_idx)
            mode_tokens = torch.gather(mode_tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, mode_tokens.size(-1)))
            anchor_vel_norm = torch.gather(
                anchor_vel_norm,
                1,
                topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, anchor_vel_norm.size(2), anchor_vel_norm.size(3)),
            )

        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        num_modes = mode_logits.size(1)
        x_t = torch.randn((bsz * num_modes, t_len, self.input_dim), device=device)
        pred_residual_norm = self.sampleFromXt(
            x_t,
            {k: v.repeat_interleave(num_modes, dim=0) for k, v in ctx.items()},
            mode_tokens.reshape(bsz * num_modes, -1),
            infer_scheduler,
        ).view(bsz, num_modes, t_len, self.output_dim)

        pred_vel_norm = anchor_vel_norm + pred_residual_norm
        _, pred_pos_phys = self.decodeAnchorToPosition(pred_vel_norm, anchor_phys)
        _, anchor_pos_phys = self.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)

        all_preds = future.clone().unsqueeze(1).repeat(1, num_modes, 1, 1)
        all_preds[..., :2] = pred_pos_phys
        ade_per_mode, fde_per_mode = compute_per_mode_distance(all_preds[..., :2], future_phys, valid_mask)
        best_mode_idx = torch.argmin(ade_per_mode, dim=1)
        top1_idx = torch.argmax(torch.softmax(mode_logits, dim=1), dim=1)

        return {
            "valid_mask": valid_mask,
            "anchor_phys": anchor_phys,
            "future_phys": future_phys,
            "ctx": ctx,
            "lat_probs": prior["lat_probs"],
            "lon_probs": prior["lon_probs"],
            "mode_logits": mode_logits,
            "mode_probs": torch.softmax(mode_logits, dim=1),
            "mode_tokens": mode_tokens,
            "anchor_vel_norm": anchor_vel_norm,
            "anchor_pos_phys": anchor_pos_phys,
            "all_pred_vel_norm": pred_vel_norm,
            "all_pred_phys": all_preds,
            "ade_per_mode": ade_per_mode,
            "fde_per_mode": fde_per_mode,
            "best_mode_idx": best_mode_idx,
            "top1_idx": top1_idx,
            "mode_nll": F.cross_entropy(mode_logits, best_mode_idx).detach(),
        }

    # ── 训练 / 评估 前向接口 ───────────────────────────────────────────────────

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                     return_components=False, lat_targets=None, lon_targets=None):
        # intent 监督 + GT 联合意图组内 anchor 监督 + 残差扩散去噪
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, target_vel_norm = self.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])

        mode_logits = prior["mode_log_probs"].view(hist.size(0), self.num_modes)
        mode_tokens = self.flattenStructuredModes(prior["mode_tokens"])
        anchor_vel_norm = self.flattenStructuredModes(prior["anchor_vel_norm"])

        _, anchor_pos_phys = self.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)
        global_metrics = self.computeGlobalMetrics(mode_logits, anchor_pos_phys, future_phys, valid_mask)
        intent_parts = self.buildIntentLosses(prior["lat_logits"], prior["lon_logits"], lat_targets, lon_targets)

        group_anchor_vel_norm = self.selectIntentGroup(prior["anchor_vel_norm"], intent_parts["lat_idx"], intent_parts["lon_idx"])
        group_anchor_pos_phys = self.selectIntentGroup(anchor_pos_phys.view(hist.size(0), self.num_lat_classes, self.num_lon_classes, self.num_submodes, self.T, self.output_dim), intent_parts["lat_idx"], intent_parts["lon_idx"])
        group_mode_tokens = self.selectIntentGroup(prior["mode_tokens"], intent_parts["lat_idx"], intent_parts["lon_idx"])
        group_submode_logits = self.selectIntentGroup(prior["submode_logits"], intent_parts["lat_idx"], intent_parts["lon_idx"])

        anchor_parts = self.computeAnchorGroupLosses(group_submode_logits, group_anchor_pos_phys, future_phys, valid_mask)
        best_anchor_vel_norm, best_mode_token, best_submode_idx = self.selectAssignedMode(
            group_anchor_vel_norm,
            group_mode_tokens,
            anchor_parts["dist_group"],
        )

        residual_target = target_vel_norm - best_anchor_vel_norm
        noise = torch.randn_like(residual_target)
        timesteps = torch.randint(0, self.num_train_timesteps, (future.size(0),), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(residual_target, noise, timesteps)
        pred_eps = self.predictNoise(x_t, timesteps, ctx, best_mode_token)

        ref_parts = self.computeRefinementLosses(
            pred_eps=pred_eps,
            noise=noise,
            valid_mask=valid_mask,
            x_t=x_t,
            timesteps=timesteps,
            best_anchor_vel_norm=best_anchor_vel_norm,
            anchor_phys=anchor_phys,
            future_phys=future_phys,
        )

        loss = (
            self.lambda_intent * (intent_parts["loss_intent_lat"] + intent_parts["loss_intent_lon"])
            + self.lambda_mode * anchor_parts["loss_mode"]
            + self.lambda_anchor * anchor_parts["loss_anchor"]
            + self.lambda_div * anchor_parts["loss_div"]
            + ref_parts["loss_eps"]
            + self.lambda_x0 * ref_parts["loss_x0"]
            + self.lambda_end * ref_parts["loss_end"]
        )

        if self.fut_enable_train_vis:
            vis_out = self.getAllModePredictions(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)
            top1_pred = gather_by_index(vis_out["all_pred_phys"], vis_out["top1_idx"])
            maybe_visualize_future_prediction(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=top1_pred,
                valid_mask=valid_mask,
                stage="train",
                enable_train_vis=self.fut_enable_train_vis,
                enable_eval_vis=self.fut_enable_eval_vis,
                pred_all=vis_out["all_pred_phys"],
                pred_best_idx=vis_out["best_mode_idx"],
                anchor_all=vis_out["anchor_pos_phys"],
                meter_per_foot=self.meter_per_foot,
                intent_probs={"lat": intent_parts["lat_probs"], "lon": intent_parts["lon_probs"]},
            )

        if return_components:
            return loss, {
                "loss_total": loss.detach(),
                "loss_intent_lat": intent_parts["loss_intent_lat"].detach(),
                "loss_intent_lon": intent_parts["loss_intent_lon"].detach(),
                "loss_mode": anchor_parts["loss_mode"].detach(),
                "loss_anchor": anchor_parts["loss_anchor"].detach(),
                "loss_div": anchor_parts["loss_div"].detach(),
                "loss_eps": ref_parts["loss_eps"].detach(),
                "loss_noise": ref_parts["loss_eps"].detach(),
                "loss_x0": ref_parts["loss_x0"].detach(),
                "loss_end": ref_parts["loss_end"].detach(),
                "top1_ade": global_metrics["top1_ade"],
                "top1_fde": global_metrics["top1_fde"],
                "minade_m": global_metrics["minade_m"],
                "minfde_m": global_metrics["minfde_m"],
                "mode_nll": global_metrics["mode_nll"],
                "lat_acc": (torch.argmax(intent_parts["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                "lon_acc": (torch.argmax(intent_parts["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                "best_submode": best_submode_idx.float().mean().detach(),
            }
        return loss

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_aux=False, select_topk=None):
        # 全量输入 → (all_preds[B,K,T,F], mode_probs[B,K], aux 字典)
        outputs = self.getAllModePredictions(
            hist=hist,
            hist_nbrs=hist_nbrs,
            mask=mask,
            temporal_mask=temporal_mask,
            future=future,
            op_mask=op_mask,
            device=device,
            select_topk=select_topk,
        )
        valid_mask = outputs["valid_mask"]
        all_preds = outputs["all_pred_phys"]
        top1_pred = gather_by_index(all_preds, outputs["top1_idx"])
        best_pred = gather_by_index(all_preds, outputs["best_mode_idx"])
        top1_ade, top1_fde = compute_ade_fde(top1_pred, future, valid_mask)
        minade_m = outputs["ade_per_mode"].min(dim=1).values.mean()
        minfde_m = outputs["fde_per_mode"].gather(1, outputs["best_mode_idx"].unsqueeze(1)).squeeze(1).mean()

        maybe_visualize_future_prediction(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=best_pred,
            valid_mask=valid_mask,
            stage="eval",
            enable_train_vis=self.fut_enable_train_vis,
            enable_eval_vis=self.fut_enable_eval_vis,
            pred_all=all_preds,
            pred_best_idx=outputs["best_mode_idx"],
            anchor_all=outputs["anchor_pos_phys"],
            meter_per_foot=self.meter_per_foot,
            intent_probs={"lat": outputs["lat_probs"], "lon": outputs["lon_probs"]},
        )

        aux = {
            "all_pred_phys": all_preds,
            "mode_probs": outputs["mode_probs"],
            "lat_probs": outputs["lat_probs"],
            "lon_probs": outputs["lon_probs"],
            "top1_pred": top1_pred,
            "best_pred": best_pred,
            "top1_ade": top1_ade.detach(),
            "top1_fde": top1_fde.detach(),
            "minade_m": minade_m.detach(),
            "minfde_m": minfde_m.detach(),
            "mode_nll": outputs["mode_nll"],
            "top1_idx": outputs["top1_idx"],
            "best_mode_idx": outputs["best_mode_idx"],
        }
        return all_preds, outputs["mode_probs"], aux

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_aux=False):
        _, _, aux = self.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_aux=True)
        if return_aux:
            return aux["top1_pred"], aux["top1_ade"], aux["top1_fde"], aux
        return aux["top1_pred"], aux["top1_ade"], aux["top1_fde"]

    @torch.no_grad()
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5, return_aux=False):
        _, _, aux = self.forwardEvalMulti(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            device,
            return_aux=True,
            select_topk=min(int(K), self.num_modes),
        )
        if return_aux:
            return aux["best_pred"], aux["minade_m"], aux["minfde_m"], aux
        return aux["best_pred"], aux["minade_m"], aux["minfde_m"]

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                return_components=False, lat_targets=None, lon_targets=None):
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            device,
            return_components=return_components,
            lat_targets=lat_targets,
            lon_targets=lon_targets,
        )
