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
        self.num_joint_classes = self.num_lat_classes * self.num_lon_classes
        self.num_modes = self.num_lat_classes * self.num_lon_classes * self.num_submodes
        self.mode_dim = int(mode_dim)

        self.global_proj = nn.Linear(hidden_dim, mode_dim)
        self.lat_head = nn.Linear(hidden_dim, self.num_lat_classes)
        self.lon_head = nn.Linear(hidden_dim, self.num_lon_classes)
        self.joint_head = nn.Linear(hidden_dim, self.num_joint_classes)

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
        joint_logits = self.joint_head(global_token)
        joint_log_probs = F.log_softmax(joint_logits, dim=-1).view(bsz, self.num_lat_classes, self.num_lon_classes, 1)
        mode_log_probs = joint_log_probs + submode_log_probs

        anchor_vel_norm = torch.clamp(self.anchor_decoder(mode_tokens), -5.0, 5.0)
        return {
            "lat_logits": lat_logits,
            "lon_logits": lon_logits,
            "joint_logits": joint_logits,
            "lat_probs": torch.softmax(lat_logits, dim=-1),
            "lon_probs": torch.softmax(lon_logits, dim=-1),
            "joint_probs": torch.softmax(joint_logits, dim=-1),
            "submode_logits": submode_logits,
            "mode_log_probs": mode_log_probs,
            "mode_probs": torch.softmax(mode_log_probs.view(bsz, self.num_modes), dim=-1),
            "mode_tokens": mode_tokens,
            "joint_mode_tokens": mode_tokens.view(bsz, self.num_joint_classes, self.num_submodes, self.mode_dim),
            "anchor_vel_norm": anchor_vel_norm,
            "joint_anchor_vel_norm": anchor_vel_norm.view(bsz, self.num_joint_classes, self.num_submodes, anchor_vel_norm.size(-2), anchor_vel_norm.size(-1)),
            "joint_submode_logits": submode_logits.view(bsz, self.num_joint_classes, self.num_submodes),
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
        self.num_joint_classes = self.num_lat_classes * self.num_lon_classes
        self.num_submodes = int(getattr(args, "num_submodes", 2))
        self.num_modes = self.num_lat_classes * self.num_lon_classes * self.num_submodes
        self.mode_dim = int(args.mode_dim)

        # 损失权重与选模策略
        self.lambda_intent = float(getattr(args, "lambda_intent", 1.0))
        self.lambda_joint = float(getattr(args, "lambda_joint", 1.0))
        self.lambda_mode = float(args.lambda_mode)
        self.lambda_anchor = float(args.lambda_anchor)
        self.lambda_div = float(args.lambda_div)
        self.lambda_rank = float(getattr(args, "lambda_rank", 0.5))
        self.lambda_x0 = float(args.lambda_x0)
        self.lambda_end = float(args.lambda_end)
        self.use_hard_assignment = int(args.use_hard_assignment) > 0
        self.intent_label_smoothing = max(0.0, min(0.2, float(getattr(args, "intent_label_smoothing", 0.05))))
        self.submode_temperature = max(0.1, float(getattr(args, "submode_temperature", 1.0)))
        self.anchor_div_margin = max(0.0, float(getattr(args, "anchor_div_margin", 2.0)))
        self.rank_temperature = max(0.1, float(getattr(args, "rank_temperature", 1.0)))
        self.submode_floor_weight = max(0.0, min(1.0, float(getattr(args, "submode_floor_weight", 0.2))))

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
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.mode_dim + 6),
            nn.Linear(self.mode_dim + 6, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

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
            lat_freq = torch.tensor([0.954152, 0.036162, 0.009686], dtype=torch.float32)
            lon_freq = torch.tensor([0.633374, 0.172099, 0.194527], dtype=torch.float32)
            joint_freq = torch.tensor(
                [0.602786, 0.165885, 0.185482, 0.023751, 0.005372, 0.007039, 0.006838, 0.000842, 0.002006],
                dtype=torch.float32,
            )
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.1502223350250087, 2.951254134709027], dtype=torch.float32), persistent=False)
            self.register_buffer("lat_class_weights", build_balanced_weights(lat_freq.tolist()), persistent=False)
            self.register_buffer("lon_class_weights", build_balanced_weights(lon_freq.tolist()), persistent=False)
            self.register_buffer("joint_class_weights", build_balanced_weights(joint_freq.tolist()), persistent=False)
        elif self.dataset_name == "highd":
            lat_freq = torch.tensor([0.942968, 0.026543, 0.030489], dtype=torch.float32)
            lon_freq = torch.tensor([0.975553, 0.011313, 0.013134], dtype=torch.float32)
            joint_freq = torch.tensor(
                [0.918904, 0.011210, 0.012854, 0.026323, 0.000015, 0.000205, 0.030326, 0.000089, 0.000074],
                dtype=torch.float32,
            )
            self.register_buffer("vel_mean", torch.tensor([0.004845835373614644, 17.01558226555126], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.10621210903901461, 4.838376260255577], dtype=torch.float32), persistent=False)
            self.register_buffer("lat_class_weights", build_balanced_weights(lat_freq.tolist()), persistent=False)
            self.register_buffer("lon_class_weights", build_balanced_weights(lon_freq.tolist()), persistent=False)
            self.register_buffer("joint_class_weights", build_balanced_weights(joint_freq.tolist()), persistent=False)
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

    def flattenJointGroups(self, tensor):
        if tensor is None:
            return None
        return tensor.view(tensor.size(0), self.num_joint_classes, self.num_submodes, *tensor.shape[4:])

    def selectIntentGroup(self, tensor, lat_idx, lon_idx):
        batch_idx = torch.arange(tensor.size(0), device=tensor.device)
        return tensor[batch_idx, lat_idx, lon_idx]

    def selectJointGroup(self, tensor, joint_idx):
        batch_idx = torch.arange(tensor.size(0), device=tensor.device)
        return tensor[batch_idx, joint_idx]

    def buildJointTargets(self, lat_targets, lon_targets, joint_targets=None):
        if joint_targets is not None:
            return joint_targets.float()
        if lat_targets is None or lon_targets is None:
            return None
        return (lat_targets.float().unsqueeze(-1) * lon_targets.float().unsqueeze(-2)).reshape(lat_targets.size(0), self.num_joint_classes)

    def buildIntentLosses(self, prior, lat_targets=None, lon_targets=None, joint_targets=None):
        lat_logits = prior["lat_logits"]
        lon_logits = prior["lon_logits"]
        joint_logits = prior["joint_logits"]
        lat_probs = torch.softmax(lat_logits, dim=-1)
        lon_probs = torch.softmax(lon_logits, dim=-1)
        joint_probs = torch.softmax(joint_logits, dim=-1)

        if lat_targets is None:
            lat_targets = lat_probs.detach()
        if lon_targets is None:
            lon_targets = lon_probs.detach()
        joint_targets = self.buildJointTargets(lat_targets, lon_targets, joint_targets)
        if joint_targets is None:
            joint_targets = joint_probs.detach()

        lat_base = lat_targets.float()
        lon_base = lon_targets.float()
        joint_base = joint_targets.float()
        lat_idx = torch.argmax(lat_base, dim=-1)
        lon_idx = torch.argmax(lon_base, dim=-1)
        joint_idx = torch.argmax(joint_base, dim=-1)

        lat_targets = smooth_one_hot(lat_base, self.intent_label_smoothing)
        lon_targets = smooth_one_hot(lon_base, self.intent_label_smoothing)
        joint_targets = smooth_one_hot(joint_base, self.intent_label_smoothing)

        loss_lat = weighted_soft_cross_entropy(lat_logits, lat_targets, self.lat_class_weights)
        loss_lon = weighted_soft_cross_entropy(lon_logits, lon_targets, self.lon_class_weights)
        loss_joint = weighted_soft_cross_entropy(joint_logits, joint_targets, self.joint_class_weights)
        return {
            "loss_intent_lat": loss_lat,
            "loss_intent_lon": loss_lon,
            "loss_intent_joint": loss_joint,
            "lat_probs": lat_probs,
            "lon_probs": lon_probs,
            "joint_probs": joint_probs,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "joint_idx": joint_idx,
        }

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

    def buildScoreFeatures(self, mode_tokens, pred_vel_norm, pred_pos_phys):
        mean_vel = pred_vel_norm.mean(dim=-2)
        end_pos = pred_pos_phys[..., -1, :2]
        mean_pos = pred_pos_phys.mean(dim=-2)
        return torch.cat([mode_tokens, mean_vel, end_pos, mean_pos], dim=-1)

    def scoreCandidates(self, mode_tokens, pred_vel_norm, pred_pos_phys):
        return self.score_head(self.buildScoreFeatures(mode_tokens, pred_vel_norm, pred_pos_phys)).squeeze(-1)

    def repeatContext(self, ctx, repeats):
        return {key: value.repeat_interleave(repeats, dim=0) for key, value in ctx.items()}

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

    def computeGroupRefinementLosses(self, ctx, group_mode_tokens, group_anchor_vel_norm, anchor_phys,
                                     future_phys, target_vel_norm, valid_mask, assign_probs, device):
        # GT 联合意图组内的全部 submode 都参与扩散训练，避免只训单一 teacher
        bsz, group_size, _, _ = group_anchor_vel_norm.shape
        residual_target = target_vel_norm.unsqueeze(1) - group_anchor_vel_norm
        noise = torch.randn_like(residual_target)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz, group_size), device=device).long()

        flat_residual = residual_target.reshape(bsz * group_size, self.T, self.output_dim)
        flat_noise = noise.reshape_as(flat_residual)
        flat_timesteps = timesteps.reshape(-1)
        x_t = self.diffusion_scheduler.add_noise(flat_residual, flat_noise, flat_timesteps)
        pred_eps = self.predictNoise(
            x_t,
            flat_timesteps,
            self.repeatContext(ctx, group_size),
            group_mode_tokens.reshape(bsz * group_size, -1),
        ).view(bsz, group_size, self.T, self.output_dim)

        valid = valid_mask.unsqueeze(1).unsqueeze(-1)
        point_denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6) * pred_eps.size(-1)
        loss_eps_per_mode = (((pred_eps - noise) ** 2) * valid).sum(dim=(2, 3)) / point_denom

        pred_r0 = self.predictResidualX0(
            x_t,
            pred_eps.reshape(bsz * group_size, self.T, self.output_dim),
            flat_timesteps,
        ).view(bsz, group_size, self.T, self.output_dim)
        pred_vel_norm = group_anchor_vel_norm + pred_r0
        _, pred_pos_phys = self.decodeAnchorToPosition(pred_vel_norm, anchor_phys)

        target_pos = future_phys.unsqueeze(1).expand(-1, group_size, -1, -1)
        loss_x0_per_mode = (
            F.smooth_l1_loss(pred_pos_phys, target_pos, reduction="none") * valid
        ).sum(dim=(2, 3)) / point_denom

        flat_pred_pos = pred_pos_phys.reshape(bsz * group_size, self.T, self.output_dim)
        flat_target_pos = future_phys.unsqueeze(1).expand(-1, group_size, -1, -1).reshape(bsz * group_size, self.T, self.output_dim)
        flat_valid_mask = valid_mask.unsqueeze(1).expand(-1, group_size, -1).reshape(bsz * group_size, self.T)
        pred_end, has_valid = gather_last_by_valid(flat_pred_pos, flat_valid_mask)
        target_end, _ = gather_last_by_valid(flat_target_pos, flat_valid_mask)
        loss_end_per_mode = (torch.abs(pred_end - target_end).sum(dim=-1) * has_valid.float()).view(bsz, group_size)

        final_ade, final_fde = compute_per_mode_distance(pred_pos_phys, future_phys, valid_mask)
        final_dist = final_ade + 0.5 * final_fde
        rank_targets = torch.softmax(-final_dist.detach() / self.rank_temperature, dim=1)
        score_logits = self.scoreCandidates(group_mode_tokens, pred_vel_norm, pred_pos_phys)
        loss_rank = weighted_soft_cross_entropy(score_logits, rank_targets, None)

        weight_floor = self.submode_floor_weight / float(group_size)
        train_weights = weight_floor + (1.0 - self.submode_floor_weight) * assign_probs.detach()
        train_weights = train_weights / train_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        return {
            "loss_eps": (train_weights * loss_eps_per_mode).sum(dim=1).mean(),
            "loss_x0": (train_weights * loss_x0_per_mode).sum(dim=1).mean(),
            "loss_end": (train_weights * loss_end_per_mode).sum(dim=1).mean(),
            "loss_rank": loss_rank,
            "score_logits": score_logits,
            "rank_targets": rank_targets,
            "pred_vel_norm": pred_vel_norm,
            "pred_pos_phys": pred_pos_phys,
            "final_ade": final_ade,
            "final_fde": final_fde,
            "best_idx": torch.argmin(final_dist, dim=1),
            "routed_idx": torch.argmax(score_logits, dim=1),
        }

    def runDiffusionForModes(self, ctx, mode_tokens, anchor_vel_norm, anchor_phys, device):
        # 任意一组候选 mode 并行 DDIM 去噪
        bsz, num_modes, _, _ = anchor_vel_norm.shape
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        x_t = torch.randn((bsz * num_modes, self.T, self.input_dim), device=device)
        pred_residual_norm = self.sampleFromXt(
            x_t,
            self.repeatContext(ctx, num_modes),
            mode_tokens.reshape(bsz * num_modes, -1),
            infer_scheduler,
        ).view(bsz, num_modes, self.T, self.output_dim)
        pred_vel_norm = anchor_vel_norm + pred_residual_norm
        _, pred_pos_phys = self.decodeAnchorToPosition(pred_vel_norm, anchor_phys)
        score_logits = self.scoreCandidates(mode_tokens, pred_vel_norm, pred_pos_phys)
        return {
            "pred_residual_norm": pred_residual_norm,
            "pred_vel_norm": pred_vel_norm,
            "pred_pos_phys": pred_pos_phys,
            "score_logits": score_logits,
        }

    # ── 推理 ──────────────────────────────────────────────────────────────────

    def getAllModePredictions(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, select_topk=None):
        # 保留全量 oracle 诊断：生成全部 mode 的 anchor 和最终去噪轨迹
        bsz = future.size(0)
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

        denoise_outputs = self.runDiffusionForModes(ctx, mode_tokens, anchor_vel_norm, anchor_phys, device)
        _, anchor_pos_phys = self.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)

        all_preds = future.clone().unsqueeze(1).repeat(1, mode_logits.size(1), 1, 1)
        all_preds[..., :2] = denoise_outputs["pred_pos_phys"]
        ade_per_mode, fde_per_mode = compute_per_mode_distance(denoise_outputs["pred_pos_phys"], future_phys, valid_mask)
        best_mode_idx = torch.argmin(ade_per_mode, dim=1)
        routed_mode_idx = torch.argmax(torch.softmax(mode_logits, dim=1), dim=1)

        return {
            "valid_mask": valid_mask,
            "anchor_phys": anchor_phys,
            "future_phys": future_phys,
            "ctx": ctx,
            "prior": prior,
            "mode_logits": mode_logits,
            "mode_probs": torch.softmax(mode_logits, dim=1),
            "mode_tokens": mode_tokens,
            "anchor_vel_norm": anchor_vel_norm,
            "anchor_pos_phys": anchor_pos_phys,
            "all_pred_vel_norm": denoise_outputs["pred_vel_norm"],
            "all_pred_phys": all_preds,
            "all_score_logits": denoise_outputs["score_logits"],
            "ade_per_mode": ade_per_mode,
            "fde_per_mode": fde_per_mode,
            "best_mode_idx": best_mode_idx,
            "routed_mode_idx": routed_mode_idx,
            "mode_nll": F.cross_entropy(mode_logits, best_mode_idx).detach(),
        }

    # ── 训练 / 评估 前向接口 ───────────────────────────────────────────────────

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                     return_components=False, lat_targets=None, lon_targets=None, joint_targets=None):
        # 联合意图主路由 + GT 联合意图组内双 submode 扩散训练 + 组内最终排序监督
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, target_vel_norm = self.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.buildIntentLosses(prior, lat_targets, lon_targets, joint_targets)

        all_anchor_vel_norm = self.flattenStructuredModes(prior["anchor_vel_norm"])
        _, all_anchor_pos_phys = self.decodeAnchorToPosition(all_anchor_vel_norm, anchor_phys)
        global_metrics = self.computeGlobalMetrics(
            prior["mode_log_probs"].view(hist.size(0), self.num_modes),
            all_anchor_pos_phys,
            future_phys,
            valid_mask,
        )

        group_anchor_vel_norm = self.selectJointGroup(prior["joint_anchor_vel_norm"], intent_parts["joint_idx"])
        group_mode_tokens = self.selectJointGroup(prior["joint_mode_tokens"], intent_parts["joint_idx"])
        group_submode_logits = self.selectJointGroup(prior["joint_submode_logits"], intent_parts["joint_idx"])
        _, group_anchor_pos_phys = self.decodeAnchorToPosition(group_anchor_vel_norm, anchor_phys)

        anchor_parts = self.computeAnchorGroupLosses(group_submode_logits, group_anchor_pos_phys, future_phys, valid_mask)
        ref_parts = self.computeGroupRefinementLosses(
            ctx=ctx,
            group_mode_tokens=group_mode_tokens,
            group_anchor_vel_norm=group_anchor_vel_norm,
            anchor_phys=anchor_phys,
            future_phys=future_phys,
            target_vel_norm=target_vel_norm,
            valid_mask=valid_mask,
            assign_probs=anchor_parts["assign_probs"],
            device=device,
        )

        loss = (
            self.lambda_intent * (intent_parts["loss_intent_lat"] + intent_parts["loss_intent_lon"])
            + self.lambda_joint * intent_parts["loss_intent_joint"]
            + self.lambda_mode * anchor_parts["loss_mode"]
            + self.lambda_anchor * anchor_parts["loss_anchor"]
            + self.lambda_div * anchor_parts["loss_div"]
            + ref_parts["loss_eps"]
            + self.lambda_rank * ref_parts["loss_rank"]
            + self.lambda_x0 * ref_parts["loss_x0"]
            + self.lambda_end * ref_parts["loss_end"]
        )

        if self.fut_enable_train_vis:
            routed_top1 = future.clone()
            routed_top1[..., :2] = gather_by_index(ref_parts["pred_pos_phys"], ref_parts["routed_idx"])
            maybe_visualize_future_prediction(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=routed_top1,
                valid_mask=valid_mask,
                stage="train",
                enable_train_vis=self.fut_enable_train_vis,
                enable_eval_vis=self.fut_enable_eval_vis,
                pred_all=None,
                pred_best_idx=None,
                anchor_all=group_anchor_pos_phys,
                meter_per_foot=self.meter_per_foot,
                intent_probs={"lat": intent_parts["lat_probs"], "lon": intent_parts["lon_probs"], "joint": intent_parts["joint_probs"]},
                intent_meta={
                    "pred_joint_idx": torch.argmax(intent_parts["joint_probs"], dim=1),
                    "gt_joint_idx": intent_parts["joint_idx"],
                    "routed_sub_idx": ref_parts["routed_idx"],
                    "best_sub_idx": ref_parts["best_idx"],
                },
            )

        if return_components:
            return loss, {
                "loss_total": loss.detach(),
                "loss_intent_lat": intent_parts["loss_intent_lat"].detach(),
                "loss_intent_lon": intent_parts["loss_intent_lon"].detach(),
                "loss_intent_joint": intent_parts["loss_intent_joint"].detach(),
                "loss_mode": anchor_parts["loss_mode"].detach(),
                "loss_anchor": anchor_parts["loss_anchor"].detach(),
                "loss_div": anchor_parts["loss_div"].detach(),
                "loss_eps": ref_parts["loss_eps"].detach(),
                "loss_noise": ref_parts["loss_eps"].detach(),
                "loss_rank": ref_parts["loss_rank"].detach(),
                "loss_x0": ref_parts["loss_x0"].detach(),
                "loss_end": ref_parts["loss_end"].detach(),
                "top1_ade": global_metrics["top1_ade"],
                "top1_fde": global_metrics["top1_fde"],
                "minade_m": global_metrics["minade_m"],
                "minfde_m": global_metrics["minfde_m"],
                "mode_nll": global_metrics["mode_nll"],
                "lat_acc": (torch.argmax(intent_parts["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                "lon_acc": (torch.argmax(intent_parts["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                "joint_acc": (torch.argmax(intent_parts["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
                "best_submode": ref_parts["best_idx"].float().mean().detach(),
                "routed_submode": ref_parts["routed_idx"].float().mean().detach(),
            }
        return loss

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                         return_aux=False, select_topk=None, lat_targets=None, lon_targets=None,
                         joint_targets=None, compute_oracle_all=True):
        # 部署一致评估：先选 top1 联合意图，再在该意图的 2 个 submode 中 rerank
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, _ = self.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.buildIntentLosses(prior, lat_targets, lon_targets, joint_targets)

        pred_joint_idx = torch.argmax(prior["joint_probs"], dim=1)
        pred_group_tokens = self.selectJointGroup(prior["joint_mode_tokens"], pred_joint_idx)
        pred_group_anchors = self.selectJointGroup(prior["joint_anchor_vel_norm"], pred_joint_idx)
        pred_group_out = self.runDiffusionForModes(ctx, pred_group_tokens, pred_group_anchors, anchor_phys, device)
        pred_group_preds = future.clone().unsqueeze(1).repeat(1, self.num_submodes, 1, 1)
        pred_group_preds[..., :2] = pred_group_out["pred_pos_phys"]
        pred_group_ade, pred_group_fde = compute_per_mode_distance(pred_group_out["pred_pos_phys"], future_phys, valid_mask)
        routed_sub_idx = torch.argmax(pred_group_out["score_logits"], dim=1)
        best_pred_sub_idx = torch.argmin(pred_group_ade + 0.5 * pred_group_fde, dim=1)
        selected_mode_idx = pred_joint_idx * self.num_submodes + routed_sub_idx
        top1_pred = gather_by_index(pred_group_preds, routed_sub_idx)
        best_pred = gather_by_index(pred_group_preds, best_pred_sub_idx)
        top1_ade, top1_fde = compute_ade_fde(top1_pred, future, valid_mask)
        best_pred_ade = pred_group_ade.gather(1, best_pred_sub_idx.unsqueeze(1)).squeeze(1).mean()
        best_pred_fde = pred_group_fde.gather(1, best_pred_sub_idx.unsqueeze(1)).squeeze(1).mean()
        rank_nll = F.cross_entropy(pred_group_out["score_logits"], best_pred_sub_idx).detach()

        gt_best_pred = None
        gt_best_ade = future_phys.new_tensor(0.0)
        gt_best_fde = future_phys.new_tensor(0.0)
        if lat_targets is not None or lon_targets is not None or joint_targets is not None:
            gt_joint_idx = intent_parts["joint_idx"]
            gt_group_tokens = self.selectJointGroup(prior["joint_mode_tokens"], gt_joint_idx)
            gt_group_anchors = self.selectJointGroup(prior["joint_anchor_vel_norm"], gt_joint_idx)
            gt_group_out = self.runDiffusionForModes(ctx, gt_group_tokens, gt_group_anchors, anchor_phys, device)
            gt_group_preds = future.clone().unsqueeze(1).repeat(1, self.num_submodes, 1, 1)
            gt_group_preds[..., :2] = gt_group_out["pred_pos_phys"]
            gt_group_ade, gt_group_fde = compute_per_mode_distance(gt_group_out["pred_pos_phys"], future_phys, valid_mask)
            gt_best_idx = torch.argmin(gt_group_ade + 0.5 * gt_group_fde, dim=1)
            gt_best_pred = gather_by_index(gt_group_preds, gt_best_idx)
            gt_best_ade = gt_group_ade.gather(1, gt_best_idx.unsqueeze(1)).squeeze(1).mean()
            gt_best_fde = gt_group_fde.gather(1, gt_best_idx.unsqueeze(1)).squeeze(1).mean()

        all_preds = None
        mode_probs = prior["mode_probs"]
        oracle_best_pred = best_pred
        oracle_best_ade = best_pred_ade.detach()
        oracle_best_fde = best_pred_fde.detach()
        mode_nll = rank_nll
        best_mode_idx = None
        if compute_oracle_all:
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
            all_preds = outputs["all_pred_phys"]
            mode_probs = outputs["mode_probs"]
            best_mode_idx = outputs["best_mode_idx"]
            oracle_best_pred = gather_by_index(all_preds, best_mode_idx)
            oracle_best_ade = outputs["ade_per_mode"].min(dim=1).values.mean().detach()
            oracle_best_fde = outputs["fde_per_mode"].gather(1, best_mode_idx.unsqueeze(1)).squeeze(1).mean().detach()
            mode_nll = outputs["mode_nll"]

        maybe_visualize_future_prediction(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=top1_pred,
            valid_mask=valid_mask,
            stage="eval",
            enable_train_vis=self.fut_enable_train_vis,
            enable_eval_vis=self.fut_enable_eval_vis,
            pred_all=all_preds,
            pred_best_idx=best_mode_idx,
            pred_selected_idx=selected_mode_idx if all_preds is not None else None,
            anchor_all=None,
            meter_per_foot=self.meter_per_foot,
            intent_probs={"lat": prior["lat_probs"], "lon": prior["lon_probs"], "joint": prior["joint_probs"]},
            intent_meta={
                "pred_joint_idx": pred_joint_idx,
                "gt_joint_idx": None if lat_targets is None and lon_targets is None and joint_targets is None else intent_parts["joint_idx"],
                "oracle_joint_idx": None if best_mode_idx is None else torch.div(best_mode_idx, self.num_submodes, rounding_mode="floor"),
                "routed_sub_idx": routed_sub_idx,
                "best_sub_idx": best_pred_sub_idx,
            },
        )

        aux = {
            "all_pred_phys": all_preds,
            "mode_probs": mode_probs,
            "lat_probs": prior["lat_probs"],
            "lon_probs": prior["lon_probs"],
            "joint_probs": prior["joint_probs"],
            "top1_pred": top1_pred,
            "best_pred": oracle_best_pred,
            "best_pred_intent": best_pred,
            "best_gt_intent": gt_best_pred,
            "top1_ade": top1_ade.detach(),
            "top1_fde": top1_fde.detach(),
            "best_pred_ade": best_pred_ade.detach(),
            "best_pred_fde": best_pred_fde.detach(),
            "best_gt_ade": gt_best_ade.detach(),
            "best_gt_fde": gt_best_fde.detach(),
            "minade_m": oracle_best_ade,
            "minfde_m": oracle_best_fde,
            "rank_nll": rank_nll,
            "mode_nll": mode_nll,
            "pred_joint_idx": pred_joint_idx,
            "gt_joint_idx": intent_parts["joint_idx"],
            "oracle_joint_idx": None if best_mode_idx is None else torch.div(best_mode_idx, self.num_submodes, rounding_mode="floor"),
            "routed_sub_idx": routed_sub_idx,
            "best_pred_sub_idx": best_pred_sub_idx,
            "best_mode_idx": best_mode_idx,
            "lat_acc": (torch.argmax(prior["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
            "lon_acc": (torch.argmax(prior["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
            "joint_acc": (torch.argmax(prior["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
        }
        return all_preds if all_preds is not None else pred_group_preds, mode_probs, aux

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                    return_aux=False, lat_targets=None, lon_targets=None, joint_targets=None):
        _, _, aux = self.forwardEvalMulti(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            device,
            return_aux=True,
            lat_targets=lat_targets,
            lon_targets=lon_targets,
            joint_targets=joint_targets,
            compute_oracle_all=False,
        )
        if return_aux:
            return aux["top1_pred"], aux["top1_ade"], aux["top1_fde"], aux
        return aux["top1_pred"], aux["top1_ade"], aux["top1_fde"]

    @torch.no_grad()
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                           K=5, return_aux=False, lat_targets=None, lon_targets=None, joint_targets=None):
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
            lat_targets=lat_targets,
            lon_targets=lon_targets,
            joint_targets=joint_targets,
            compute_oracle_all=True,
        )
        if return_aux:
            return aux["best_pred"], aux["minade_m"], aux["minfde_m"], aux
        return aux["best_pred"], aux["minade_m"], aux["minfde_m"]

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                return_components=False, lat_targets=None, lon_targets=None, joint_targets=None):
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
            joint_targets=joint_targets,
        )
