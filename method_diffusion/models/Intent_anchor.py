import torch
import torch.nn.functional as F
from diffusers.schedulers import DDIMScheduler
from torch import nn

from method_diffusion.models import dit_fut as dit
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding


class AnchorDecoder(nn.Module):
    # 输入 mode_tokens[...,D] → 输出归一化速度序列[...,T_f,2]
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
        return self.decoder(mode_tokens.unsqueeze(-2) + future_queries)


class IntentConditionedModePrior(nn.Module):
    # 输入 global_token[B,D] → 输出横纵向/联合意图概率与结构化 anchor
    def __init__(self, hidden_dim, mode_dim, num_lat_classes, num_lon_classes, num_submodes, future_steps, output_dim):
        super().__init__()
        self.num_lat_classes = int(num_lat_classes)
        self.num_lon_classes = int(num_lon_classes)
        self.num_submodes = int(num_submodes)
        self.num_joint_classes = self.num_lat_classes * self.num_lon_classes
        self.num_modes = self.num_joint_classes * self.num_submodes
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
        joint_log_probs = F.log_softmax(joint_logits, dim=-1).view(
            bsz,
            self.num_lat_classes,
            self.num_lon_classes,
            1,
        )
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
            "joint_anchor_vel_norm": anchor_vel_norm.view(
                bsz,
                self.num_joint_classes,
                self.num_submodes,
                anchor_vel_norm.size(-2),
                anchor_vel_norm.size(-1),
            ),
            "joint_submode_logits": submode_logits.view(bsz, self.num_joint_classes, self.num_submodes),
        }


class FutureMotionCodec(nn.Module):
    def __init__(self, dataset_name, output_dim=2):
        super().__init__()
        self.dataset_name = str(dataset_name).strip().lower()
        self.output_dim = int(output_dim)

        if self.dataset_name == "ngsim":
            vel_mean = [-0.004181504611623526, 5.041936610524995]
            vel_std = [0.1502223350250087, 2.951254134709027]
        elif self.dataset_name == "highd":
            vel_mean = [0.004845835373614644, 17.01558226555126]
            vel_std = [0.10621210903901461, 4.838376260255577]
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'. Supported: ngsim, highd")

        self.register_buffer("vel_mean", torch.tensor(vel_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("vel_std", torch.tensor(vel_std, dtype=torch.float32), persistent=False)

    def buildTargetVelNorm(self, hist, future, device):
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        shifted = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        vel_phys = future_phys - shifted
        std_vel = self.vel_std.view(1, 1, self.output_dim).to(device)
        mean_vel = self.vel_mean.view(1, 1, self.output_dim).to(device)
        target_vel_norm = torch.clamp((vel_phys - mean_vel) / std_vel, -5.0, 5.0)
        return anchor_phys, future_phys, target_vel_norm

    def decodeVelocityToTrajectory(self, pred_vel_norm, anchor_phys):
        shape_prefix = [1] * (pred_vel_norm.dim() - 1)
        std_vel = self.vel_std.view(*shape_prefix, self.output_dim).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(*shape_prefix, self.output_dim).to(pred_vel_norm.device)
        vel_phys = pred_vel_norm[..., :self.output_dim] * std_vel + mean_vel
        pos_phys = torch.cumsum(vel_phys, dim=-2) + anchor_phys[..., :self.output_dim]
        return vel_phys, pos_phys

    def decodeAnchorToPosition(self, anchor_vel_norm, anchor_phys):
        while anchor_phys.dim() < anchor_vel_norm.dim():
            anchor_phys = anchor_phys.unsqueeze(1)
        return self.decodeVelocityToTrajectory(anchor_vel_norm, anchor_phys)

    @staticmethod
    def flattenStructuredModes(tensor):
        if tensor is None:
            return None
        return tensor.view(tensor.size(0), -1, *tensor.shape[4:])

    @staticmethod
    def selectJointGroup(tensor, joint_idx):
        batch_idx = torch.arange(tensor.size(0), device=tensor.device)
        return tensor[batch_idx, joint_idx]


class FutureDiffusionRefiner(nn.Module):
    def __init__(self, args, hidden_dim, input_dim, output_dim, mode_dim, future_steps):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mode_dim = int(mode_dim)
        self.future_steps = int(future_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.x0_clip = float(args.x0_clip) if float(args.x0_clip) > 0 else None
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)

        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.mode_condition_proj = nn.Linear(self.mode_dim, self.hidden_dim)
        self.score_input_proj = nn.Sequential(
            nn.LayerNorm(self.output_dim * 4),
            nn.Linear(self.output_dim * 4, self.hidden_dim),
            nn.GELU(),
        )
        self.score_gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(self.mode_dim + self.hidden_dim * 2 + self.output_dim),
            nn.Linear(self.mode_dim + self.hidden_dim * 2 + self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, int(args.time_embedding_size_fut))
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        dit_block = dit.DiTBlock(
            self.hidden_dim,
            int(args.heads_fut),
            float(args.dropout_fut),
            int(args.mlp_ratio_fut),
        )
        final_layer = dit.FinalLayer(self.hidden_dim, self.future_steps, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=int(args.depth_fut), model_type="score")

    @staticmethod
    def repeatContext(ctx, repeats):
        return {key: value.repeat_interleave(repeats, dim=0) for key, value in ctx.items()}

    def buildScoreFeatures(self, mode_tokens, pred_vel_norm, pred_pos_phys, anchor_pos_phys, detach_inputs=True):
        if detach_inputs:
            pred_vel_norm = pred_vel_norm.detach()
            pred_pos_phys = pred_pos_phys.detach()
            anchor_pos_phys = anchor_pos_phys.detach()

        traj_delta = pred_pos_phys - anchor_pos_phys
        seq_features = torch.cat([pred_vel_norm, pred_pos_phys, anchor_pos_phys, traj_delta], dim=-1)
        flat_seq = seq_features.reshape(-1, seq_features.size(-2), seq_features.size(-1))
        seq_embed = self.score_input_proj(flat_seq)
        seq_out, seq_hidden = self.score_gru(seq_embed)
        seq_hidden = seq_hidden[-1]
        seq_mean = seq_out.mean(dim=1)
        end_pos = pred_pos_phys[..., -1, :self.output_dim].reshape(-1, self.output_dim)
        mode_flat = mode_tokens.reshape(-1, mode_tokens.size(-1))
        score_features = torch.cat([mode_flat, seq_hidden, seq_mean, end_pos], dim=-1)
        return score_features.view(*mode_tokens.shape[:-1], score_features.size(-1))

    def scoreCandidates(self, mode_tokens, pred_vel_norm, pred_pos_phys, anchor_pos_phys, detach_inputs=True):
        score_features = self.buildScoreFeatures(
            mode_tokens,
            pred_vel_norm,
            pred_pos_phys,
            anchor_pos_phys,
            detach_inputs=detach_inputs,
        )
        return self.score_head(score_features).squeeze(-1)

    def predictNoise(self, x_t, timesteps, ctx, mode_token):
        t_emb = self.timestep_embedder(timesteps) + self.mode_condition_proj(mode_token)
        x_emb = self.input_embedding(x_t) + self.pos_embedding(x_t)
        return self.dit(x=x_emb, t_cond=t_emb, cross=ctx["cross_tokens"])

    def predictResidualX0(self, x_t, pred_eps, timesteps):
        alpha_cumprod = self.diffusion_scheduler.alphas_cumprod.to(x_t.device)[timesteps].view(-1, 1, 1)
        pred_r0 = (x_t - torch.sqrt(1.0 - alpha_cumprod) * pred_eps) / (torch.sqrt(alpha_cumprod) + 1e-6)
        if self.x0_clip is not None:
            pred_r0 = torch.clamp(pred_r0, -self.x0_clip, self.x0_clip)
        return pred_r0

    def sampleFromXt(self, x_t, ctx, mode_token, infer_scheduler):
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

    def runDiffusionForModes(self, ctx, mode_tokens, anchor_vel_norm, anchor_phys, device, motion_codec):
        bsz, num_modes, _, _ = anchor_vel_norm.shape
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        x_t = torch.randn((bsz * num_modes, self.future_steps, self.input_dim), device=device)
        pred_residual_norm = self.sampleFromXt(
            x_t,
            self.repeatContext(ctx, num_modes),
            mode_tokens.reshape(bsz * num_modes, -1),
            infer_scheduler,
        ).view(bsz, num_modes, self.future_steps, self.output_dim)
        pred_vel_norm = anchor_vel_norm + pred_residual_norm
        _, anchor_pos_phys = motion_codec.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)
        _, pred_pos_phys = motion_codec.decodeAnchorToPosition(pred_vel_norm, anchor_phys)
        score_logits = self.scoreCandidates(
            mode_tokens,
            pred_vel_norm,
            pred_pos_phys,
            anchor_pos_phys,
            detach_inputs=True,
        )
        return {
            "pred_residual_norm": pred_residual_norm,
            "pred_vel_norm": pred_vel_norm,
            "pred_pos_phys": pred_pos_phys,
            "anchor_pos_phys": anchor_pos_phys,
            "score_logits": score_logits,
        }


__all__ = [
    "AnchorDecoder",
    "IntentConditionedModePrior",
    "FutureMotionCodec",
    "FutureDiffusionRefiner",
]
