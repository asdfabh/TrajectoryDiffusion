from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from method_diffusion.utils.fut_utils import gather_last_by_valid, compute_per_mode_distance


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


def get_intent_statistics(dataset_name):
    dataset_key = str(dataset_name).strip().lower()
    if dataset_key == "ngsim":
        return {
            "lat_freq": [0.954152, 0.036162, 0.009686],
            "lon_freq": [0.633374, 0.172099, 0.194527],
            "joint_freq": [0.602786, 0.165885, 0.185482, 0.023751, 0.005372, 0.007039, 0.006838, 0.000842, 0.002006],
        }
    if dataset_key == "highd":
        return {
            "lat_freq": [0.942968, 0.026543, 0.030489],
            "lon_freq": [0.975553, 0.011313, 0.013134],
            "joint_freq": [0.918904, 0.011210, 0.012854, 0.026323, 0.000015, 0.000205, 0.030326, 0.000089, 0.000074],
        }
    raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: ngsim, highd")


@dataclass(frozen=True)
class FutureLossWeights:
    intent: float = 1.0
    joint: float = 1.0
    mode: float = 0.5
    anchor: float = 1.0
    div: float = 0.05
    eps: float = 1.0
    rank: float = 0.5
    x0: float = 0.1
    end: float = 0.1


@dataclass(frozen=True)
class FutureLossHyperParams:
    intent_label_smoothing: float = 0.05
    submode_temperature: float = 1.0
    anchor_div_margin: float = 2.0
    rank_temperature: float = 1.0
    rank_gap_threshold: float = 0.05
    submode_floor_weight: float = 0.2


class FutureLossComputer(nn.Module):
    DEFAULT_WEIGHTS = FutureLossWeights()
    DEFAULT_HYPER_PARAMS = FutureLossHyperParams()

    def __init__(self, dataset_name, num_joint_classes, num_submodes, future_steps, output_dim, num_train_timesteps):
        super().__init__()
        stats = get_intent_statistics(dataset_name)
        self.num_joint_classes = int(num_joint_classes)
        self.num_submodes = int(num_submodes)
        self.future_steps = int(future_steps)
        self.output_dim = int(output_dim)
        self.loss_weights = self.DEFAULT_WEIGHTS
        self.intent_label_smoothing = max(0.0, min(0.2, float(self.DEFAULT_HYPER_PARAMS.intent_label_smoothing)))
        self.submode_temperature = max(0.1, float(self.DEFAULT_HYPER_PARAMS.submode_temperature))
        self.anchor_div_margin = max(0.0, float(self.DEFAULT_HYPER_PARAMS.anchor_div_margin))
        self.rank_temperature = max(0.1, float(self.DEFAULT_HYPER_PARAMS.rank_temperature))
        self.rank_gap_threshold = max(0.0, float(self.DEFAULT_HYPER_PARAMS.rank_gap_threshold))
        self.submode_floor_weight = max(0.0, min(1.0, float(self.DEFAULT_HYPER_PARAMS.submode_floor_weight)))
        self.num_train_timesteps = int(num_train_timesteps)

        self.register_buffer("lat_class_weights", build_balanced_weights(stats["lat_freq"]), persistent=False)
        self.register_buffer("lon_class_weights", build_balanced_weights(stats["lon_freq"]), persistent=False)
        self.register_buffer("joint_class_weights", build_balanced_weights(stats["joint_freq"]), persistent=False)

    def combineTrainingLosses(self, intent_parts, anchor_parts, ref_parts):
        return (
            self.loss_weights.intent * (intent_parts["loss_intent_lat"] + intent_parts["loss_intent_lon"])
            + self.loss_weights.joint * intent_parts["loss_intent_joint"]
            + self.loss_weights.mode * anchor_parts["loss_mode"]
            + self.loss_weights.anchor * anchor_parts["loss_anchor"]
            + self.loss_weights.div * anchor_parts["loss_div"]
            + self.loss_weights.eps * ref_parts["loss_eps"]
            + self.loss_weights.rank * ref_parts["loss_rank"]
            + self.loss_weights.x0 * ref_parts["loss_x0"]
            + self.loss_weights.end * ref_parts["loss_end"]
        )

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

        return {
            "loss_intent_lat": weighted_soft_cross_entropy(lat_logits, lat_targets, self.lat_class_weights),
            "loss_intent_lon": weighted_soft_cross_entropy(lon_logits, lon_targets, self.lon_class_weights),
            "loss_intent_joint": weighted_soft_cross_entropy(joint_logits, joint_targets, self.joint_class_weights),
            "lat_probs": lat_probs,
            "lon_probs": lon_probs,
            "joint_probs": joint_probs,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "joint_idx": joint_idx,
        }

    def computeAnchorGroupLosses(self, group_submode_logits, group_anchor_pos_phys, future_phys, valid_mask):
        ade_group, fde_group = compute_per_mode_distance(group_anchor_pos_phys, future_phys, valid_mask)
        dist_group = ade_group + 0.5 * fde_group
        assign_probs = torch.softmax(-dist_group.detach() / self.submode_temperature, dim=1)
        loss_mode = weighted_soft_cross_entropy(group_submode_logits, assign_probs, None)
        loss_anchor = (assign_probs * dist_group).sum(dim=1).mean()

        if self.num_submodes > 1:
            checkpoints = sorted({min(self.future_steps - 1, idx) for idx in [4, 9, 14, 19, self.future_steps - 1]})
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

    def computeGroupRefinementLosses(
        self,
        ctx,
        group_mode_tokens,
        group_anchor_vel_norm,
        anchor_phys,
        future_phys,
        target_vel_norm,
        valid_mask,
        assign_probs,
        device,
        refiner,
        motion_codec,
    ):
        bsz, group_size, _, _ = group_anchor_vel_norm.shape
        residual_target = target_vel_norm.unsqueeze(1) - group_anchor_vel_norm
        noise = torch.randn_like(residual_target)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz, group_size), device=device).long()

        flat_residual = residual_target.reshape(bsz * group_size, self.future_steps, self.output_dim)
        flat_noise = noise.reshape_as(flat_residual)
        flat_timesteps = timesteps.reshape(-1)
        x_t = refiner.diffusion_scheduler.add_noise(flat_residual, flat_noise, flat_timesteps)
        pred_eps = refiner.predictNoise(
            x_t,
            flat_timesteps,
            refiner.repeatContext(ctx, group_size),
            group_mode_tokens.reshape(bsz * group_size, -1),
        ).view(bsz, group_size, self.future_steps, self.output_dim)

        valid = valid_mask.unsqueeze(1).unsqueeze(-1)
        point_denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1e-6) * pred_eps.size(-1)
        loss_eps_per_mode = (((pred_eps - noise) ** 2) * valid).sum(dim=(2, 3)) / point_denom

        pred_r0 = refiner.predictResidualX0(
            x_t,
            pred_eps.reshape(bsz * group_size, self.future_steps, self.output_dim),
            flat_timesteps,
        ).view(bsz, group_size, self.future_steps, self.output_dim)
        pred_vel_norm = group_anchor_vel_norm + pred_r0
        _, pred_pos_phys = motion_codec.decodeAnchorToPosition(pred_vel_norm, anchor_phys)

        target_pos = future_phys.unsqueeze(1).expand(-1, group_size, -1, -1)
        loss_x0_per_mode = (
            F.smooth_l1_loss(pred_pos_phys, target_pos, reduction="none") * valid
        ).sum(dim=(2, 3)) / point_denom

        flat_pred_pos = pred_pos_phys.reshape(bsz * group_size, self.future_steps, self.output_dim)
        flat_target_pos = target_pos.reshape(bsz * group_size, self.future_steps, self.output_dim)
        flat_valid_mask = valid_mask.unsqueeze(1).expand(-1, group_size, -1).reshape(bsz * group_size, self.future_steps)
        pred_end, has_valid = gather_last_by_valid(flat_pred_pos, flat_valid_mask)
        target_end, _ = gather_last_by_valid(flat_target_pos, flat_valid_mask)
        loss_end_per_mode = (torch.abs(pred_end - target_end).sum(dim=-1) * has_valid.float()).view(bsz, group_size)

        final_ade, final_fde = compute_per_mode_distance(pred_pos_phys, future_phys, valid_mask)
        final_dist = final_ade + 0.5 * final_fde
        _, anchor_pos_phys = motion_codec.decodeAnchorToPosition(group_anchor_vel_norm, anchor_phys)
        score_logits = refiner.scoreCandidates(
            group_mode_tokens,
            pred_vel_norm,
            pred_pos_phys,
            anchor_pos_phys,
            detach_inputs=True,
        )
        loss_rank = self.computePairwiseRankLoss(score_logits, final_dist)

        weight_floor = self.submode_floor_weight / float(group_size)
        train_weights = weight_floor + (1.0 - self.submode_floor_weight) * assign_probs.detach()
        train_weights = train_weights / train_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        best_idx = torch.argmin(final_dist, dim=1)
        routed_idx = torch.argmax(score_logits, dim=1)
        submode_hit = (routed_idx == best_idx).float().mean().detach()

        if group_size > 1:
            first_pair_gap = torch.abs(final_dist[:, 0] - final_dist[:, 1]).mean().detach()
            final_pair_dist = torch.norm(
                pred_pos_phys[:, 0, :, :self.output_dim] - pred_pos_phys[:, 1, :, :self.output_dim],
                dim=-1,
            )
            final_pair_dist = (
                (final_pair_dist * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)
            ).mean().detach()
        else:
            first_pair_gap = pred_pos_phys.new_tensor(0.0)
            final_pair_dist = pred_pos_phys.new_tensor(0.0)

        return {
            "loss_eps": (train_weights * loss_eps_per_mode).sum(dim=1).mean(),
            "loss_x0": (train_weights * loss_x0_per_mode).sum(dim=1).mean(),
            "loss_end": (train_weights * loss_end_per_mode).sum(dim=1).mean(),
            "loss_rank": loss_rank,
            "score_logits": score_logits,
            "pred_vel_norm": pred_vel_norm,
            "pred_pos_phys": pred_pos_phys,
            "final_ade": final_ade,
            "final_fde": final_fde,
            "best_idx": best_idx,
            "routed_idx": routed_idx,
            "submode_hit": submode_hit,
            "submode_gap": first_pair_gap,
            "final_pair_dist": final_pair_dist,
        }

    def computePairwiseRankLoss(self, score_logits, final_dist):
        group_size = score_logits.size(1)
        if group_size <= 1:
            return score_logits.new_tensor(0.0)

        pair_losses = []
        pair_weights = []
        for i in range(group_size):
            for j in range(i + 1, group_size):
                dist_i = final_dist[:, i]
                dist_j = final_dist[:, j]
                gap = torch.abs(dist_i - dist_j).detach()
                better_is_i = dist_i < dist_j
                better_score = torch.where(better_is_i, score_logits[:, i], score_logits[:, j])
                worse_score = torch.where(better_is_i, score_logits[:, j], score_logits[:, i])
                pair_loss = F.softplus(-(better_score - worse_score) / self.rank_temperature)
                active = gap >= self.rank_gap_threshold
                if active.any():
                    pair_losses.append(pair_loss[active])
                    pair_weights.append(gap[active])

        if not pair_losses:
            return score_logits.new_tensor(0.0)

        all_losses = torch.cat(pair_losses, dim=0)
        all_weights = torch.cat(pair_weights, dim=0).clamp(min=1e-6)
        return (all_losses * all_weights).sum() / all_weights.sum()


__all__ = ["FutureLossComputer"]
