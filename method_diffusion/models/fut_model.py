import torch
import torch.nn.functional as F
from torch import nn

from method_diffusion.models.Intent_anchor import (
    IntentConditionedModePrior,
    FutureMotionCodec,
    FutureDiffusionRefiner,
)
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.models.fut_loss import FutureLossComputer
from method_diffusion.utils.fut_utils import (
    to_valid_mask,
    gather_by_index,
    compute_ade_fde,
    compute_per_mode_distance,
)


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()

        self.hidden_dim = int(args.hidden_dim_fut)
        self.input_dim = int(args.input_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.T = int(args.T_f)
        self.mode_dim = int(args.mode_dim)
        self.num_lat_classes = int(getattr(args, "num_lat_classes", 3))
        self.num_lon_classes = int(getattr(args, "num_lon_classes", 3))
        self.num_joint_classes = self.num_lat_classes * self.num_lon_classes
        self.num_submodes = int(getattr(args, "num_submodes", 2))
        self.num_modes = self.num_joint_classes * self.num_submodes

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
        self.motion_codec = FutureMotionCodec(self.dataset_name, output_dim=self.output_dim)
        self.refiner = FutureDiffusionRefiner(
            args=args,
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            mode_dim=self.mode_dim,
            future_steps=self.T,
        )
        self.loss_helper = FutureLossComputer(
            dataset_name=self.dataset_name,
            num_joint_classes=self.num_joint_classes,
            num_submodes=self.num_submodes,
            future_steps=self.T,
            output_dim=self.output_dim,
            num_train_timesteps=args.num_train_timesteps_fut,
        )

        # 兼容旧代码的只读别名。
        self.vel_mean = self.motion_codec.vel_mean
        self.vel_std = self.motion_codec.vel_std
        self.diffusion_scheduler = self.refiner.diffusion_scheduler

    def buildTargetVelNorm(self, hist, future, device):
        return self.motion_codec.buildTargetVelNorm(hist, future, device)

    def decodeVelocityToTrajectory(self, pred_vel_norm, anchor_phys):
        return self.motion_codec.decodeVelocityToTrajectory(pred_vel_norm, anchor_phys)

    def decodeAnchorToPosition(self, anchor_vel_norm, anchor_phys):
        return self.motion_codec.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)

    def flattenStructuredModes(self, tensor):
        return self.motion_codec.flattenStructuredModes(tensor)

    def selectJointGroup(self, tensor, joint_idx):
        return self.motion_codec.selectJointGroup(tensor, joint_idx)

    def runDiffusionForModes(self, ctx, mode_tokens, anchor_vel_norm, anchor_phys, device):
        return self.refiner.runDiffusionForModes(ctx, mode_tokens, anchor_vel_norm, anchor_phys, device, self.motion_codec)


    def getAllModePredictions(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, select_topk=None):
        bsz = future.size(0)
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, _ = self.motion_codec.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])

        mode_logits = prior["mode_log_probs"].view(bsz, self.num_modes)
        mode_tokens = self.motion_codec.flattenStructuredModes(prior["mode_tokens"])
        anchor_vel_norm = self.motion_codec.flattenStructuredModes(prior["anchor_vel_norm"])
        mode_indices = torch.arange(self.num_modes, device=device).view(1, self.num_modes).expand(bsz, -1)

        if select_topk is not None and select_topk < self.num_modes:
            topk = max(1, int(select_topk))
            topk_idx = torch.topk(mode_logits, k=topk, dim=1).indices
            mode_logits = mode_logits.gather(1, topk_idx)
            mode_indices = mode_indices.gather(1, topk_idx)
            mode_tokens = torch.gather(mode_tokens, 1, topk_idx.unsqueeze(-1).expand(-1, -1, mode_tokens.size(-1)))
            anchor_vel_norm = torch.gather(
                anchor_vel_norm,
                1,
                topk_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, anchor_vel_norm.size(2), anchor_vel_norm.size(3)),
            )

        denoise_outputs = self.runDiffusionForModes(ctx, mode_tokens, anchor_vel_norm, anchor_phys, device)
        _, anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(anchor_vel_norm, anchor_phys)

        all_preds = future.clone().unsqueeze(1).repeat(1, mode_logits.size(1), 1, 1)
        all_preds[..., :2] = denoise_outputs["pred_pos_phys"]
        ade_per_mode, fde_per_mode = compute_per_mode_distance(denoise_outputs["pred_pos_phys"], future_phys, valid_mask)
        best_mode_pos = torch.argmin(ade_per_mode, dim=1)
        best_mode_idx = mode_indices.gather(1, best_mode_pos.unsqueeze(1)).squeeze(1)

        return {
            "mode_probs": torch.softmax(mode_logits, dim=1),
            "anchor_pos_phys": anchor_pos_phys,
            "all_pred_phys": all_preds,
            "ade_per_mode": ade_per_mode,
            "fde_per_mode": fde_per_mode,
            "mode_indices": mode_indices,
            "best_mode_pos": best_mode_pos,
            "best_mode_idx": best_mode_idx,
            "mode_nll": F.cross_entropy(mode_logits, best_mode_pos).detach(),
        }

    # ── 训练 / 评估 主流程 ───────────────────────────────────────────────────

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                     return_components=False, lat_targets=None, lon_targets=None, joint_targets=None):
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, target_vel_norm = self.motion_codec.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.loss_helper.buildIntentLosses(prior, lat_targets, lon_targets, joint_targets)

        all_anchor_vel_norm = self.motion_codec.flattenStructuredModes(prior["anchor_vel_norm"])
        _, all_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(all_anchor_vel_norm, anchor_phys)
        global_metrics = self.loss_helper.computeGlobalMetrics(
            prior["mode_log_probs"].view(hist.size(0), self.num_modes),
            all_anchor_pos_phys,
            future_phys,
            valid_mask,
        )

        group_anchor_vel_norm = self.motion_codec.selectJointGroup(prior["joint_anchor_vel_norm"], intent_parts["joint_idx"])
        group_mode_tokens = self.motion_codec.selectJointGroup(prior["joint_mode_tokens"], intent_parts["joint_idx"])
        group_submode_logits = self.motion_codec.selectJointGroup(prior["joint_submode_logits"], intent_parts["joint_idx"])
        _, group_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(group_anchor_vel_norm, anchor_phys)

        anchor_parts = self.loss_helper.computeAnchorGroupLosses(
            group_submode_logits,
            group_anchor_pos_phys,
            future_phys,
            valid_mask,
        )
        ref_parts = self.loss_helper.computeGroupRefinementLosses(
            ctx=ctx,
            group_mode_tokens=group_mode_tokens,
            group_anchor_vel_norm=group_anchor_vel_norm,
            anchor_phys=anchor_phys,
            future_phys=future_phys,
            target_vel_norm=target_vel_norm,
            valid_mask=valid_mask,
            assign_probs=anchor_parts["assign_probs"],
            device=device,
            refiner=self.refiner,
            motion_codec=self.motion_codec,
        )

        loss = self.loss_helper.combineTrainingLosses(intent_parts, anchor_parts, ref_parts)

        if return_components:
            return loss, {
                "loss_total": loss.detach(),
                "losses": {
                    "intent_lat": intent_parts["loss_intent_lat"].detach(),
                    "intent_lon": intent_parts["loss_intent_lon"].detach(),
                    "intent_joint": intent_parts["loss_intent_joint"].detach(),
                    "mode": anchor_parts["loss_mode"].detach(),
                    "anchor": anchor_parts["loss_anchor"].detach(),
                    "div": anchor_parts["loss_div"].detach(),
                    "eps": ref_parts["loss_eps"].detach(),
                    "rank": ref_parts["loss_rank"].detach(),
                    "x0": ref_parts["loss_x0"].detach(),
                    "end": ref_parts["loss_end"].detach(),
                },
                "metrics": {
                    "coarse_top1_ade": global_metrics["top1_ade"],
                    "coarse_top1_fde": global_metrics["top1_fde"],
                    "coarse_minade": global_metrics["minade_m"],
                    "coarse_minfde": global_metrics["minfde_m"],
                    "coarse_mode_nll": global_metrics["mode_nll"],
                    "submode_hit": ref_parts["submode_hit"],
                    "submode_gap": ref_parts["submode_gap"],
                    "final_pair_dist": ref_parts["final_pair_dist"],
                    "lat_acc": (torch.argmax(intent_parts["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                    "lon_acc": (torch.argmax(intent_parts["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                    "joint_acc": (torch.argmax(intent_parts["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
                },
            }
        return loss

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
                    return_aux=False, lat_targets=None, lon_targets=None, joint_targets=None):
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, _ = self.motion_codec.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.loss_helper.buildIntentLosses(prior, lat_targets, lon_targets, joint_targets)

        pred_joint_idx = torch.argmax(prior["joint_probs"], dim=1)
        all_anchor_vel_norm = self.motion_codec.flattenStructuredModes(prior["anchor_vel_norm"])
        _, all_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(all_anchor_vel_norm, anchor_phys)
        pred_group_tokens = self.motion_codec.selectJointGroup(prior["joint_mode_tokens"], pred_joint_idx)
        pred_group_anchors = self.motion_codec.selectJointGroup(prior["joint_anchor_vel_norm"], pred_joint_idx)
        pred_group_out = self.runDiffusionForModes(ctx, pred_group_tokens, pred_group_anchors, anchor_phys, device)
        pred_group_preds = future.clone().unsqueeze(1).repeat(1, self.num_submodes, 1, 1)
        pred_group_preds[..., :2] = pred_group_out["pred_pos_phys"]
        pred_group_ade, pred_group_fde = compute_per_mode_distance(pred_group_out["pred_pos_phys"], future_phys, valid_mask)
        routed_sub_idx = torch.argmax(pred_group_out["score_logits"], dim=1)
        best_pred_sub_idx = torch.argmin(pred_group_ade + 0.5 * pred_group_fde, dim=1)
        selected_mode_idx = pred_joint_idx * self.num_submodes + routed_sub_idx
        top1_pred = gather_by_index(pred_group_preds, routed_sub_idx)
        top1_ade, top1_fde = compute_ade_fde(top1_pred, future, valid_mask)
        submode_hit = (routed_sub_idx == best_pred_sub_idx).float().mean().detach()
        if self.num_submodes > 1:
            submode_gap = torch.abs(
                (pred_group_ade + 0.5 * pred_group_fde)[:, 0] - (pred_group_ade + 0.5 * pred_group_fde)[:, 1]
            ).mean().detach()
            final_pair_dist = torch.norm(
                pred_group_out["pred_pos_phys"][:, 0, :, :self.output_dim]
                - pred_group_out["pred_pos_phys"][:, 1, :, :self.output_dim],
                dim=-1,
            )
            final_pair_dist = (
                (final_pair_dist * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-6)
            ).mean().detach()
        else:
            submode_gap = top1_ade.new_tensor(0.0)
            final_pair_dist = top1_ade.new_tensor(0.0)

        aux = {
            "predictions": {
                "group": pred_group_preds,
                "top1": top1_pred,
            },
            "metrics": {
                "top1_ade": top1_ade.detach(),
                "top1_fde": top1_fde.detach(),
                "submode_hit": submode_hit,
                "submode_gap": submode_gap,
                "final_pair_dist": final_pair_dist,
                "lat_acc": (torch.argmax(prior["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                "lon_acc": (torch.argmax(prior["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                "joint_acc": (torch.argmax(prior["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
            },
            "routing": {
                "mode_probs": prior["mode_probs"],
                "lat_probs": prior["lat_probs"],
                "lon_probs": prior["lon_probs"],
                "joint_probs": prior["joint_probs"],
                "selected_mode_idx": selected_mode_idx,
                "pred_joint_idx": pred_joint_idx,
                "gt_joint_idx": intent_parts["joint_idx"],
                "routed_sub_idx": routed_sub_idx,
                "best_pred_joint_sub_idx": best_pred_sub_idx,
                "score_logits": pred_group_out["score_logits"],
                "group_ade": pred_group_ade,
                "group_fde": pred_group_fde,
            },
        }
        if return_aux:
            return top1_pred, aux
        return top1_pred

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
