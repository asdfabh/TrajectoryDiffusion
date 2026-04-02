import torch
import torch.nn.functional as F
from torch import nn

from method_diffusion.models.Intent_anchor import (
    IntentConditionedModePrior,
    FutureMotionCodec,
    FutureDiffusionRefiner,
)
from method_diffusion.models.fut_loss import FutureLossComputer
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.fut_utils import (
    compute_ade_fde,
    compute_per_mode_distance,
    gather_by_index,
    to_valid_mask,
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
        self.num_anchor_per_joint = int(getattr(args, "num_anchor_per_joint", getattr(args, "num_submodes", 2)))
        self.num_submodes = self.num_anchor_per_joint
        self.num_modes = self.num_joint_classes * self.num_anchor_per_joint
        self.topk_intents = max(1, min(int(getattr(args, "topk_intents", 3)), self.num_joint_classes))
        self.topk_refine = self.topk_intents * self.num_anchor_per_joint
        self.lambda_fde_route = max(0.0, float(getattr(args, "lambda_fde_route", 0.5)))
        self.route_beta = float(getattr(args, "route_beta", 1.0))

        self.hist_encoder = HistEncoder(args)
        self.mode_prior = IntentConditionedModePrior(
            hidden_dim=self.hidden_dim,
            mode_dim=self.mode_dim,
            num_lat_classes=self.num_lat_classes,
            num_lon_classes=self.num_lon_classes,
            num_anchor_per_joint=self.num_anchor_per_joint,
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
            num_anchor_per_joint=self.num_anchor_per_joint,
            future_steps=self.T,
            output_dim=self.output_dim,
            num_train_timesteps=args.num_train_timesteps_fut,
            lambda_fde_route=self.lambda_fde_route,
            tau_anchor_teacher=float(getattr(args, "tau_anchor_teacher", 1.0)),
            tau_score_teacher=float(getattr(args, "tau_score_teacher", 1.0)),
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

    def gatherTopIntents(self, tensor, top_intent_idx):
        if tensor is None:
            return None
        expand_shape = [top_intent_idx.size(0), top_intent_idx.size(1)] + [1] * (tensor.dim() - 2)
        gather_idx = top_intent_idx.view(*expand_shape).expand(
            top_intent_idx.size(0),
            top_intent_idx.size(1),
            *tensor.shape[2:],
        )
        return tensor.gather(1, gather_idx)

    @staticmethod
    def flattenIntentAnchors(tensor):
        if tensor is None:
            return None
        return tensor.reshape(tensor.size(0), -1, *tensor.shape[3:])

    def buildTopIntentCandidates(
        self,
        top_intent_idx,
        joint_mode_tokens,
        joint_anchor_vel_norm,
        joint_anchor_prior_logits,
        joint_q_anchor=None,
    ):
        selected_mode_tokens = self.gatherTopIntents(joint_mode_tokens, top_intent_idx)
        selected_anchor_vel_norm = self.gatherTopIntents(joint_anchor_vel_norm, top_intent_idx)
        selected_anchor_prior_logits = self.gatherTopIntents(joint_anchor_prior_logits, top_intent_idx)
        selected_q_anchor = self.gatherTopIntents(joint_q_anchor, top_intent_idx) if joint_q_anchor is not None else None

        local_anchor_idx = torch.arange(
            self.num_anchor_per_joint,
            device=top_intent_idx.device,
        ).view(1, 1, self.num_anchor_per_joint)
        global_mode_idx = top_intent_idx.unsqueeze(-1) * self.num_anchor_per_joint + local_anchor_idx

        return {
            "top_intent_idx": top_intent_idx,
            "selected_mode_tokens": selected_mode_tokens,
            "selected_anchor_vel_norm": selected_anchor_vel_norm,
            "selected_anchor_prior_logits": selected_anchor_prior_logits,
            "selected_q_anchor": selected_q_anchor,
            "selected_mode_idx": global_mode_idx,
            "flat_mode_tokens": self.flattenIntentAnchors(selected_mode_tokens),
            "flat_anchor_vel_norm": self.flattenIntentAnchors(selected_anchor_vel_norm),
            "flat_anchor_prior_logits": self.flattenIntentAnchors(selected_anchor_prior_logits),
            "flat_q_anchor": self.flattenIntentAnchors(selected_q_anchor),
            "flat_mode_idx": self.flattenIntentAnchors(global_mode_idx),
        }

    def getIntentScores(self, prior):
        return torch.logsumexp(prior["joint_anchor_prior_logits"], dim=-1)

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
        dist_per_mode = ade_per_mode + self.lambda_fde_route * fde_per_mode
        best_mode_pos = torch.argmin(dist_per_mode, dim=1)
        best_mode_idx = mode_indices.gather(1, best_mode_pos.unsqueeze(1)).squeeze(1)

        return {
            "mode_probs": torch.softmax(mode_logits, dim=1),
            "anchor_pos_phys": anchor_pos_phys,
            "all_pred_phys": all_preds,
            "ade_per_mode": ade_per_mode,
            "fde_per_mode": fde_per_mode,
            "dist_per_mode": dist_per_mode,
            "mode_indices": mode_indices,
            "best_mode_pos": best_mode_pos,
            "best_mode_idx": best_mode_idx,
            "mode_nll": F.cross_entropy(mode_logits, best_mode_pos).detach(),
        }

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
        lat_targets=None,
        lon_targets=None,
        joint_targets=None,
    ):
        bsz = hist.size(0)
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, target_vel_norm = self.motion_codec.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.loss_helper.buildSemanticLosses(prior, lat_targets, lon_targets, joint_targets)

        all_anchor_vel_norm = self.motion_codec.flattenStructuredModes(prior["anchor_vel_norm"])
        _, all_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(all_anchor_vel_norm, anchor_phys)
        global_metrics = self.loss_helper.computeGlobalMetrics(
            prior["mode_log_probs"].view(bsz, self.num_modes),
            all_anchor_pos_phys,
            future_phys,
            valid_mask,
        )

        joint_mode_tokens = prior["joint_mode_tokens"]
        joint_anchor_vel_norm = prior["joint_anchor_vel_norm"]
        joint_anchor_prior_logits = prior["joint_anchor_prior_logits"]
        _, joint_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(joint_anchor_vel_norm, anchor_phys)

        anchor_parts = self.loss_helper.computeGlobalAnchorLosses(
            joint_anchor_prior_logits,
            joint_anchor_pos_phys,
            future_phys,
            valid_mask,
        )
        q_anchor_joint = anchor_parts["q_anchor"].view(bsz, self.num_joint_classes, self.num_anchor_per_joint)
        top_intent_idx = torch.topk(anchor_parts["q_intent"], k=self.topk_intents, dim=1).indices
        selected = self.buildTopIntentCandidates(
            top_intent_idx=top_intent_idx,
            joint_mode_tokens=joint_mode_tokens,
            joint_anchor_vel_norm=joint_anchor_vel_norm,
            joint_anchor_prior_logits=joint_anchor_prior_logits,
            joint_q_anchor=q_anchor_joint,
        )

        ref_parts = self.loss_helper.computeTopKRefinementLosses(
            ctx=ctx,
            selected_mode_tokens=selected["flat_mode_tokens"],
            selected_anchor_vel_norm=selected["flat_anchor_vel_norm"],
            selected_anchor_prior_logits=selected["flat_anchor_prior_logits"],
            selected_q_anchor=selected["flat_q_anchor"],
            anchor_phys=anchor_phys,
            future_phys=future_phys,
            target_vel_norm=target_vel_norm,
            valid_mask=valid_mask,
            device=device,
            refiner=self.refiner,
            motion_codec=self.motion_codec,
            route_beta=self.route_beta,
        )

        loss = self.loss_helper.combineTrainingLosses(intent_parts, anchor_parts, ref_parts)
        best_intent_acc = (
            torch.argmax(self.getIntentScores(prior), dim=1) == anchor_parts["best_intent_idx"]
        ).float().mean().detach()
        intent_topk_hit = (
            top_intent_idx == anchor_parts["best_intent_idx"].unsqueeze(1)
        ).any(dim=1).float().mean().detach()

        if return_components:
            return loss, {
                "loss_total": loss.detach(),
                "losses": {
                    "intent_lat": intent_parts["loss_intent_lat"].detach(),
                    "intent_lon": intent_parts["loss_intent_lon"].detach(),
                    "mode": anchor_parts["loss_mode"].detach(),
                    "anchor": anchor_parts["loss_anchor"].detach(),
                    "div": anchor_parts["loss_div"].detach(),
                    "eps": ref_parts["loss_eps"].detach(),
                    "score": ref_parts["loss_score"].detach(),
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
                    "intent_topk_hit": intent_topk_hit,
                    "best_intent_acc": best_intent_acc,
                    "route_hit": ref_parts["route_hit"],
                    "route_gap": ref_parts["route_gap"],
                    "lat_acc": (torch.argmax(intent_parts["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                    "lon_acc": (torch.argmax(intent_parts["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                    "joint_acc": (torch.argmax(intent_parts["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
                },
            }
        return loss

    @torch.no_grad()
    def forwardEval(
        self,
        hist,
        hist_nbrs,
        mask,
        temporal_mask,
        future,
        op_mask,
        device,
        return_aux=False,
        lat_targets=None,
        lon_targets=None,
        joint_targets=None,
    ):
        bsz = hist.size(0)
        valid_mask = to_valid_mask(op_mask, device)
        anchor_phys, future_phys, _ = self.motion_codec.buildTargetVelNorm(hist, future, device)
        ctx = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        prior = self.mode_prior(ctx["global_token"])
        intent_parts = self.loss_helper.buildSemanticLosses(prior, lat_targets, lon_targets, joint_targets)

        joint_mode_tokens = prior["joint_mode_tokens"]
        joint_anchor_vel_norm = prior["joint_anchor_vel_norm"]
        joint_anchor_prior_logits = prior["joint_anchor_prior_logits"]
        intent_scores = self.getIntentScores(prior)
        top_intent_idx = torch.topk(intent_scores, k=self.topk_intents, dim=1).indices
        selected = self.buildTopIntentCandidates(
            top_intent_idx=top_intent_idx,
            joint_mode_tokens=joint_mode_tokens,
            joint_anchor_vel_norm=joint_anchor_vel_norm,
            joint_anchor_prior_logits=joint_anchor_prior_logits,
        )

        pred_group_out = self.runDiffusionForModes(
            ctx,
            selected["flat_mode_tokens"],
            selected["flat_anchor_vel_norm"],
            anchor_phys,
            device,
        )
        pred_group_preds = future.clone().unsqueeze(1).repeat(1, self.topk_refine, 1, 1)
        pred_group_preds[..., :2] = pred_group_out["pred_pos_phys"]
        pred_group_ade, pred_group_fde = compute_per_mode_distance(pred_group_out["pred_pos_phys"], future_phys, valid_mask)
        final_dist = pred_group_ade + self.lambda_fde_route * pred_group_fde
        route_logits = selected["flat_anchor_prior_logits"] + self.route_beta * pred_group_out["score_logits"]
        routed_idx = torch.argmax(route_logits, dim=1)
        best_pred_idx = torch.argmin(final_dist, dim=1)
        selected_mode_idx = selected["flat_mode_idx"].gather(1, routed_idx.unsqueeze(1)).squeeze(1)
        top1_pred = gather_by_index(pred_group_preds, routed_idx)
        top1_ade, top1_fde = compute_ade_fde(top1_pred, future, valid_mask)

        _, joint_anchor_pos_phys = self.motion_codec.decodeAnchorToPosition(joint_anchor_vel_norm, anchor_phys)
        anchor_parts = self.loss_helper.computeGlobalAnchorLosses(
            joint_anchor_prior_logits,
            joint_anchor_pos_phys,
            future_phys,
            valid_mask,
        )
        best_intent_acc = (
            torch.argmax(intent_scores, dim=1) == anchor_parts["best_intent_idx"]
        ).float().mean().detach()
        intent_topk_hit = (
            top_intent_idx == anchor_parts["best_intent_idx"].unsqueeze(1)
        ).any(dim=1).float().mean().detach()
        route_hit = (routed_idx == best_pred_idx).float().mean().detach()
        route_gap = (
            final_dist.gather(1, routed_idx.unsqueeze(1)).squeeze(1) - final_dist.min(dim=1).values
        ).mean().detach()

        aux = {
            "predictions": {
                "candidates": pred_group_preds,
                "top1": top1_pred,
            },
            "metrics": {
                "top1_ade": top1_ade.detach(),
                "top1_fde": top1_fde.detach(),
                "intent_topk_hit": intent_topk_hit,
                "best_intent_acc": best_intent_acc,
                "route_hit": route_hit,
                "route_gap": route_gap,
                "lat_acc": (torch.argmax(prior["lat_probs"], dim=1) == intent_parts["lat_idx"]).float().mean().detach(),
                "lon_acc": (torch.argmax(prior["lon_probs"], dim=1) == intent_parts["lon_idx"]).float().mean().detach(),
                "joint_acc": (torch.argmax(prior["joint_probs"], dim=1) == intent_parts["joint_idx"]).float().mean().detach(),
            },
            "routing": {
                "mode_probs": prior["mode_probs"],
                "lat_probs": prior["lat_probs"],
                "lon_probs": prior["lon_probs"],
                "joint_probs": prior["joint_probs"],
                "intent_scores": intent_scores,
                "top_intent_idx": top_intent_idx,
                "selected_mode_idx": selected_mode_idx,
                "selected_global_mode_idx": selected["flat_mode_idx"],
                "best_intent_idx": anchor_parts["best_intent_idx"],
                "routed_idx": routed_idx,
                "best_pred_idx": best_pred_idx,
                "score_logits": pred_group_out["score_logits"],
                "route_logits": route_logits,
                "candidate_ade": pred_group_ade,
                "candidate_fde": pred_group_fde,
                "candidate_dist": final_dist,
            },
        }
        if return_aux:
            return top1_pred, aux
        return top1_pred

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
        lat_targets=None,
        lon_targets=None,
        joint_targets=None,
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
            lat_targets=lat_targets,
            lon_targets=lon_targets,
            joint_targets=joint_targets,
        )
