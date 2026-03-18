import os

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import visualize_batch_trajectories


class DiffusionFut(nn.Module):
    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args
        if int(args.feature_dim) != 4:
            raise ValueError("Current future branch requires feature_dim=4: [rel_x, rel_y, v, a]")

        self.hidden_dim = int(args.hidden_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.dropout = float(args.dropout_fut)
        self.depth = int(args.depth_fut)
        self.mlp_ratio = int(args.mlp_ratio_fut)
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.time_embedding_size = int(args.time_embedding_size_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.inference_timestep_spacing = str(args.inference_timestep_spacing)
        self.ddim_eta = float(args.ddim_eta)
        self.T = int(args.T_f)

        self.self_condition_prob = min(max(float(args.self_condition_prob), 0.0), 1.0)
        self.huber_delta = max(1e-4, float(args.fut_huber_delta))
        self.loss_w_vel = 1.0
        self.loss_w_pos = max(0.0, float(args.fut_pos_loss_weight))
        self.loss_w_lat = max(0.0, float(getattr(args, "intent_loss_weight_lat", 0.20)))
        self.loss_w_lon = max(0.0, float(getattr(args, "intent_loss_weight_lon", 0.20)))

        x0_clip_val = float(args.x0_clip)
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.input_embedding = nn.Linear(self.output_dim * 2, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)
        if int(getattr(self.hist_encoder, "hidden_dim", self.hidden_dim)) != self.hidden_dim:
            raise ValueError(
                f"HistEncoder hidden_dim must match hidden_dim_fut, got {getattr(self.hist_encoder, 'hidden_dim', None)} vs {self.hidden_dim}"
            )

        self.cond_projs = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.depth)])
        for proj in self.cond_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)

        self.lat_emb = nn.Embedding(3, self.hidden_dim)
        self.lon_emb = nn.Embedding(3, self.hidden_dim)
        self.intent_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

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

        self.register_buffer(
            "hist_pos_mean",
            torch.tensor([0.05130798, -35.39044909], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "hist_pos_std",
            torch.tensor([9.63184438, 60.19290744], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "hist_va_mean",
            torch.tensor([23.92449619, 0.04203195], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "hist_va_std",
            torch.tensor([13.34587118, 4.61229342], dtype=torch.float32),
            persistent=False,
        )

        self.register_buffer(
            "fut_delta_mean",
            torch.tensor([-0.00407632, 5.53086274], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "fut_delta_std",
            torch.tensor([0.15694872, 2.88311335], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("lat_class_weight", torch.ones(3, dtype=torch.float32))
        self.register_buffer("lon_class_weight", torch.ones(3, dtype=torch.float32))

    def set_intent_class_weights(self, lat_weight, lon_weight):
        self.lat_class_weight.copy_(lat_weight.detach().to(self.lat_class_weight.device, dtype=self.lat_class_weight.dtype))
        self.lon_class_weight.copy_(lon_weight.detach().to(self.lon_class_weight.device, dtype=self.lon_class_weight.dtype))

    def prepareExtras(self, extras, hist, hist_nbrs, device):
        if extras is None:
            extras = {}
        batch_size, hist_len, _ = hist.shape
        nbr_total = hist_nbrs.size(0)
        default = {
            "ego_lane": torch.zeros(batch_size, hist_len, 1, device=device, dtype=hist.dtype),
            "nbr_lane": torch.zeros(nbr_total, hist_len, 1, device=device, dtype=hist.dtype),
            "nbr_dist": torch.zeros(nbr_total, hist_len, 1, device=device, dtype=hist.dtype),
            "lat_gt": None,
            "lon_gt": None,
        }
        merged = {}
        for key, value in default.items():
            merged[key] = extras.get(key, value)
        return merged

    def resolveForwardInputs(self, hist, hist_nbrs, extras, device):
        if device is None and not isinstance(extras, dict):
            device = extras
            extras = None
        if device is None:
            device = hist.device
        extras = self.prepareExtras(extras, hist, hist_nbrs, device)
        return extras, device

    def buildTargetVelNorm(self, future_phys, anchor_phys, device):
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys
        std_vel = self.fut_delta_std.view(1, 1, 2).to(device)
        mean_vel = self.fut_delta_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -10.0, 10.0)
        return target_vel_norm, target_vel_phys

    def buildSoftIntentEmbedding(self, lat_logits, lon_logits):
        lat_prob = torch.softmax(lat_logits, dim=-1)
        lon_prob = torch.softmax(lon_logits, dim=-1)
        lat_emb = lat_prob @ self.lat_emb.weight
        lon_emb = lon_prob @ self.lon_emb.weight
        return lat_emb + lon_emb

    def composeIntentCondition(self, intent_emb):
        return self.intent_fuse(intent_emb)

    def buildIntentCondition(self, lat_logits, lon_logits):
        return self.composeIntentCondition(self.buildSoftIntentEmbedding(lat_logits, lon_logits))

    def buildDiscreteIntentCondition(self, lat_idx, lon_idx):
        intent_emb = self.lat_emb(lat_idx) + self.lon_emb(lon_idx)
        return self.composeIntentCondition(intent_emb)

    @staticmethod
    def topKIntentCombinations(lat_logits, lon_logits, K):
        lat_prob = torch.softmax(lat_logits, dim=-1)
        lon_prob = torch.softmax(lon_logits, dim=-1)
        joint_prob = lat_prob.unsqueeze(-1) * lon_prob.unsqueeze(-2)
        flat_joint = joint_prob.reshape(joint_prob.size(0), -1)
        k_eff = min(max(1, int(K)), flat_joint.size(-1))
        topk_prob, topk_idx = torch.topk(flat_joint, k=k_eff, dim=-1)
        lon_classes = lon_prob.size(-1)
        lat_idx = topk_idx // lon_classes
        lon_idx = topk_idx % lon_classes
        return lat_idx, lon_idx, topk_prob

    def computeLoss(
        self,
        pred_vel_norm,
        target_vel_norm,
        future_phys,
        anchor_phys,
        valid_mask,
        lat_logits=None,
        lon_logits=None,
        lat_gt=None,
        lon_gt=None,
        return_components=False,
    ):
        if pred_vel_norm.size(-1) != 2 or target_vel_norm.size(-1) != 2:
            raise ValueError(
                f"computeLoss currently expects dim=2, got pred={pred_vel_norm.size(-1)}, target={target_vel_norm.size(-1)}"
            )

        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)

        std_vel = self.fut_delta_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.fut_delta_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        gt_pos_phys = future_phys[..., :2]
        loss_pos = F.smooth_l1_loss(pred_pos_phys, gt_pos_phys, reduction="none", beta=self.huber_delta)

        loss_vel_mean = self.maskedMean3d(loss_vel, valid_mask)
        loss_pos_mean = self.maskedMean3d(loss_pos, valid_mask)

        zero_scalar = pred_vel_norm.new_tensor(0.0)
        loss_lat = zero_scalar
        loss_lon = zero_scalar
        acc_lat = zero_scalar
        acc_lon = zero_scalar

        if lat_logits is not None and lat_gt is not None:
            loss_lat = F.cross_entropy(lat_logits, lat_gt, weight=self.lat_class_weight.to(lat_logits.device))
            acc_lat = (lat_logits.argmax(dim=-1) == lat_gt).float().mean()
        if lon_logits is not None and lon_gt is not None:
            loss_lon = F.cross_entropy(lon_logits, lon_gt, weight=self.lon_class_weight.to(lon_logits.device))
            acc_lon = (lon_logits.argmax(dim=-1) == lon_gt).float().mean()

        loss = (
            self.loss_w_vel * loss_vel_mean
            + self.loss_w_pos * loss_pos_mean
            + self.loss_w_lat * loss_lat
            + self.loss_w_lon * loss_lon
        )
        if not return_components:
            return loss

        loss_metrics = self.summarizeLossForLog(
            loss_vel=loss_vel,
            loss_pos=loss_pos,
            valid_mask=valid_mask,
            loss_total=loss.detach(),
            loss_lat=loss_lat.detach(),
            loss_lon=loss_lon.detach(),
            acc_lat=acc_lat.detach(),
            acc_lon=acc_lon.detach(),
        )
        return loss, loss_metrics

    @staticmethod
    def maskedMean3d(loss_tensor, valid_mask):
        valid = valid_mask.unsqueeze(-1)
        numer = (loss_tensor * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

    def summarizeLossForLog(self, loss_vel, loss_pos, valid_mask, loss_total, loss_lat, loss_lon, acc_lat, acc_lon):
        with torch.no_grad():
            return {
                "loss_total": loss_total,
                "loss_vel": self.maskedMean3d(loss_vel.detach(), valid_mask),
                "loss_pos": self.maskedMean3d(loss_pos.detach(), valid_mask),
                "loss_lat": loss_lat,
                "loss_lon": loss_lon,
                "acc_lat": acc_lat,
                "acc_lon": acc_lon,
            }

    @staticmethod
    def computeAdeFde(pred, target, valid_mask):
        diff = pred[..., :2] - target[..., :2]
        dist = torch.norm(diff, dim=-1)
        ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        t_idx = torch.arange(dist.size(1), device=dist.device).unsqueeze(0).expand_as(dist)
        masked_idx = torch.where(valid_mask > 0, t_idx, t_idx.new_full(t_idx.shape, -1))
        last_idx = masked_idx.max(dim=1).values
        has_valid = last_idx >= 0
        final_dist = dist.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
        return ade, fde

    def normHistoryInput(self, x):
        x_norm = x.clone()
        if x_norm.numel() == 0:
            return x_norm
        x_norm[..., 0:2] = (x[..., 0:2] - self.hist_pos_mean) / self.hist_pos_std
        x_norm[..., 2:4] = (x[..., 2:4] - self.hist_va_mean) / self.hist_va_std
        x_norm = torch.clamp(x_norm, -10.0, 10.0)
        return x_norm

    def encodeHistoryCondition(self, hist, hist_nbrs, mask, temporal_mask, extras):
        hist_state_norm = self.normHistoryInput(hist)
        nbr_state_norm = self.normHistoryInput(hist_nbrs)
        memory_tokens, memory_mask, lat_logits, lon_logits = self.hist_encoder(
            hist_state_norm,
            nbr_state_norm,
            mask,
            temporal_mask,
            extras["ego_lane"],
            extras["nbr_lane"],
            extras["nbr_dist"],
            ego_state_raw=hist,
            nbr_state_raw=hist_nbrs,
        )
        return memory_tokens, memory_mask, lat_logits, lon_logits

    def buildLayerCondition(self, timesteps, intent_cond):
        t_emb = self.timestep_embedder(timesteps)
        return [t_emb + proj(intent_cond) for proj in self.cond_projs]

    def predictX0(self, x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_x0_cond):
        y_layers = self.buildLayerCondition(timesteps, intent_cond)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        return self.dit(x=input_embedded, y=y_layers, cross=memory_tokens, cross_attn_mask=memory_mask)

    def rolloutFromXt(self, x_t, memory_tokens, intent_cond, memory_mask, infer_scheduler):
        batch_size, t_len, _ = x_t.shape
        pred_vel_cond = torch.zeros((batch_size, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((batch_size,), t_scalar, device=x_t.device, dtype=torch.long)
            pred_vel_norm = self.predictX0(x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_vel_cond)
            if self.x0_clip is not None:
                pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)
            pred_vel_cond = pred_vel_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t).prev_sample
        return pred_vel_cond

    def maybeVisualize(self, hist, hist_nbrs, temporal_mask, future, pred, valid_mask, stage, pred_all=None, pred_best_idx=None):
        if not self.is_main_process:
            return
        if stage == "train":
            if not self.fut_enable_train_vis:
                return
        else:
            if not self.fut_enable_eval_vis:
                return

        vis_batch_idx = 0
        b_idx = min(max(int(vis_batch_idx), 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - future[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        vis_ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_idx = torch.nonzero(vm > 0, as_tuple=False).squeeze(-1)
        vis_fde = dist[valid_idx[-1]] if valid_idx.numel() > 0 else dist.new_tensor(0.0)
        metrics = {
            "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
            "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        }
        visualize_batch_trajectories(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=pred,
            pred_all=pred_all,
            pred_best_idx=pred_best_idx,
            future_mask=valid_mask,
            batch_idx=vis_batch_idx,
            save_path=None,
            metrics=metrics,
            input_unit="ft",
            show_plot=True,
        )

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, epoch=1, return_components=False):
        extras, device = self.resolveForwardInputs(hist, hist_nbrs, extras, device)
        batch_size, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        target_vel_norm, _ = self.buildTargetVelNorm(future_phys, anchor_phys, device)

        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        memory_tokens, memory_mask, lat_logits, lon_logits = self.encodeHistoryCondition(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            extras,
        )
        intent_cond = self.buildIntentCondition(lat_logits, lon_logits)

        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(batch_size, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(
                        x_t,
                        timesteps,
                        memory_tokens,
                        intent_cond,
                        memory_mask,
                        pred_vel_cond,
                    )
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        pred_vel_norm_t = self.predictX0(x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_vel_cond)
        loss, loss_metrics = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
            return_components=True,
        )

        if self.fut_enable_train_vis:
            std_vel = self.fut_delta_std.view(1, 1, 2).to(device)
            mean_vel = self.fut_delta_mean.view(1, 1, 2).to(device)
            pred_vel_phys_t = pred_vel_norm_t[..., :2] * std_vel + mean_vel
            pred_pos_phys = torch.cumsum(pred_vel_phys_t, dim=1) + anchor_phys[..., :2]
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            self.maybeVisualize(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=pred_phys_abs,
                valid_mask=valid_mask,
                stage="train",
            )

        if return_components:
            return loss, loss_metrics
        return loss

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None):
        extras, device = self.resolveForwardInputs(hist, hist_nbrs, extras, device)
        batch_size, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        memory_tokens, memory_mask, lat_logits, lon_logits = self.encodeHistoryCondition(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            extras,
        )
        intent_cond = self.buildIntentCondition(lat_logits, lon_logits)

        infer_scheduler = DDIMScheduler.from_config(
            self.diffusion_scheduler.config,
            timestep_spacing=self.inference_timestep_spacing,
        )
        infer_scheduler.set_timesteps(self.num_inference_steps)

        x_t = torch.randn((batch_size, t_len, self.output_dim), device=device)
        pred_vel_norm = self.rolloutFromXt(x_t, memory_tokens, intent_cond, memory_mask, infer_scheduler)

        std_vel = self.fut_delta_std.view(1, 1, 2).to(device)
        mean_vel = self.fut_delta_mean.view(1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]

        pred_phys_abs = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys
        target_vel_norm, _ = self.buildTargetVelNorm(future_phys, anchor_phys, device)
        loss = self.computeLoss(
            pred_vel_norm,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
        )

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        self.maybeVisualize(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=pred_phys_abs,
            valid_mask=valid_mask,
            stage="eval",
        )
        return loss, pred_phys_abs, ade, fde

    @torch.no_grad()
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, K=5):
        extras, device = self.resolveForwardInputs(hist, hist_nbrs, extras, device)
        batch_size, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        memory_tokens, memory_mask, lat_logits, lon_logits = self.encodeHistoryCondition(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            extras,
        )
        lat_idx, lon_idx, topk_prob = self.topKIntentCombinations(lat_logits, lon_logits, K)
        k_eff = lat_idx.size(1)

        memory_tokens_k = memory_tokens.repeat_interleave(k_eff, dim=0)
        memory_mask_k = memory_mask.repeat_interleave(k_eff, dim=0)
        intent_cond_k = self.buildDiscreteIntentCondition(lat_idx.reshape(-1), lon_idx.reshape(-1))

        infer_scheduler = DDIMScheduler.from_config(
            self.diffusion_scheduler.config,
            timestep_spacing=self.inference_timestep_spacing,
        )
        infer_scheduler.set_timesteps(self.num_inference_steps)
        x_t_k = torch.randn((batch_size * k_eff, t_len, self.output_dim), device=device)
        pred_vel_norm_k = self.rolloutFromXt(x_t_k, memory_tokens_k, intent_cond_k, memory_mask_k, infer_scheduler)

        pred_vel_norm = pred_vel_norm_k.view(batch_size, k_eff, t_len, self.output_dim)
        std_vel = self.fut_delta_std.view(1, 1, 1, 2).to(device)
        mean_vel = self.fut_delta_mean.view(1, 1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys[..., :2].unsqueeze(1)

        all_preds = future_phys.unsqueeze(1).repeat(1, k_eff, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys
        target_phys = future_phys[..., :2].unsqueeze(1)
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)
        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        _, best_k_idx = torch.min(ade_k, dim=1)

        best_pred_idx = best_k_idx.view(batch_size, 1, 1, 1).expand(batch_size, 1, t_len, self.output_dim)
        best_pred_phys = all_preds.gather(1, best_pred_idx).squeeze(1)
        best_pred_vel_norm = pred_vel_norm.gather(1, best_pred_idx).squeeze(1)
        target_vel_norm, _ = self.buildTargetVelNorm(future_phys, anchor_phys, device)
        loss = self.computeLoss(
            best_pred_vel_norm,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
        )

        self.last_minade_all_preds = all_preds.detach()
        self.last_minade_best_idx = best_k_idx.detach()
        self.last_minade_intent_pairs = torch.stack((lat_idx, lon_idx), dim=-1).detach()
        self.last_minade_intent_prob = topk_prob.detach()

        ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
        self.maybeVisualize(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=best_pred_phys,
            valid_mask=valid_mask,
            stage="eval",
            pred_all=all_preds,
            pred_best_idx=best_k_idx,
        )
        return loss, best_pred_phys, ade_batch, fde_batch

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, epoch=1, return_components=False):
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            extras,
            device,
            epoch=epoch,
            return_components=return_components,
        )

    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.hist_pos_mean) / self.hist_pos_std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -10.0, 10.0)
        channels = x_norm.shape[-1]
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.hist_va_mean) / self.hist_va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -10.0, 10.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.hist_pos_std + self.hist_pos_mean
        channels = x.shape[-1]
        if channels >= 4:
            x_denorm[..., 2:4] = x[..., 2:4] * self.hist_va_std + self.hist_va_mean
        return x_denorm
