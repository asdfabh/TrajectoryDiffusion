from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import torch
import torch.nn.functional as F
import os
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import visualize_batch_trajectories


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args

        self.feature_dim = int(args.feature_dim_fut)
        self.input_dim = int(args.input_dim_fut)
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
        x0_clip_val = float(args.x0_clip)
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None

        self.self_condition_prob = float(args.self_condition_prob)
        self.self_condition_prob = min(max(self.self_condition_prob, 0.0), 1.0)
        self.train_unroll_weight = float(args.train_unroll_weight)
        self.train_timestep_align_ratio = float(args.train_timestep_align_ratio)
        self.train_timestep_align_ratio = min(max(self.train_timestep_align_ratio, 0.0), 1.0)
        self.train_unroll_detach_x0 = int(args.train_unroll_detach_x0) > 0

        self.fut_loss_mode = str(args.fut_loss_mode)
        self.fut_loss_pos_weight = float(args.fut_loss_pos_weight)
        self.fut_loss_vel_weight = float(args.fut_loss_vel_weight)
        self.fut_time_weight_min = float(args.fut_time_weight_min)
        self.fut_time_weight_max = float(args.fut_time_weight_max)

        self.fut_pos_loss_type = str(args.fut_pos_loss_type)
        self.fut_huber_delta = max(1e-4, float(args.fut_huber_delta))
        self.fut_loss_acc_weight = float(args.fut_loss_acc_weight)
        self.fut_loss_endpoint_weight = float(args.fut_loss_endpoint_weight)
        self.fut_high_noise_threshold = min(max(float(args.fut_high_noise_threshold), 0.0), 1.0)
        self.fut_high_noise_weight = max(1.0, float(args.fut_high_noise_weight))

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.fut_vis_every_n = max(1, int(args.fut_vis_every_n))
        self.trainForwardCalls = 0
        self.evalForwardCalls = 0

        self.T = int(args.T_f)
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.input_embedding = nn.Linear(self.feature_dim + self.output_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)
        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            beta_start=1e-4,
            beta_end=2e-2,
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=self.final_layer, depth=self.depth, model_type="x_start")

        self.register_buffer("alphas_cumprod", self.diffusion_scheduler.alphas_cumprod.float().clone(), persistent=False)

        align_scheduler = self.buildInferenceScheduler()
        align_ts = align_scheduler.timesteps.long().clone()
        if align_ts.ndim == 0:
            align_ts = align_ts.unsqueeze(0)
        self.register_buffer("aligned_train_timesteps", align_ts, persistent=False)

        self.register_buffer("pos_mean", torch.tensor([0.0, 0.0]).float(), persistent=False)
        self.register_buffer("pos_std", torch.tensor([10, 150]).float(), persistent=False)
        self.register_buffer("va_mean", torch.tensor([20, 0.01]).float(), persistent=False)
        self.register_buffer("va_std", torch.tensor([15, 5]).float(), persistent=False)

    @staticmethod
    def toValidMask(op_mask, seq_len, batch_size, device):
        if op_mask is None:
            return torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)
        if op_mask.dim() == 3:
            valid = op_mask[..., 0]
        elif op_mask.dim() == 2:
            valid = op_mask
        else:
            raise ValueError(f"Unsupported op_mask shape: {tuple(op_mask.shape)}")
        valid = (valid > 0.5).float()
        if valid.size(1) > seq_len:
            valid = valid[:, :seq_len]
        elif valid.size(1) < seq_len:
            pad = torch.ones((batch_size, seq_len - valid.size(1)), dtype=valid.dtype, device=valid.device)
            valid = torch.cat([valid, pad], dim=1)
        return valid.to(device)

    @staticmethod
    def maskedReducePerSample(value, mask):
        value = value * mask
        sum_dims = tuple(range(1, value.dim()))
        numer = value.sum(dim=sum_dims)
        denom = mask.sum(dim=sum_dims) + 1e-6
        return numer / denom

    @staticmethod
    def weightedBatchReduce(loss_per_sample, sample_weight=None):
        if sample_weight is None:
            return loss_per_sample.mean()
        weight = torch.clamp(sample_weight, min=0.0)
        weight_sum = weight.sum()
        if weight_sum <= 1e-6:
            return loss_per_sample.new_tensor(0.0)
        return (loss_per_sample * weight).sum() / (weight_sum + 1e-6)

    def sampleTrainTimesteps(self, batch_size, device, need_prev_step=False):
        low = 1 if need_prev_step else 0
        uniform_t = torch.randint(low, self.num_train_timesteps, (batch_size,), device=device).long()
        if self.train_timestep_align_ratio <= 0.0:
            return uniform_t
        align_pool = self.aligned_train_timesteps.to(device)
        if need_prev_step:
            align_pool = align_pool[align_pool > 0]
        if align_pool.numel() == 0:
            return uniform_t
        align_ids = torch.randint(0, align_pool.numel(), (batch_size,), device=device)
        aligned_t = align_pool[align_ids].long()
        aligned_t = torch.clamp(aligned_t, min=low, max=self.num_train_timesteps - 1)
        choose_align = (torch.rand(batch_size, device=device) < self.train_timestep_align_ratio)
        return torch.where(choose_align, aligned_t, uniform_t)

    def buildInferenceScheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config, timestep_spacing=self.inference_timestep_spacing)
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    def buildNoiseSampleWeight(self, timesteps):
        weights = torch.ones_like(timesteps, dtype=torch.float32)
        if self.fut_high_noise_weight <= 1.0:
            return weights
        t_ratio = timesteps.float() / float(max(1, self.num_train_timesteps - 1))
        high_mask = t_ratio >= self.fut_high_noise_threshold
        high_value = torch.full_like(weights, self.fut_high_noise_weight)
        return torch.where(high_mask, high_value, weights)

    def predictX0(self, x_t, timesteps, context, enc_emb, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        y = t_emb + enc_emb
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(combined_input)
        pred_x0 = self.dit(x=input_embedded, y=y, cross=context)
        if self.x0_clip is not None:
            pred_x0 = torch.clamp(pred_x0, -self.x0_clip, self.x0_clip)
        return pred_x0

    def rollToPrevXt(self, x_t, pred_x0, timesteps):
        bsz = x_t.size(0)
        t_idx = torch.clamp(timesteps.long(), min=0, max=self.num_train_timesteps - 1)
        prev_idx = torch.clamp(t_idx - 1, min=0)

        alpha_t = self.alphas_cumprod[t_idx].view(bsz, 1, 1).to(x_t.device)
        alpha_prev = self.alphas_cumprod[prev_idx].view(bsz, 1, 1).to(x_t.device)

        sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-8))
        sqrt_one_minus_alpha_t = torch.sqrt((1.0 - alpha_t).clamp(min=1e-8))
        eps_hat = (x_t - sqrt_alpha_t * pred_x0) / (sqrt_one_minus_alpha_t + 1e-8)

        x_prev = torch.sqrt(alpha_prev.clamp(min=1e-8)) * pred_x0 + torch.sqrt((1.0 - alpha_prev).clamp(min=1e-8)) * eps_hat
        has_prev = (t_idx > 0).view(bsz, 1, 1)
        x_prev = torch.where(has_prev, x_prev, x_t)
        return x_prev

    def buildTimeWeights(self, seq_len, device, dtype):
        w_min = self.fut_time_weight_min
        w_max = self.fut_time_weight_max
        if w_max < w_min:
            w_min, w_max = w_max, w_min
        if seq_len <= 1:
            return torch.ones((1, seq_len, 1), device=device, dtype=dtype) * w_max
        weights = torch.linspace(w_min, w_max, seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
        return weights

    def computeLossL1TimeVel(self, pred, target, valid_mask, sample_weight=None):
        pred_pos = pred[..., :2]
        target_pos = target[..., :2]
        mask_xy = valid_mask.unsqueeze(-1).expand_as(pred_pos)
        time_weight = self.buildTimeWeights(pred_pos.size(1), pred.device, pred.dtype)
        weighted_mask_xy = mask_xy * time_weight.expand(pred_pos.size(0), -1, pred_pos.size(2))

        pos_err = torch.abs(pred_pos - target_pos)
        loss_pos_ps = self.maskedReducePerSample(pos_err, weighted_mask_xy)

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        if pred_vel.numel() == 0:
            loss_vel_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_vel = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1).expand_as(pred_vel)
            time_weight_vel = 0.5 * (time_weight[:, 1:, :] + time_weight[:, :-1, :])
            weighted_valid_vel = valid_vel * time_weight_vel.expand(pred_pos.size(0), -1, pred_pos.size(2))
            vel_err = torch.abs(pred_vel - target_vel)
            loss_vel_ps = self.maskedReducePerSample(vel_err, weighted_valid_vel)

        total_ps = self.fut_loss_pos_weight * loss_pos_ps + self.fut_loss_vel_weight * loss_vel_ps
        return self.weightedBatchReduce(total_ps, sample_weight=sample_weight)

    # Legacy full loss (position + velocity + acceleration + endpoint) for later experiments.
    # This path is intentionally not default for current training stage.
    def computeLossLegacy(self, pred, target, valid_mask, sample_weight=None):
        mask_xy = valid_mask.unsqueeze(-1).expand_as(pred[..., :2])
        pred_pos = pred[..., :2]
        target_pos = target[..., :2]

        if self.fut_pos_loss_type == "huber":
            pos_err = F.smooth_l1_loss(pred_pos, target_pos, reduction="none", beta=self.fut_huber_delta)
        else:
            pos_err = torch.abs(pred_pos - target_pos)
        loss_pos_ps = self.maskedReducePerSample(pos_err, mask_xy)

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        if pred_vel.numel() == 0:
            loss_vel_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_vel = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1).expand_as(pred_vel)
            vel_err = torch.abs(pred_vel - target_vel)
            loss_vel_ps = self.maskedReducePerSample(vel_err, valid_vel)

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        if pred_acc.numel() == 0:
            loss_acc_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_acc = (valid_mask[:, 2:] * valid_mask[:, 1:-1] * valid_mask[:, :-2]).unsqueeze(-1).expand_as(pred_acc)
            acc_err = torch.abs(pred_acc - target_acc)
            loss_acc_ps = self.maskedReducePerSample(acc_err, valid_acc)

        valid_counts = valid_mask.sum(dim=1).long()
        has_valid = (valid_counts > 0).float()
        last_idx = torch.clamp(valid_counts - 1, min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)
        pred_end = pred_pos.gather(1, gather_idx).squeeze(1)
        target_end = target_pos.gather(1, gather_idx).squeeze(1)
        if self.fut_pos_loss_type == "huber":
            end_err = F.smooth_l1_loss(pred_end, target_end, reduction="none", beta=self.fut_huber_delta).mean(dim=-1)
        else:
            end_err = torch.abs(pred_end - target_end).mean(dim=-1)
        loss_end_ps = end_err * has_valid

        total_ps = (
            self.fut_loss_pos_weight * loss_pos_ps
            + self.fut_loss_vel_weight * loss_vel_ps
            + self.fut_loss_acc_weight * loss_acc_ps
            + self.fut_loss_endpoint_weight * loss_end_ps
        )
        return self.weightedBatchReduce(total_ps, sample_weight=sample_weight)

    def computeLoss(self, pred, target, valid_mask, sample_weight=None):
        if self.fut_loss_mode == "legacy":
            return self.computeLossLegacy(pred, target, valid_mask, sample_weight=sample_weight)
        # Legacy loss path is disabled by default; current default is L1 + time weight + velocity.
        return self.computeLossL1TimeVel(pred, target, valid_mask, sample_weight=sample_weight)

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

    @staticmethod
    def computeSingleAdeFde(pred, target, valid_mask, batch_idx=0):
        b_idx = min(max(batch_idx, 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - target[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_count = int(vm.sum().item())
        if valid_count > 0:
            fde = dist[valid_count - 1]
        else:
            fde = dist.new_tensor(0.0)
        return ade, fde

    def maybeVisualize(self, hist, future, pred, valid_mask, stage):
        if not self.is_main_process:
            return
        if stage == "train":
            self.trainForwardCalls += 1
            if (not self.fut_enable_train_vis) or (self.trainForwardCalls % self.fut_vis_every_n != 0):
                return
        else:
            self.evalForwardCalls += 1
            if (not self.fut_enable_eval_vis) or (self.evalForwardCalls % self.fut_vis_every_n != 0):
                return

        vis_batch_idx = 0
        vis_ade, vis_fde = self.computeSingleAdeFde(pred, future, valid_mask, batch_idx=vis_batch_idx)
        metrics = {
            "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
            "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        }
        visualize_batch_trajectories(
            hist=hist,
            hist_nbrs=None,
            temporal_mask=None,
            future=future,
            pred=pred,
            future_mask=valid_mask,
            batch_idx=vis_batch_idx,
            save_path=None,
            metrics=metrics,
            input_unit="ft",
            show_plot=True,
        )
        print(
            f"[{stage}][Vis Traj idx={vis_batch_idx}] ADE: {vis_ade.item():.4f} ft ({vis_ade.item() * self.meter_per_foot:.4f} m), "
            f"FDE: {vis_fde.item():.4f} ft ({vis_fde.item() * self.meter_per_foot:.4f} m)"
        )

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, feat_dim = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
        future_norm = self.norm(future)
        x_start = future_norm
        noise = torch.randn_like(x_start)

        need_prev_step = self.train_unroll_weight > 0.0
        timesteps = self.sampleTrainTimesteps(bsz, device, need_prev_step=need_prev_step)
        x_t = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.enc_embedding(hist_enc[:, -1, :])

        pred_x0_cond = torch.zeros((bsz, t_len, self.output_dim), device=device)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, device=device) < self.self_condition_prob).view(bsz, 1, 1).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
                pred_x0_cond = prev_pred.detach() * use_sc

        pred_x0_t = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
        sample_weight = self.buildNoiseSampleWeight(timesteps)
        loss_t = self.computeLoss(pred_x0_t, future_norm, valid_mask, sample_weight=sample_weight)
        total_loss = loss_t

        if self.train_unroll_weight > 0.0:
            prev_timesteps = torch.clamp(timesteps - 1, min=0)
            x0_for_roll = pred_x0_t.detach() if self.train_unroll_detach_x0 else pred_x0_t
            x_tm1 = self.rollToPrevXt(x_t, x0_for_roll, timesteps)
            pred_x0_tm1 = self.predictX0(x_tm1, prev_timesteps, context, enc_emb, pred_x0_t.detach())
            prev_weight = sample_weight * (timesteps > 0).float()
            loss_tm1 = self.computeLoss(pred_x0_tm1, future_norm, valid_mask, sample_weight=prev_weight)
            total_loss = total_loss + self.train_unroll_weight * loss_tm1

        pred = self.denorm(pred_x0_t)
        ade, fde = self.computeAdeFde(pred, future, valid_mask)
        self.maybeVisualize(hist, future, pred, valid_mask, stage="train")
        return total_loss, pred, ade, fde

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, feat_dim = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)

        x_t = torch.randn((bsz, t_len, feat_dim), device=device)
        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.enc_embedding(hist_enc[:, -1, :])

        infer_scheduler = self.buildInferenceScheduler()
        pred_x0_cond = torch.zeros((bsz, t_len, self.output_dim), device=device)

        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=device, dtype=torch.long)
            pred_x0_norm = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
            pred_x0_cond = pred_x0_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        pred = self.denorm(x_t)
        pred_norm = self.norm(pred)
        future_norm = self.norm(future)
        loss = self.computeLoss(pred_norm, future_norm, valid_mask)
        ade, fde = self.computeAdeFde(pred, future, valid_mask)
        self.maybeVisualize(hist, future, pred, valid_mask, stage="eval")
        return loss, pred, ade, fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        channels = x_norm.shape[-1]
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -5.0, 5.0)
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
        channels = x.shape[-1]
        if channels >= 4:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean
        return x_denorm
