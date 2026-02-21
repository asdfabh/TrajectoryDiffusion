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
        self.y_loss_weight = max(1.0, float(args.fut_y_loss_weight))
        self.huber_delta = max(1e-4, float(args.fut_huber_delta))
        x0_clip_val = float(args.x0_clip)
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.fut_vis_every_n = max(1, int(args.fut_vis_every_n))
        self.trainForwardCalls = 0
        self.evalForwardCalls = 0
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.input_embedding = nn.Linear(self.output_dim * 2, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)

        self.enc_embedding = nn.Linear(int(args.encoder_input_dim) * 2, self.hidden_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        # 投影桥：对齐 HistEncoder(128维) 和 DiT 的 hidden_dim
        self.context_proj = nn.Linear(int(args.encoder_input_dim) * 2, self.hidden_dim)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.constant_(self.context_proj.bias, 0)

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

        self.register_buffer("pos_mean", torch.tensor([0.0330, -15.9150]).float(), persistent=False)
        self.register_buffer("pos_std", torch.tensor([8.8866, 68.8105]).float(), persistent=False)
        self.register_buffer("va_mean", torch.tensor([21.1503, 0.0060]).float(), persistent=False)
        self.register_buffer("va_std", torch.tensor([13.5983, 4.5057]).float(), persistent=False)

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

    def buildInferenceScheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config, timestep_spacing=self.inference_timestep_spacing)
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    def encodeGlobalCondition(self, context):
        latest_context = context[:, -1, :]
        return self.enc_embedding(latest_context)

    # Predict clean residual x0 from noisy residual x_t.
    def predictX0(self, x_t, timesteps, context_aligned, enc_emb, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        pred_x0 = self.dit(x=input_embedded, y=t_emb + enc_emb, cross=context_aligned)
        return pred_x0

    def computeLoss(self, pred_res, target_res, valid_mask):
        weights = [1.0] * self.output_dim
        if self.output_dim >= 2:
            weights[1] = self.y_loss_weight
        axis_weight = torch.tensor(weights, device=pred_res.device, dtype=pred_res.dtype).view(1, 1, self.output_dim)

        # 使用 Huber Loss 保证早期极高噪声时的梯度截断安全
        loss = F.smooth_l1_loss(pred_res, target_res, reduction="none", beta=self.huber_delta)
        loss = loss * axis_weight

        # 时序递增惩罚：逼迫模型随着时间推移咬死远端 FDE (从 1.0 线性递增到 2.0)
        T_len = loss.size(1)
        time_weights = torch.linspace(1.0, 2.0, T_len, device=loss.device, dtype=loss.dtype).view(1, T_len, 1)
        loss = loss * time_weights

        valid = valid_mask.unsqueeze(-1)
        numer = (loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

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
        b_idx = min(max(int(batch_idx), 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - target[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_count = int(vm.sum().item())
        fde = dist[valid_count - 1] if valid_count > 0 else dist.new_tensor(0.0)
        return ade, fde

    def rolloutFromXt(self, x_t, context_aligned, enc_emb, infer_scheduler):
        bsz, t_len, _ = x_t.shape
        pred_res_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)
            pred_res_norm = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_res_cond)
            if self.x0_clip is not None:
                pred_res_norm = torch.clamp(pred_res_norm, -self.x0_clip, self.x0_clip)
            pred_res_cond = pred_res_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_res_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_res_norm, t, x_t).prev_sample
        return pred_res_cond

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

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
        hist_norm = self.norm(hist)
        future_norm = self.norm(future)

        anchor = hist_norm[:, -1:, :self.output_dim]
        target_res = future_norm[..., :self.output_dim] - anchor

        noise = torch.randn_like(target_res)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_res, noise, timesteps)

        context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context)

        context_aligned = self.context_proj(context)

        pred_res_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_res = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_res_cond)
                pred_res_cond = prev_pred_res.detach() * use_sc

        pred_res_t = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_res_cond)
        loss = self.computeLoss(pred_res_t, target_res, valid_mask)

        if self.fut_enable_train_vis:
            pred_norm_abs = pred_res_t + anchor
            pred_phys = self.denorm(pred_norm_abs)
            self.maybeVisualize(hist, future, pred_phys, valid_mask, stage="train")

        return loss

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
        hist_norm = self.norm(hist)
        future_norm = self.norm(future)

        anchor = hist_norm[:, -1:, :self.output_dim]
        context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context)

        context_aligned = self.context_proj(context)

        infer_scheduler = self.buildInferenceScheduler()
        x_t = torch.randn((bsz, t_len, self.output_dim), device=device)
        pred_res_norm = self.rolloutFromXt(x_t, context_aligned, enc_emb, infer_scheduler)

        target_res = future_norm[..., :self.output_dim] - anchor
        loss = self.computeLoss(pred_res_norm, target_res, valid_mask)

        pred_norm_abs = pred_res_norm + anchor
        pred_phys = self.denorm(pred_norm_abs)
        ade, fde = self.computeAdeFde(pred_phys, future, valid_mask)
        self.maybeVisualize(hist, future, pred_phys, valid_mask, stage="eval")
        return loss, pred_phys, ade, fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -10.0, 10.0)
        channels = x_norm.shape[-1]
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -10.0, 10.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
        channels = x.shape[-1]
        if channels >= 4:
            x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean
        return x_denorm