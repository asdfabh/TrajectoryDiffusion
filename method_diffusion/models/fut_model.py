from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import torch
import os
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import visualize_batch_trajectories

class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        # Net parameters
        self.args = args
        self.feature_dim = int(args.feature_dim_fut) # 输入特征维度 default: 6 (x, y, v, a, laneID, class)
        self.input_dim = int(args.input_dim_fut)   # 输入到Dit的维度 default: 128
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
        self.train_unroll_weight = float(args.train_unroll_weight)
        self.train_timestep_align_ratio = float(args.train_timestep_align_ratio)
        self.train_timestep_align_ratio = max(0.0, min(1.0, self.train_timestep_align_ratio))
        self.T = int(args.T_f)
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
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
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            depth=self.depth,
            model_type="x_start"
        )

        self.register_buffer('pos_mean', torch.tensor([0.0, 0.0]).float(), persistent=False)
        self.register_buffer('pos_std', torch.tensor([10, 150]).float(), persistent=False)
        self.register_buffer('va_mean', torch.tensor([20, 0.01]).float(), persistent=False)
        self.register_buffer('va_std', torch.tensor([15, 5]).float(), persistent=False)

        self.train_timestep_stride = max(1, self.num_train_timesteps // max(1, self.num_inference_steps))
        aligned_timesteps = torch.arange(self.num_train_timesteps, dtype=torch.long)
        try:
            align_scheduler = DDIMScheduler.from_config(
                self.diffusion_scheduler.config,
                timestep_spacing=self.inference_timestep_spacing,
            )
            align_scheduler.set_timesteps(self.num_inference_steps)
            aligned_timesteps = torch.unique(align_scheduler.timesteps.long())
        except Exception:
            pass
        self.register_buffer("train_align_timesteps", aligned_timesteps, persistent=False)

    @staticmethod
    def _to_valid_mask(op_mask, seq_len, batch_size, device):
        if op_mask is None:
            return torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)

        if op_mask.dim() == 3:
            valid = op_mask[..., 0]
        elif op_mask.dim() == 2:
            valid = op_mask
        else:
            raise ValueError(f"Unsupported op_mask shape: {tuple(op_mask.shape)}")

        valid = (valid > 0.5).float()
        if valid.size(1) != seq_len:
            valid = valid[:, :seq_len]
        return valid.to(device)

    @staticmethod
    def _masked_l1(diff, mask):
        return (diff * mask).sum() / (mask.sum() + 1e-6)

    def _sample_train_timesteps(self, batch_size, device):
        base = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)
        if self.train_timestep_align_ratio <= 0.0 or self.train_align_timesteps.numel() == 0:
            return base

        aligned = self.train_align_timesteps.to(device)
        aligned_idx = torch.randint(0, aligned.numel(), (batch_size,), device=device)
        aligned_t = aligned[aligned_idx]
        use_aligned = torch.rand(batch_size, device=device) < self.train_timestep_align_ratio
        timesteps = torch.where(use_aligned, aligned_t, base)
        return timesteps.clamp(0, self.num_train_timesteps - 1).long()

    def _build_inference_scheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(
                self.diffusion_scheduler.config,
                timestep_spacing=self.inference_timestep_spacing,
            )
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    def compute_motion_loss(self, pred, target, valid_mask):
        """
        pred: [B, T, D]
        target: [B, T, D]
        valid_mask: [B, T], 1 indicates valid future timestep.
        """
        mask_xy = valid_mask.unsqueeze(-1).expand_as(pred[..., :2])
        loss_l1 = self._masked_l1(torch.abs(pred[..., :2] - target[..., :2]), mask_xy)

        pred_pos = pred[..., :2]
        target_pos = target[..., :2]

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        valid_vel = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1).expand_as(pred_vel)
        loss_vel = self._masked_l1(torch.abs(pred_vel - target_vel), valid_vel)

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        if pred_acc.numel() == 0:
            loss_acc = pred.new_tensor(0.0)
        else:
            valid_acc = (valid_mask[:, 2:] * valid_mask[:, 1:-1] * valid_mask[:, :-2]).unsqueeze(-1).expand_as(pred_acc)
            loss_acc = self._masked_l1(torch.abs(pred_acc - target_acc), valid_acc)

        total_loss = 1.2 * loss_l1 + 0.5 * loss_vel + 0.15 * loss_acc
        return total_loss

    @staticmethod
    def compute_ade_fde(pred, target, valid_mask):
        diff = pred[..., :2] - target[..., :2]
        dist = torch.norm(diff, dim=-1)  # [B, T]

        ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)

        valid_counts = valid_mask.sum(dim=1).long()
        has_valid = valid_counts > 0
        last_idx = torch.clamp(valid_counts - 1, min=0)
        final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
        return ade, fde

    @staticmethod
    def compute_single_ade_fde(pred, target, valid_mask, batch_idx=0):
        b_idx = min(max(batch_idx, 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - target[b_idx, :, :2]  # [T, 2]
        dist = torch.norm(diff, dim=-1)  # [T]
        vm = valid_mask[b_idx]  # [T]

        ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_count = int(vm.sum().item())
        if valid_count > 0:
            fde = dist[valid_count - 1]
        else:
            fde = dist.new_tensor(0.0)
        return ade, fde

    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        B, T, dim = future.shape
        valid_mask = self._to_valid_mask(op_mask, T, B, device)
        future_norm = self.norm(future)  # [B, T, 2]
        x_start = future_norm
        noise = torch.randn_like(x_start)
        timesteps = self._sample_train_timesteps(B, device)
        x_noisy = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)
        model_input = x_noisy  # [B, T, 2]

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)

        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)  # [B, T, hidden_dim]
        t_emb = self.timestep_embedder(timesteps)
        enc_emb = self.enc_embedding(hist_enc[:, -1, :])  # [B, D]
        y = t_emb + enc_emb

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, y=y, cross=context)
        loss_main = self.compute_motion_loss(pred_x0, future_norm, valid_mask)
        loss = loss_main
        if self.train_unroll_weight > 0.0:
            prev_t = torch.clamp(timesteps - self.train_timestep_stride, min=0)
            # Re-noise predicted x0 to expose the model to its own off-manifold states.
            x_prev_noisy = self.diffusion_scheduler.add_noise(pred_x0.detach(), noise, prev_t)
            y_prev = self.timestep_embedder(prev_t) + enc_emb
            input_prev_embedded = self.input_embedding(x_prev_noisy) + self.pos_embedding(x_prev_noisy)
            pred_x0_prev = self.dit(x=input_prev_embedded, y=y_prev, cross=context)
            loss_unroll = self.compute_motion_loss(pred_x0_prev, future_norm, valid_mask)
            loss = loss + self.train_unroll_weight * loss_unroll
        pred = self.denorm(pred_x0)
        ade, fde = self.compute_ade_fde(pred, future, valid_mask)

        # ===== 可视化调试代码（注释本段即可关闭）=====
        # if self.is_main_process:
        #     vis_batch_idx = 0
        #     vis_ade, vis_fde = self.compute_single_ade_fde(pred, future, valid_mask, batch_idx=vis_batch_idx)
        #     metrics = {
        #         "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
        #         "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        #     }
        #     visualize_batch_trajectories(
        #         hist=hist,
        #         hist_nbrs=None,
        #         temporal_mask=None,
        #         future=future,
        #         pred=pred,
        #         future_mask=valid_mask,
        #         batch_idx=vis_batch_idx,
        #         save_path=None,   # 不保存，只show
        #         metrics=metrics,
        #         input_unit='ft',
        #         show_plot=True
        #     )
        #     print(
        #         f"[Train][Vis Traj idx={vis_batch_idx}] "
        #         f"ADE: {vis_ade.item():.4f} ft ({vis_ade.item() * self.meter_per_foot:.4f} m), "
        #         f"FDE: {vis_fde.item():.4f} ft ({vis_fde.item() * self.meter_per_foot:.4f} m)"
        #     )
        # ===== 可视化调试代码结束 =====

        return loss, pred, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        B, T, dim = future.shape
        valid_mask = self._to_valid_mask(op_mask, T, B, device)
        x_start = torch.randn((B, T, dim), device=device)
        x_t = x_start
        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)  # [B, T, hidden_dim]
        enc_emb = self.enc_embedding(hist_enc[:, -1, :]) # [B, D]

        infer_scheduler = self._build_inference_scheduler()
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((B,), t_scalar, device=device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            y = t_emb + enc_emb
            input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
            pred_x0_norm = self.dit(x=input_embedded, y=y, cross=context)
            if self.x0_clip is not None:
                pred_x0_norm = torch.clamp(pred_x0_norm, -self.x0_clip, self.x0_clip)
            try:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        pred = self.denorm(x_t)
        pred_norm = self.norm(pred)
        future_norm = self.norm(future)
        loss = self.compute_motion_loss(pred_norm, future_norm, valid_mask)
        ade, fde = self.compute_ade_fde(pred, future, valid_mask)

        # ===== 可视化调试代码（注释本段即可关闭）=====
        # if self.is_main_process:
        #     vis_batch_idx = 0
        #     vis_ade, vis_fde = self.compute_single_ade_fde(pred, future, valid_mask, batch_idx=vis_batch_idx)
        #     metrics = {
        #         "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
        #         "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        #     }
        #     visualize_batch_trajectories(
        #         hist=hist,
        #         hist_nbrs=None,
        #         temporal_mask=None,
        #         future=future,
        #         pred=pred,
        #         future_mask=valid_mask,
        #         batch_idx=vis_batch_idx,
        #         save_path=None,   # 不保存，只show
        #         metrics=metrics,
        #         input_unit='ft',
        #         show_plot=True
        #     )
        #     print(
        #         f"[Eval][Vis Traj idx={vis_batch_idx}] "
        #         f"ADE: {vis_ade.item():.4f} ft ({vis_ade.item() * self.meter_per_foot:.4f} m), "
        #         f"FDE: {vis_fde.item():.4f} ft ({vis_fde.item() * self.meter_per_foot:.4f} m)"
        #     )
        # ===== 可视化调试代码结束 =====

        return loss, pred, ade, fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        C = x_norm.shape[-1]
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -5.0, 5.0)
        if C >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        C = x.shape[-1]
        if C >= 4:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean  # v, a
        return x_denorm
