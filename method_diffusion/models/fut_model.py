"""
DiffusionFut — future trajectory prediction via denoising diffusion.

Design (post-refactor):
  - HistEncoder: encodes raw physical hist + nbrs (NO normalization) into context tokens.
  - TrajectoryDenoiser: MMPD-style lightweight epsilon predictor (adaLN + temporal Conv1d + cross-attn).
  - Diffusion target: normalized frame-to-frame velocity (displacement), ~N(0,1).
  - Prediction type: epsilon (predict noise), following MMPD's recommendation.
  - Loss: MSE(eps_pred, eps_true) in normalized space, masked by valid future frames.
  - No planning heads (intent / bridge / align modules removed).
  - Inference: DDIM reverse process → denormalize velocity → cumsum → physical trajectory.
"""

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models.denoiser import TrajectoryDenoiser
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.gmm import DiffusionVariationalGaussianMixture, multi_mode_summary
from method_diffusion.utils.visualization import maybe_visualize_future_prediction


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()

        # Model dimensions
        self.hidden_dim = int(args.hidden_dim_fut)
        self.input_dim = int(args.input_dim_fut)    # trajectory channels for diffusion (2: x,y)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.depth = int(args.depth_fut)
        self.dropout = float(args.dropout_fut)
        self.mlp_ratio = int(args.mlp_ratio_fut)
        self.time_embedding_size = int(args.time_embedding_size_fut)
        self.T = int(args.T_f)  # future sequence length

        # Diffusion schedule params
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.n_gmm_modes = int(getattr(args, "n_gmm_modes", 3))
        self.gmm_K       = int(getattr(args, "gmm_K", 10))

        # History encoder: accepts raw physical coordinates (feature_dim=6: x,y,v,a,lane,class)
        self.hist_encoder = HistEncoder(args)

        # context_dim = 2 * encoder_input_dim (HistEncoder returns [temporal+social ; hist_enc])
        context_dim = 2 * int(args.encoder_input_dim)

        # Lightweight MMPD-aligned denoiser v2 (epsilon prediction, per-frame condition)
        self.denoiser = TrajectoryDenoiser(
            input_dim=self.input_dim,
            hidden_size=self.hidden_dim,
            context_dim=context_dim,
            T_f=self.T,
            depth=self.depth,
            num_heads=self.heads,
            freq_embed_size=self.time_embedding_size,
            dropout=self.dropout,
            radius=int(getattr(args, "radius_fut", 2)),
        )

        # DDIM scheduler with epsilon prediction
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )

        # Velocity normalization statistics for diffusion target (fit on training set)
        if self.dataset_name == "ngsim":
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std",  torch.tensor([0.1502223350250087,   2.951254134709027], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("vel_mean", torch.tensor([0.004845835373614644, 17.01558226555126], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std",  torch.tensor([0.10621210903901461,   4.838376260255577], dtype=torch.float32), persistent=False)
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}'. Supported: ngsim, highd")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def toValidMask(op_mask, device):
        """Convert output-channel mask to per-frame validity mask."""
        return (op_mask[..., 0] > 0.5).float().to(device)

    # ------------------------------------------------------------------ forward
    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        """Encode raw physical hist + nbrs into context tokens (NO normalization)."""
        context, _ = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        return context  # [B, T_hist, 2*encoder_input_dim]

    def predictNoise(self, x_t, timesteps, context):
        """Run denoiser: predict epsilon from noisy velocity x_t given history context."""
        return self.denoiser(x_t, timesteps, context)

    # ------------------------------------------------------------------ training
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        """
        Single-batch training forward pass.

        Returns:
            loss (scalar) — MSE between predicted and true noise in normalized velocity space.
            loss_parts (dict) — breakdown for logging, only when return_components=True.
        """
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, device)  # [B, T_f]

        # ---- build diffusion target: normalized frame-to-frame velocity ----
        anchor_phys = hist[:, -1:, :self.output_dim]          # [B, 1, 2]  last hist position ≈ (0,0)
        future_phys = future[..., :self.output_dim]            # [B, T_f, 2]
        shifted      = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted                # [B, T_f, 2]  frame-to-frame displacement

        std_vel  = self.vel_std.view(1, 1, 2)
        mean_vel = self.vel_mean.view(1, 1, 2)
        target_vel_norm = (target_vel_phys - mean_vel) / std_vel
        target_vel_norm = torch.clamp(target_vel_norm, -5.0, 5.0)  # clip outliers

        # ---- add noise ----
        noise     = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t       = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        # ---- encode history context (raw physical, no normalization) ----
        context = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)

        # ---- predict noise (epsilon) ----
        eps_pred = self.predictNoise(x_t, timesteps, context)

        # ---- loss: MSE in normalized space, masked by valid frames ----
        loss_eps = F.mse_loss(eps_pred, noise, reduction="none")   # [B, T_f, 2]
        valid    = valid_mask.unsqueeze(-1)                         # [B, T_f, 1]
        loss     = (loss_eps * valid).sum() / (valid.sum() * self.input_dim + 1e-6)

        # ---- visualization (only when flag is set — runs a quick DDIM pass) ----
        if bool(getattr(self.args, "fut_enable_train_vis", 0)):
            with torch.no_grad():
                infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
                infer_scheduler.set_timesteps(self.num_inference_steps)
                pred_vel_norm = self.sampleFromNoise(bsz, t_len, context, infer_scheduler, device)
                pred_vel_phys = pred_vel_norm * std_vel + mean_vel
                pred_pos      = torch.cumsum(pred_vel_phys, dim=1) + hist[:, -1:, :self.output_dim]
                pred_vis      = future_phys.clone()
                pred_vis[..., :2] = pred_pos
            maybe_visualize_future_prediction(
                hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask,
                future=future, pred=pred_vis, valid_mask=valid_mask,
                stage="train", enable_train_vis=True,
            )

        if not return_components:
            return loss

        parts = {
            "loss_total": loss.detach(),
            "loss_vel":   loss.detach(),    # same thing; kept for logging compat
        }
        return loss, parts

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def sampleFromNoise(self, bsz, t_len, context, infer_scheduler, device):
        """DDIM reverse process: Gaussian noise → predicted normalized velocity."""
        x_t = torch.randn((bsz, t_len, self.input_dim), device=device)
        for t in infer_scheduler.timesteps:
            t_scalar  = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=device, dtype=torch.long)
            eps_pred  = self.predictNoise(x_t, timesteps, context)
            try:
                x_t = infer_scheduler.step(eps_pred, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(eps_pred, t, x_t).prev_sample
        # x_t at t=0 is the predicted target_vel_norm (denoised sample)
        return x_t

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        """Single-sample inference evaluation."""
        bsz, t_len, _ = future.shape
        valid_mask  = self.toValidMask(op_mask, device)
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        context = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        pred_vel_norm = self.sampleFromNoise(bsz, t_len, context, infer_scheduler, device)

        std_vel  = self.vel_std.view(1, 1, 2)
        mean_vel = self.vel_mean.view(1, 1, 2)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys

        pred_phys_abs         = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        return pred_phys_abs, ade, fde

    @torch.no_grad()
    def forwardEval_multimodal(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=None):
        """
        Multimodal inference using MMPD's Variational GMM updated at every DDIM step.

        At each reverse-diffusion step the GMM E-M is run on the current K noisy
        samples (exactly as in MMPD p_sample_loop_progressive).  After the final
        step a last round of E-M determines cluster assignments, and
        multi_mode_summary() computes the per-mode median trajectory and count.

        Recommended K >= 20 for meaningful cluster coverage (MMPD uses 100+).

        Returns:
            best_pred:   [B, T_f, 2]          highest-probability mode trajectory
            ade, fde:    scalars              metrics vs GT
            mode_trajs:  [B, n_modes, T_f, 2] modes sorted by probability descending
            mode_probs:  [B, n_modes]          probability = cluster_count / K
        """
        bsz, t_len, _ = future.shape
        K = K if K is not None else self.gmm_K
        n_modes = self.n_gmm_modes
        valid_mask  = self.toValidMask(op_mask, device)
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        context = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        # Expand context for K parallel samples
        context_k = context.repeat_interleave(K, dim=0)   # [B*K, T_hist, H]

        # Initialise x_T ~ N(0, I) in normalised-velocity space
        x_t = torch.randn((bsz * K, t_len, self.input_dim), device=device)  # [B*K, T_f, 2]

        # alphas_cumprod from the training scheduler (full [num_train_timesteps] array)
        alphas_cumprod = self.diffusion_scheduler.alphas_cumprod.to(device)  # [T_train]

        # Initialise variational GMM at t=T (pure noise)
        x_flat = x_t.reshape(bsz, K, t_len * self.input_dim)   # [B, K, T_f*2]
        gmm = DiffusionVariationalGaussianMixture(
            n_components=n_modes,
            alphas_cumprod=alphas_cumprod,
            prior_pi_decay=0.5,
            prior_precision_shape=1e3,
            batch_x=x_flat,
        )

        # DDIM reverse loop with per-step GMM E-M update
        gmm_iter = 10   # E-M iterations per diffusion step (MMPD default)
        last_t = 0
        for t_step in infer_scheduler.timesteps:
            t_scalar = int(t_step.item()) if isinstance(t_step, torch.Tensor) else int(t_step)
            last_t = t_scalar

            # GMM update on current noisy samples (MMPD pattern)
            x_flat = x_t.reshape(bsz, K, t_len * self.input_dim)
            for _ in range(gmm_iter):
                log_resp = gmm.e_step(x_flat)
                gmm.m_step(x_flat, log_resp, t_scalar)

            # DDIM denoise step
            t_batch = torch.full((bsz * K,), t_scalar, device=device, dtype=torch.long)
            eps_pred = self.predictNoise(x_t, t_batch, context_k)
            try:
                x_t = infer_scheduler.step(eps_pred, t_step, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(eps_pred, t_step, x_t).prev_sample

        # Final E-M on denoised samples (x_0)
        x_flat = x_t.reshape(bsz, K, t_len * self.input_dim)
        for _ in range(gmm_iter):
            log_resp = gmm.e_step(x_flat)
            gmm.m_step(x_flat, log_resp, last_t)

        # Cluster assignment and mode summary
        log_resp = gmm.predict(x_flat)                   # [B, K, n_modes]
        assigned = log_resp.argmax(-1)                   # [B, K]

        num_in_mode, mode_median_flat, _ = multi_mode_summary(
            x_flat, assigned, num_modes=n_modes
        )  # [B, M], [B, M, T_f*2]

        mode_probs = num_in_mode.float() / K             # [B, n_modes]  (sum ≈ 1)

        # Denormalise: normalised velocity → physical positions
        std_vel  = self.vel_std.view(1, 1, 2)
        mean_vel = self.vel_mean.view(1, 1, 2)

        # mode_median_flat: [B, n_modes, T_f*2] → [B, n_modes, T_f, 2]
        mode_vel_norm  = mode_median_flat.reshape(bsz, n_modes, t_len, self.input_dim)
        mode_vel_phys  = mode_vel_norm * std_vel + mean_vel
        mode_pos_phys  = torch.cumsum(mode_vel_phys, dim=2) + anchor_phys.unsqueeze(1)  # [B, M, T_f, 2]

        mode_trajs = mode_pos_phys                        # [B, n_modes, T_f, 2]

        # best_pred = highest-probability mode (index 0, already sorted by multi_mode_summary)
        best_pred_xy = mode_trajs[:, 0, :, :]            # [B, T_f, 2]
        best_pred    = future_phys.clone()
        best_pred[..., :2] = best_pred_xy

        self.last_minade_mode_trajs = mode_trajs.detach()
        self.last_minade_mode_probs = mode_probs.detach()

        ade, fde = self.computeAdeFde(best_pred, future, valid_mask)

        maybe_visualize_future_prediction(
            hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask,
            future=future, pred=best_pred, valid_mask=valid_mask,
            stage="eval", enable_eval_vis=bool(getattr(self.args, "fut_enable_eval_vis", 0)),
            pred_all=mode_trajs, mode_probs=mode_probs,
        )

        return best_pred, ade, fde, mode_trajs, mode_probs

    # ------------------------------------------------------------------ metrics
    @staticmethod
    def computeAdeFde(pred, target, valid_mask):
        diff   = pred[..., :2] - target[..., :2]
        dist   = torch.norm(diff, dim=-1)
        ade    = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        valid_counts = valid_mask.sum(dim=1).long()
        has_valid    = valid_counts > 0
        last_idx     = torch.clamp(valid_counts - 1, min=0)
        final_dist   = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        fde          = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
        return ade, fde

    # ------------------------------------------------------------------ unified entry
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        return self.forwardTrain(
            hist, hist_nbrs, mask, temporal_mask, future, op_mask, device,
            return_components=return_components,
        )
