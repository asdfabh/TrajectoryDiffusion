from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
import torch.nn.functional as F
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.traj_vis_metrics import visualize_hist_nbrs_fut_pred
from method_diffusion.utils.traj_metrics import compute_batch_metrics
from pathlib import Path

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
        self.dropout = args.dropout_fut
        self.depth = int(args.depth_fut)
        self.mlp_ratio = args.mlp_ratio_fut
        self.num_train_timesteps = args.num_train_timesteps_fut
        self.time_embedding_size = args.time_embedding_size_fut
        self.num_inference_steps = args.num_inference_steps
        self.T = int(args.T_f)
        self.loss_time_power = float(getattr(args, "loss_time_power_fut", 2.0))
        self.loss_end_weight = float(getattr(args, "loss_end_weight_fut", 2.0))
        self.loss_vel_weight = float(getattr(args, "loss_vel_weight_fut", 0.2))
        if self.hidden_dim != self.input_dim:
            raise ValueError(
                f"hidden_dim_fut ({self.hidden_dim}) must equal input_dim_fut ({self.input_dim}) for DiT."
            )

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

        self.norm_config_path = Path(__file__).resolve().parent.parent / "dataset/ngsim_stats.npz"
        if self.norm_config_path.exists():
            norm_config = np.load(self.norm_config_path)
            self.register_buffer("pos_mean", torch.from_numpy(norm_config["pos_mean"]).float(), persistent=False)
            self.register_buffer("pos_std", torch.from_numpy(norm_config["pos_std"]).float(), persistent=False)
            self.register_buffer("va_mean", torch.from_numpy(norm_config["va_mean"]).float(), persistent=False)
            self.register_buffer("va_std", torch.from_numpy(norm_config["va_std"]).float(), persistent=False)
        else:
            # Fallback to precomputed NGSIM stats (hist + nbrs + fut).
            self.register_buffer("pos_mean", torch.tensor([0.03300306, -15.91495069]).float(), persistent=False)
            self.register_buffer("pos_std", torch.tensor([8.8865948, 68.81046473]).float(), persistent=False)
            self.register_buffer("va_mean", torch.tensor([21.15030837, 0.00604141]).float(), persistent=False)
            self.register_buffer("va_std", torch.tensor([13.59830645, 4.5057365]).float(), persistent=False)

    def _build_mask(self, tensor_like, op_mask=None):
        B, T = tensor_like.shape[0], tensor_like.shape[1]
        if op_mask is None:
            return torch.ones(B, T, 1, device=tensor_like.device, dtype=tensor_like.dtype)

        if op_mask.dim() == 3:
            mask = op_mask[..., :1]
        elif op_mask.dim() == 2:
            mask = op_mask.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported op_mask dim: {op_mask.dim()}")

        mask = mask.to(device=tensor_like.device, dtype=tensor_like.dtype).clamp(0, 1)
        if mask.shape[1] < T:
            pad_t = T - mask.shape[1]
            mask = F.pad(mask, (0, 0, 0, pad_t))
        return mask[:, :T, :]

    def _compute_ade_fde(self, pred, target, op_mask=None):
        metrics = compute_batch_metrics(
            pred=pred,
            target=target,
            op_mask=op_mask,
            t_max=min(pred.shape[1], target.shape[1]),
            unit_conversion=1.0,
        )
        return metrics["ade"], metrics["fde"]

    def _build_time_weights(self, tensor_like):
        t_len = tensor_like.shape[1]
        if self.loss_time_power <= 0:
            return torch.ones(1, t_len, 1, device=tensor_like.device, dtype=tensor_like.dtype)

        t = torch.arange(1, t_len + 1, device=tensor_like.device, dtype=tensor_like.dtype)
        t = t / max(float(t_len), 1.0)
        w = torch.pow(t, self.loss_time_power)
        w = w / w.mean().clamp(min=1e-6)
        return w.view(1, t_len, 1)

    def compute_motion_loss(self, pred, target, op_mask=None):
        """
        pred: [B, T, D]
        target: [B, T, D]
        """
        mask = self._build_mask(pred, op_mask)
        time_w = self._build_time_weights(pred)
        weighted_mask = mask * time_w

        pos_se = torch.sum((pred[..., :2] - target[..., :2]) ** 2, dim=-1, keepdim=True)
        loss_pos = (pos_se * weighted_mask).sum() / weighted_mask.sum().clamp(min=1e-6)

        pos_se_2d = pos_se.squeeze(-1)
        valid_mask = mask.squeeze(-1)
        valid_per_sample = valid_mask.sum(dim=1)
        has_valid = (valid_per_sample > 0).float()
        last_valid_idx = (valid_per_sample.long() - 1).clamp(min=0)
        batch_idx = torch.arange(pred.shape[0], device=pred.device)
        end_se = pos_se_2d[batch_idx, last_valid_idx]
        loss_end = (end_se * has_valid).sum() / has_valid.sum().clamp(min=1.0)

        pred_vel = pred[..., :2][:, 1:, :] - pred[..., :2][:, :-1, :]
        target_vel = target[..., :2][:, 1:, :] - target[..., :2][:, :-1, :]
        vel_se = torch.sum((pred_vel - target_vel) ** 2, dim=-1, keepdim=True)
        vel_mask = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1)
        vel_time_w = time_w[:, 1:, :]
        vel_weighted_mask = vel_mask * vel_time_w
        if vel_weighted_mask.sum().item() > 0:
            loss_vel = (vel_se * vel_weighted_mask).sum() / vel_weighted_mask.sum().clamp(min=1e-6)
        else:
            loss_vel = torch.zeros((), device=pred.device, dtype=pred.dtype)

        return loss_pos + self.loss_end_weight * loss_end + self.loss_vel_weight * loss_vel

    # hist: [B, T, dim], hist_masked: [B, T, dim+1]
    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device, op_mask=None):
        B, T, dim = future.shape
        if dim != self.feature_dim:
            raise ValueError(
                f"future feature dim mismatch: expected {self.feature_dim}, got {dim}"
            )
        future_norm = self.norm(future)  # [B, T, 2]
        x_start = future_norm
        noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
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
        loss = self.compute_motion_loss(pred_x0, future_norm, op_mask=op_mask)
        pred = self.denorm(pred_x0)

        ade, fde = self._compute_ade_fde(pred, future, op_mask=op_mask)

        # visualize_hist_nbrs_fut_pred(
        #     hist=hist,
        #     nbrs=hist_nbrs,
        #     fut=future,
        #     pred=pred,
        #     op_mask=op_mask,
        #     sample_index=0,
        #     save_path=None,  # always show
        #     title_prefix="fut_model.forward_train",
        #     temporal_mask=temporal_mask,
        # )
        return loss, pred, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device, op_mask=None):
        B, T, dim = future.shape
        if dim != self.feature_dim:
            raise ValueError(
                f"future feature dim mismatch: expected {self.feature_dim}, got {dim}"
            )
        future_norm = self.norm(future)
        x_start = torch.randn((B, T, dim), device=device)
        x_t = x_start
        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)  # [B, T, hidden_dim]
        enc_emb = self.enc_embedding(hist_enc[:, -1, :]) # [B, D]

        self.diffusion_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.diffusion_scheduler.timesteps:
            timesteps = torch.full((B,), t, device=device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            y = t_emb + enc_emb
            input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
            pred_x0_norm = self.dit(x=input_embedded, y=y, cross=context)
            x_t = self.diffusion_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        pred = self.denorm(x_t)
        loss = self.compute_motion_loss(x_t, future_norm, op_mask=op_mask)
        ade, fde = self._compute_ade_fde(pred, future, op_mask=op_mask)

        # visualize_hist_nbrs_fut_pred(
        #     hist=hist,
        #     nbrs=hist_nbrs,
        #     fut=future,
        #     pred=pred,
        #     op_mask=op_mask,
        #     sample_index=0,
        #     save_path=None,  # always show
        #     title_prefix="fut_model.forward_eval",
        #     temporal_mask=temporal_mask,
        # )
        return loss, pred, ade, fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, device, op_mask=None):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_nbrs, mask, temporal_mask, future, device, op_mask=op_mask)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        C = x_norm.shape[-1]
        if C >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -5.0, 5.0)
        if C >= 4:
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        C = x.shape[-1]
        if C >= 4:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean  # v, a
        return x_denorm
