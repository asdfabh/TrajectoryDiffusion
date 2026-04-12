from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import gen_sineembed_for_position


class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args
        self.dataset_name = str(args.dataset).strip().lower()

        # 模型结构参数：控制 DiT 主干维度、层数和 future 序列长度。
        self.hidden_dim = int(args.hidden_dim_fut)
        self.input_dim = int(args.input_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.depth = int(args.depth_fut)
        self.dropout = float(args.dropout_fut)
        self.mlp_ratio = int(args.mlp_ratio_fut)
        self.time_embedding_size = int(args.time_embedding_size_fut)
        self.T = int(args.T_f)
        self.fut_k = max(1, int(args.fut_k))
        self.traj_dim = self.T * self.input_dim
        self.point_embed_dim = 64

        # 扩散与推理参数。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)

        # 输入编码模块：将物理轨迹位置编码后，压缩为 mode token。
        self.plan_anchor_encoder = nn.Sequential(
            nn.Linear(self.T * self.point_embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.hist_encoder = HistEncoder(args)

        # DiT 主干与扩散调度器。
        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.hidden_dim, self.heads, self.dropout, self.mlp_ratio)
        final_layer = dit.FinalLayer(self.hidden_dim, self.fut_k, self.traj_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=self.depth)

        # 仅对 Ego future 做归一化。
        if self.dataset_name == "ngsim":
            self.register_buffer("xy_mean", torch.tensor([-0.0606, 65.2935], dtype=torch.float32), persistent=False)
            self.register_buffer("xy_std", torch.tensor([1.3011, 56.2487], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("xy_mean", torch.tensor([0.0654, 221.1319], dtype=torch.float32), persistent=False)
            self.register_buffer("xy_std", torch.tensor([1.3484, 142.1689], dtype=torch.float32), persistent=False)
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}' for fut normalization. Supported: highd, ngsim")

        anchor_path = Path(__file__).resolve().parent.parent / "dataset" / "anchor" / f"{self.dataset_name}_k{self.fut_k}.pt"
        plan_anchor = torch.load(anchor_path, map_location="cpu")
        self.plan_anchor = nn.Parameter(plan_anchor.float(), requires_grad=False)

    # 采用DiffusionDrive的anchor分配方式：先找离GT最近的anchor，再只对该mode回传回归损失。
    # pred_x0 [B,K,T,D]; target_x0 [B,T,D]; anchor_x0 [B,K,T,D]; valid_mask [B,T]
    def computeLoss(self, pred_x0, target_x0, anchor_x0, valid_mask):
        bsz, _, t_len, coord_dim = pred_x0.shape
        valid_time = valid_mask.unsqueeze(1) # [B,1,T]
        dist = torch.linalg.norm(target_x0.unsqueeze(1) - anchor_x0, dim=-1) # [B,K,T]
        numer = (dist * valid_time).sum(dim=-1) # [B,K]
        denom = valid_time.sum(dim=-1) + 1e-6 # [B,1]
        mode_idx = torch.argmin(numer / denom, dim=-1) # [B]

        gather_index = mode_idx.view(bsz, 1, 1, 1).expand(-1, 1, t_len, coord_dim)
        best_pred = torch.gather(pred_x0, 1, gather_index).squeeze(1) # [B,T,D]
        valid = valid_mask.unsqueeze(-1).expand(-1, -1, coord_dim) # [B,T,D]
        loss_map = F.l1_loss(best_pred, target_x0, reduction="none") # [B,T,D]
        numer = (loss_map * valid).sum(dim=(1, 2)) # [B]
        denom = valid.sum(dim=(1, 2)) + 1e-6 # [B]
        loss = (numer / denom).mean()
        logs = {"loss_x0": loss.detach()}
        return loss, logs

    # 多模态训练
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device)  # [B, T]
        target_x0 = future  # [B, T, D]

        # 先归一化加噪，再反归一化做物理位置编码，与DiffusionDrive保持一致。
        anchor_x0 = self.plan_anchor.to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1) # [B,K,T,D]
        anchor_x0_norm = self.norm(anchor_x0)
        noise = torch.randn_like(anchor_x0_norm) # [B,K,T,D]
        timesteps = torch.randint(0, 50, (bsz,), device=device).long() # [B]
        x_t_norm = self.diffusion_scheduler.add_noise(anchor_x0_norm, noise, timesteps).float() # [B,K,T,D]
        x_t_norm = torch.clamp(x_t_norm, min=-5.0, max=5.0)
        x_t_phys = self.denorm(x_t_norm) # [B,K,T,D]

        traj_pos_embed = gen_sineembed_for_position(x_t_phys, hidden_dim=self.point_embed_dim) # [B,K,T,C]
        traj_pos_embed = traj_pos_embed.flatten(-2) # [B,K,T*C]
        traj_feature = self.plan_anchor_encoder(traj_pos_embed) # [B,K,128]
        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask) # [B,T_ctx,128]

        t_emb = self.timestep_embedder(timesteps) # [B,128]
        pred_delta = self.dit(traj_feature, t_emb, context_tokens) # [B,K,T*D]
        pred_delta = pred_delta.view(bsz, self.fut_k, t_len, self.output_dim) # [B,K,T,D]
        pred_x0 = x_t_phys + pred_delta # 预测物理空间下相对noisy anchor的修正量 # [B,K,T,D]
        loss, loss_logs = self.computeLoss(pred_x0, target_x0, anchor_x0, valid_mask)
        return loss, loss_logs

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=None):
        bsz, t_len, _ = future.shape
        k = self.fut_k if K is None else max(1, int(K))

        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)

        diffusion_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        diffusion_scheduler.set_timesteps(self.num_train_timesteps, device=device)
        step_ratio = 30 / self.num_inference_steps
        roll_timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        trunc_timesteps = torch.full((bsz,), 20, device=device, dtype=torch.long)
        anchor_x0 = self.plan_anchor[:k].to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1)
        anchor_x0_norm = self.norm(anchor_x0)
        noise = torch.randn_like(anchor_x0_norm)
        x_t = diffusion_scheduler.add_noise(anchor_x0_norm, noise, trunc_timesteps).float()

        pred_x0 = None
        for t in roll_timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            x_boxes = torch.clamp(x_t, min=-5.0, max=5.0)
            noisy_traj_points = self.denorm(x_boxes)
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=self.point_embed_dim)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            pred_delta = self.dit(traj_feature, t_emb, context_tokens)
            pred_delta = pred_delta.view(bsz, k, t_len, self.output_dim)
            pred_x0 = noisy_traj_points + pred_delta
            pred_x0_norm = self.norm(pred_x0)
            x_t = diffusion_scheduler.step(pred_x0_norm, t, x_t).prev_sample

        pred_phys = pred_x0
        all_preds = future.unsqueeze(1).repeat(1, k, 1, 1).clone()
        all_preds[..., :2] = pred_phys[..., :2]
        return all_preds

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    # 归一化与反归一化，仅处理 future xy。
    def norm(self, x):
        x_norm = x.clone()
        mean = self.xy_mean.to(device=x.device, dtype=x.dtype)
        std = self.xy_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        x_norm[..., 0:2] = (x[..., 0:2] - mean) / std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        mean = self.xy_mean.to(device=x.device, dtype=x.dtype)
        std = self.xy_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        x_denorm[..., 0:2] = x[..., 0:2] * std + mean
        return x_denorm
