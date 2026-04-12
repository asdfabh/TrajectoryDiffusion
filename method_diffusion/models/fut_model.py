from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding


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

        # 扩散与推理参数。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)

        # 输入编码模块：future 噪声序列和 history context。
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
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
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
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

    # best of K赢者通吃
    # pred_x0 [B*K,T,D]; target_x0 [B,T,D]; valid_mask [B,T]
    def computeLoss(self, pred_x0, target_x0, valid_mask, bsz, k):
        pred_x0 = pred_x0.view(bsz, k, self.T, self.output_dim) # [B,K,T,D]
        target_x0 = target_x0.unsqueeze(1).expand(-1, k, -1, -1) # [B,K,T,D]
        valid = valid_mask.unsqueeze(1).unsqueeze(-1).expand(-1, k, -1, self.output_dim) # [B,K,T,D]
        loss_map = F.smooth_l1_loss(pred_x0, target_x0, reduction="none") # [B,K,T,D]
        numer = (loss_map * valid).sum(dim=(2, 3)) # [B,K]
        denom = valid.sum(dim=(2, 3)) + 1e-6 # [B,K]
        loss_per_traj = numer / denom
        best_loss, _ = torch.min(loss_per_traj, dim=1) # [B]
        loss = best_loss.mean()
        logs = {"loss_x0": loss.detach()}
        return loss, logs

    # 多模态训练
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device)  # [B, T]
        target_x0 = self.norm(future)  # [B, T, D]

        # 对anchor进行加噪，保持[B,K,T,D]维度
        anchor_x0 = self.plan_anchor.to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1) # [B,K,T,D]
        anchor_x0 = self.norm(anchor_x0)
        noise = torch.randn_like(anchor_x0) # [B,K,T,D]
        timesteps = torch.randint(0, 50, (bsz,), device=device).long() # [B]
        x_t = self.diffusion_scheduler.add_noise(anchor_x0, noise, timesteps).float() # [B,K,T,D]
        # 维度转化为[B*K,T,D]
        timesteps = timesteps.unsqueeze(1).expand(-1, self.fut_k).reshape(bsz * self.fut_k)  # [B*K]
        x_t = x_t.reshape(bsz * self.fut_k, t_len, self.output_dim) # [B*K,T,D]
        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask) # [B,T,D]
        context_tokens = context_tokens.repeat_interleave(self.fut_k, dim=0)  # [B*K, T, D]

        t_emb = self.timestep_embedder(timesteps) # [B*K]
        input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t) # [B*K,T,D]
        pred_delta = self.dit(input_embedded, t_emb, context_tokens) # [B*K,T,D]
        pred_x0 = x_t + pred_delta # 预测在anchor下需要的修正量，得到轨迹 # [B*K,T,D]
        loss, loss_logs = self.computeLoss(pred_x0, target_x0, valid_mask, bsz, self.fut_k)
        return loss, loss_logs

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=None):
        bsz, t_len, _ = future.shape
        k = self.fut_k if K is None else max(1, int(K))

        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        context_tokens = context_tokens.repeat_interleave(k, dim=0)

        diffusion_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        diffusion_scheduler.set_timesteps(self.num_train_timesteps, device=device)
        step_ratio = 30 / self.num_inference_steps
        roll_timesteps = (np.arange(0, self.num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        trunc_timesteps = torch.full((bsz,), 20, device=device, dtype=torch.long)
        anchor_x0 = self.plan_anchor[:k].to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1)
        anchor_x0 = self.norm(anchor_x0)
        noise = torch.randn_like(anchor_x0)
        x_t = diffusion_scheduler.add_noise(anchor_x0, noise, trunc_timesteps).float()
        x_t = x_t.reshape(bsz * k, t_len, self.output_dim)

        pred_x0 = None
        for t in roll_timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((x_t.size(0),), t_scalar, device=x_t.device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
            pred_delta = self.dit(input_embedded, t_emb, context_tokens)
            pred_x0 = x_t + pred_delta
            x_t = diffusion_scheduler.step(pred_x0, t, x_t).prev_sample

        pred_x0 = pred_x0.view(bsz, k, t_len, self.output_dim)
        pred_phys = self.denorm(pred_x0)
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
