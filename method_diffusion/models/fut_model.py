import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.fut_utils import build_future_traj_pos_embed


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

        # 输入编码模块：先对物理坐标做点级位置编码，再将整条轨迹展平成 mode token。
        self.input_embedding = nn.Sequential(
            nn.LayerNorm(self.T * self.hidden_dim),
            nn.Linear(self.T * self.hidden_dim, self.hidden_dim),
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
        final_layer = dit.FinalLayer(self.hidden_dim, self.fut_k, self.T * self.output_dim)
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

    # best of K赢者通吃
    def computeLoss(self, pred_x0, target_x0, valid_mask):
        loss_map = F.smooth_l1_loss(pred_x0, target_x0, reduction="none")
        valid = valid_mask.unsqueeze(-1)
        numer = (loss_map * valid).sum(dim=(2, 3))
        denom = valid_mask.sum(dim=2) + 1e-6
        loss_per_traj = numer / denom
        best_loss, _ = torch.min(loss_per_traj, dim=1)
        loss = best_loss.mean()
        logs = {"loss_x0": loss.detach()}
        return loss, logs

    # 多模态训练
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device)  # [B, T]
        target_x0 = self.norm(future)  # [B, T, D]
        # 对每条 GT 复制 K 份，K 个分支共享同一个扩散步 t，但各自采样独立噪声。
        # 进入 DiT 前将整条 future 轨迹展平为 mode token：[B, K, T, D] -> [B, K, T*D]。
        target_x0 = target_x0.unsqueeze(1).repeat(1, self.fut_k, 1, 1)  # [B, K, T, D]
        valid_mask = valid_mask.unsqueeze(1).repeat(1, self.fut_k, 1)  # [B, K, T]
        noise = torch.randn_like(target_x0)  # [B, K, T, D]
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()  # [B]
        # add_noise会自动按 batch 维广播 timestep 到 [B, K, T, D]。
        x_t = self.diffusion_scheduler.add_noise(target_x0, noise, timesteps)  # [B, K, T, D]

        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)

        t_emb = self.timestep_embedder(timesteps)
        x_t_phys = self.denorm(x_t)[..., :2]
        x_t_pos_embed = build_future_traj_pos_embed(x_t_phys, hidden_dim=self.hidden_dim)
        input_embedded = self.input_embedding(x_t_pos_embed.flatten(start_dim=2))
        pred_x0 = self.dit(input_embedded, t_emb, context_tokens).reshape(bsz, self.fut_k, t_len, self.output_dim)

        loss, loss_logs = self.computeLoss(pred_x0, target_x0, valid_mask)
        return loss, loss_logs

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=None):
        bsz, t_len, _ = future.shape
        k = self.fut_k if K is None else max(1, int(K))

        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)

        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        x_t = torch.randn((bsz, k, t_len, self.input_dim), device=device)
        pred_x0 = None
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            x_t_phys = self.denorm(x_t)[..., :2]
            x_t_pos_embed = build_future_traj_pos_embed(x_t_phys, hidden_dim=self.hidden_dim)
            input_embedded = self.input_embedding(x_t_pos_embed.flatten(start_dim=2))
            pred_x0 = self.dit(input_embedded, t_emb, context_tokens).reshape(bsz, k, t_len, self.output_dim)
            x_t = infer_scheduler.step(pred_x0, t, x_t).prev_sample

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
