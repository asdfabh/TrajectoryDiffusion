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

    def computeLoss(self, pred_x0, target_x0, valid_mask, return_parts=False):
        loss_map = F.smooth_l1_loss(pred_x0, target_x0, reduction="none")
        valid = valid_mask.unsqueeze(-1)
        numer = (loss_map * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        loss = (numer / denom).mean()

        if not return_parts:
            return loss

        parts = {
            "loss_total": loss.detach(),
            "loss_diffusion": loss.detach(),
            "loss_x0": loss.detach(),
        }
        return loss, parts

    # 多模态训练
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device)
        target_x0 = self.norm(future)

        target_x0 = target_x0.unsqueeze(1).repeat(1, self.fut_k, 1, 1).reshape(bsz * self.fut_k, t_len, self.output_dim)
        valid_mask = valid_mask.unsqueeze(1).repeat(1, self.fut_k, 1).reshape(bsz * self.fut_k, t_len)

        noise = torch.randn_like(target_x0)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz * self.fut_k,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_x0, noise, timesteps)

        context_tokens, _ = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        context_tokens = context_tokens.repeat_interleave(self.fut_k, dim=0)

        t_emb = self.timestep_embedder(timesteps)
        input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
        pred_x0 = self.dit(input_embedded, t_emb, context_tokens)

        loss, loss_parts = self.computeLoss(pred_x0, target_x0, valid_mask, return_parts=True)
        if return_components:
            return loss, loss_parts
        return loss

    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=None):
        bsz, t_len, _ = future.shape
        k = self.fut_k if K is None else max(1, int(K))
        future_phys = future[..., :self.output_dim]

        context_tokens, _ = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        context_tokens = context_tokens.repeat_interleave(k, dim=0)

        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        x_t = torch.randn((bsz * k, t_len, self.input_dim), device=device)
        pred_x0 = None
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((x_t.size(0),), t_scalar, device=x_t.device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
            pred_x0 = self.dit(input_embedded, t_emb, context_tokens)
            x_t = infer_scheduler.step(pred_x0, t, x_t).prev_sample

        pred_x0 = pred_x0.view(bsz, k, t_len, self.output_dim)
        pred_phys = self.denorm(pred_x0)
        all_preds = future_phys.unsqueeze(1).repeat(1, k, 1, 1).clone()
        all_preds[..., :2] = pred_phys[..., :2]
        return all_preds

    @torch.no_grad()
    # 单模态推理接口，兼容旧调用链。
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        return self.forwardEvalMulti(hist, hist_nbrs, mask, temporal_mask, future, device, K=1).squeeze(1)

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=return_components)

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
