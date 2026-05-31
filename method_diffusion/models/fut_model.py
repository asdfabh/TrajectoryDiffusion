from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.schedulers import DDIMScheduler

from method_diffusion.models import dit_fut as dit
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.fut_utils import build_eval_timestep_pairs, ddim_step, wrap_angle
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

        # Anchor条件下扩散与推理参数。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.hybrid_alpha = float(args.fut_hybrid_alpha)
        self.cascade_intermediate_weight = float(args.fut_cascade_intermediate_weight)
        self.cascade_final_weight = float(args.fut_cascade_final_weight)

        self.train_timestep_max = 50
        self.inference_trunc_timestep = 30
        self.eval_current_timesteps, self.eval_next_timesteps = build_eval_timestep_pairs(
            self.train_timestep_max,
            self.num_inference_steps,
            self.inference_trunc_timestep,
        )

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

        self.dit_blocks = nn.ModuleList()
        self.output_heads = nn.ModuleList()
        for _ in range(self.depth):
            self.dit_blocks.append(dit.DiTBlock(self.hidden_dim, self.heads, self.dropout, self.mlp_ratio))
            self.output_heads.append(dit.FinalLayer(self.hidden_dim, self.output_dim))

        # 仅对 Ego future 做归一化，当前 future 定义为 [x, y, theta, v]。
        if self.dataset_name == "ngsim":
            self.register_buffer("fut_mean", torch.tensor([-0.06063419, 65.293495, 1.5347151, 25.216747], dtype=torch.float32), persistent=False)
            self.register_buffer("fut_std", torch.tensor([1.3011292, 56.24867, 0.26432952, 14.760194], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("fut_mean", torch.tensor([0.06540795, 221.13193, 1.5697229, 85.08167], dtype=torch.float32), persistent=False)
            self.register_buffer("fut_std", torch.tensor([1.3484404, 142.16895, 0.048561804, 24.186338], dtype=torch.float32), persistent=False)
        else:
            raise ValueError(f"Unsupported dataset '{self.dataset_name}' for fut normalization. Supported: highd, ngsim")

        anchor_path = Path(__file__).resolve().parent.parent / "dataset" / "anchor" / f"{self.dataset_name}_k{self.fut_k}.pt"
        plan_anchor = torch.load(anchor_path, map_location="cpu")
        self.plan_anchor = nn.Parameter(plan_anchor.float(), requires_grad=False)

    def computeKinematicLoss(self, pred_phys, valid_mask, fut_dt):
        loss_fn = F.l1_loss # F.smooth_l1_loss
        x = pred_phys[..., 0]
        y = pred_phys[..., 1]
        theta = pred_phys[..., 2]
        v = pred_phys[..., 3]

        x_kin_next = x[:, :-1] + v[:, :-1] * torch.cos(theta[:, :-1]) * fut_dt
        y_kin_next = y[:, :-1] + v[:, :-1] * torch.sin(theta[:, :-1]) * fut_dt
        rx = x[:, 1:] - x_kin_next
        ry = y[:, 1:] - y_kin_next

        valid_pair = valid_mask[:, :-1] * valid_mask[:, 1:]
        valid_pair_sum = valid_pair.sum()
        if valid_pair_sum.item() <= 0:
            zero = pred_phys.new_zeros(())
            return zero, zero

        kin_pred = torch.stack([rx, ry], dim=-1)
        kin_target = torch.zeros_like(kin_pred)
        loss_map = loss_fn(kin_pred, kin_target, reduction="none").mean(dim=-1)
        loss_kin = (loss_map * valid_pair).sum() / (valid_pair_sum + 1e-6)
        kin_res = torch.sqrt(rx.square() + ry.square() + 1e-12)
        kin_res_mean = (kin_res * valid_pair).sum() / (valid_pair_sum + 1e-6)
        return loss_kin, kin_res_mean

    def computeLoss(self, pred_phys, target_phys, valid_mask):
        lambda_xy = 1.0
        lambda_theta = 0.05
        lambda_v = 0.05
        lambda_kin = 0.10
        fut_dt = 0.2
        loss_fn = F.l1_loss

        valid_sum = valid_mask.sum(dim=1) + 1e-6

        xy_loss_map = loss_fn(pred_phys[..., :2], target_phys[..., :2], reduction="none").mean(dim=-1)
        loss_xy = ((xy_loss_map * valid_mask).sum(dim=1) / valid_sum).mean()

        theta_diff = wrap_angle(pred_phys[..., 2] - target_phys[..., 2])
        theta_loss_map = loss_fn(theta_diff, torch.zeros_like(theta_diff), reduction="none")
        loss_theta = ((theta_loss_map * valid_mask).sum(dim=1) / valid_sum).mean()

        v_loss_map = loss_fn(pred_phys[..., 3], target_phys[..., 3], reduction="none")
        loss_v = ((v_loss_map * valid_mask).sum(dim=1) / valid_sum).mean()

        loss_kin, kin_res_mean = self.computeKinematicLoss(pred_phys, valid_mask, fut_dt)
        loss_total = lambda_xy * loss_xy + lambda_theta * loss_theta + lambda_v * loss_v + lambda_kin * loss_kin
        logs = {
            "loss": loss_total.detach(),
            "loss_xy": loss_xy.detach(),
            "loss_theta": loss_theta.detach(),
            "loss_v": loss_v.detach(),
            "loss_kin": loss_kin.detach(),
            "kin_res_mean": kin_res_mean.detach(),
        }
        return loss_total, logs

    def mergeCascadeLoss(self, pred_phys_levels, target_phys, valid_mask):
        final_loss, final_logs = self.computeLoss(pred_phys_levels[-1], target_phys, valid_mask)
        aux_loss = final_loss.new_zeros(())
        for pred_phys in pred_phys_levels[:-1]:
            level_loss, _ = self.computeLoss(pred_phys, target_phys, valid_mask)
            aux_loss = aux_loss + level_loss
        aux_loss = aux_loss / float(len(pred_phys_levels) - 1)

        weighted_aux_loss = self.cascade_intermediate_weight * aux_loss
        total_loss = self.cascade_final_weight * final_loss + weighted_aux_loss
        final_logs["loss"] = total_loss.detach()
        final_logs["loss_final"] = final_loss.detach()
        final_logs["loss_cascade_aux"] = weighted_aux_loss.detach()
        return total_loss, final_logs

    # 训练入口：Index 与 WTA score 在 raw xy 空间计算，损失在物理坐标系(ft)计算。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device) # [B,T]

        anchor_phys_all = self.plan_anchor.to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1) # [B,K,T,D]
        anchor_x0_all = self.norm(anchor_phys_all)
        anchor_nearest_idx = torch.argmin(self.computeRawXyMseScore(anchor_phys_all, future, valid_mask), dim=-1) # [B]

        k = anchor_x0_all.size(1)
        noise = torch.randn_like(anchor_x0_all) # [B,K,T,D]
        timesteps = torch.randint(0, self.train_timestep_max, (bsz,), device=device).long() # [B]
        timesteps_all = timesteps.repeat_interleave(k) # [B*K]
        x_t = self.diffusion_scheduler.add_noise(
            anchor_x0_all.reshape(bsz * k, t_len, self.output_dim),
            noise.reshape(bsz * k, t_len, self.output_dim),
            timesteps_all,
        ).float()
        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        context_tokens_all = context_tokens.repeat_interleave(k, dim=0)
        pred_x0_levels = self.forwardDenoiseCascade(x_t, timesteps_all, context_tokens_all)
        pred_x0_all = pred_x0_levels[-1].view(bsz, k, t_len, self.output_dim)
        pred_phys_all = self.denorm(pred_x0_all)

        with torch.no_grad():
            pred_score = self.computeRawXyMseScore(pred_phys_all, future, valid_mask)
            anchor_score = self.computeRawXyMseScore(anchor_phys_all, future, valid_mask)
            mode_idx = torch.argmin(pred_score + self.hybrid_alpha * anchor_score, dim=-1)

        pred_phys_levels = [
            self.gatherMode(self.denorm(pred_x0.view(bsz, k, t_len, self.output_dim)), mode_idx)
            for pred_x0 in pred_x0_levels
        ]
        loss, loss_logs = self.mergeCascadeLoss(pred_phys_levels, future, valid_mask)
        same_rate = (mode_idx == anchor_nearest_idx).float().mean()
        max_win_ratio = torch.bincount(mode_idx.detach(), minlength=k).float().max() / max(float(bsz), 1.0)
        loss_logs["assign_same_rate"] = same_rate.detach()
        loss_logs["assign_max_win_ratio"] = max_win_ratio.detach()
        return loss, loss_logs


    @torch.no_grad()
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=None):
        bsz, t_len, _ = future.shape
        k = self.fut_k if K is None else max(1, int(K))

        context_tokens = self.hist_encoder(hist, hist_nbrs, mask, temporal_mask)
        context_tokens = context_tokens.repeat_interleave(k, dim=0)

        diffusion_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        trunc_timesteps = torch.full((bsz,), self.eval_current_timesteps[0], device=device, dtype=torch.long)
        anchor_x0_phys = self.plan_anchor[:k].to(device=device).unsqueeze(0).expand(bsz, -1, -1, -1)
        anchor_x0 = self.norm(anchor_x0_phys)
        noise = torch.randn_like(anchor_x0)
        x_t_init = diffusion_scheduler.add_noise(anchor_x0, noise, trunc_timesteps).float()
        x_t = x_t_init.reshape(bsz * k, t_len, self.output_dim)

        for current_timestep, next_timestep in zip(self.eval_current_timesteps, self.eval_next_timesteps):
            timesteps = torch.full((x_t.size(0),), int(current_timestep), device=x_t.device, dtype=torch.long)
            pred_x0 = self.forwardDenoiseCascade(x_t, timesteps, context_tokens)[-1]
            x_t = ddim_step(diffusion_scheduler, pred_x0, x_t, int(current_timestep), int(next_timestep))

        pred_phys = self.denorm(pred_x0.view(bsz, k, t_len, self.output_dim))
        all_preds = future.unsqueeze(1).repeat(1, k, 1, 1).clone()
        all_preds[..., :self.output_dim] = pred_phys
        return all_preds

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    # 归一化与反归一化，处理 future [x, y, theta, v]。
    def norm(self, x):
        x_norm = x.clone()
        mean = self.fut_mean.to(device=x.device, dtype=x.dtype)
        std = self.fut_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        x_norm[..., :self.output_dim] = (x[..., :self.output_dim] - mean[:self.output_dim]) / std[:self.output_dim]
        x_norm[..., :self.output_dim] = torch.clamp(x_norm[..., :self.output_dim], -8.0, 8.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        mean = self.fut_mean.to(device=x.device, dtype=x.dtype)
        std = self.fut_std.to(device=x.device, dtype=x.dtype).clamp(min=1e-6)
        x_denorm[..., :self.output_dim] = x[..., :self.output_dim] * std[:self.output_dim] + mean[:self.output_dim]
        return x_denorm

    def forwardDenoiseCascade(self, x_t, timesteps, context_tokens):
        preds = []
        current_x = x_t
        t_emb = self.timestep_embedder(timesteps)
        for block, output_head in zip(self.dit_blocks, self.output_heads):
            input_embedded = self.input_embedding(current_x) + self.pos_embedding(current_x)
            hidden = block(input_embedded, t_emb, context_tokens)
            pred_delta = output_head(hidden)
            pred_x0 = current_x + pred_delta
            preds.append(pred_x0)
            current_x = pred_x0.detach()
        return preds

    def computeRawXyMseScore(self, pred_phys_all, target_phys, valid_mask):
        valid_time = valid_mask.unsqueeze(1)
        diff = pred_phys_all[..., :2] - target_phys.unsqueeze(1)[..., :2]
        dist_sq = torch.sum(diff.square(), dim=-1)
        numer = (dist_sq * valid_time).sum(dim=-1)
        denom = valid_time.sum(dim=-1) + 1e-6
        return numer / denom

    def gatherMode(self, values, mode_idx):
        bsz, _, t_len, feat_dim = values.shape
        gather_index = mode_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, feat_dim)
        return torch.gather(values, 1, gather_index).squeeze(1)
