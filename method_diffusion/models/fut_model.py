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
        self.pos_loss_weight = max(0.0, float(args.fut_pos_loss_weight))
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

        context_dim = int(args.encoder_input_dim) * 2
        self.enc_embedding = nn.Linear(context_dim, self.hidden_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        # 极其优雅的判定：同维度时省去参数与计算量
        if context_dim == self.hidden_dim:
            self.context_proj = nn.Identity()
        else:
            self.context_proj = nn.Linear(context_dim, self.hidden_dim)
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

        # ================= 双空间归一化参数 =================
        # 保留坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        self.register_buffer("pos_mean", torch.tensor([0.0330, -15.9150]).float(), persistent=False)
        self.register_buffer("pos_std", torch.tensor([8.8866, 68.8105]).float(), persistent=False)
        self.register_buffer("va_mean", torch.tensor([21.1503, 0.0060]).float(), persistent=False)
        self.register_buffer("va_std", torch.tensor([13.5983, 4.5057]).float(), persistent=False)

        # 针对帧间相对位移 (Velocity) 的归一化参数
        vel_mean = torch.tensor([-0.004182, 5.041937], dtype=torch.float32)
        vel_std = torch.tensor([0.150222, 2.951254], dtype=torch.float32)

        self.register_buffer("vel_mean", vel_mean, persistent=False)
        self.register_buffer("vel_std", vel_std, persistent=False)

    @staticmethod
    def toValidMask(op_mask, seq_len, batch_size, device):
        if op_mask is None: return torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)
        if op_mask.dim() == 3:
            valid = op_mask[..., 0]
        elif op_mask.dim() == 2:
            valid = op_mask
        valid = (valid > 0.5).float()
        if valid.size(1) > seq_len:
            valid = valid[:, :seq_len]
        elif valid.size(1) < seq_len:
            pad = torch.ones((batch_size, seq_len - valid.size(1)), dtype=valid.dtype, device=valid.device)
            valid = torch.cat([valid, pad], dim=1)
        return valid.to(device)

    def buildInferenceScheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config,
                                                  timestep_spacing=self.inference_timestep_spacing)
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    def encodeGlobalCondition(self, context):
        return self.enc_embedding(context[:, -1, :])

    # 预测函数：此时模型吐出的是 归一化后的相对位移(速度)
    def predictX0(self, x_t, timesteps, context_aligned, enc_emb, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        return self.dit(x=input_embedded, y=t_emb + enc_emb, cross=context_aligned)

    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=False):
        """双轨 Loss: vel_huber + pos_huber(physical, 1/sqrt(t))."""
        vel_weights = [1.0] * self.output_dim
        if self.output_dim >= 2:
            vel_weights[1] = self.y_loss_weight
        vel_axis_weight = torch.tensor(
            vel_weights, device=pred_vel_norm.device, dtype=pred_vel_norm.dtype
        ).view(1, 1, self.output_dim)

        # 1. 速度域微观约束 (Huber Loss)：在归一化空间算，梯度极度稳定，防高频抖动
        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)
        loss_vel = loss_vel * vel_axis_weight

        # 2. 积分回物理位置域 (宏观绝对约束)
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)

        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        t_len = pred_pos_phys.size(1)
        pos_err = pred_pos_phys - future_phys[..., :2]

        # 3. 物理空间位置损失 + 1/sqrt(t) 时间折扣
        loss_pos_raw = F.smooth_l1_loss(
            pos_err, torch.zeros_like(pos_err), reduction="none", beta=self.huber_delta
        )
        t_seq = torch.arange(1, t_len + 1, device=pred_pos_phys.device, dtype=pred_pos_phys.dtype).view(1, t_len, 1)
        time_weights = 1.0 / torch.sqrt(t_seq)
        loss_pos = loss_pos_raw * time_weights

        # 维度安全补齐 (应对 output_dim > 2 的情况)
        if self.output_dim > 2:
            pad = torch.zeros(loss_pos.shape[:-1] + (self.output_dim - 2,), device=loss_pos.device,
                              dtype=loss_pos.dtype)
            loss_pos = torch.cat([loss_pos, pad], dim=-1)

        total_loss = loss_vel + self.pos_loss_weight * loss_pos

        valid = valid_mask.unsqueeze(-1)
        numer = (total_loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        total_mean = (numer / denom).mean()

        if not return_parts:
            return total_mean

        vel_mean = ((loss_vel * valid).sum(dim=(1, 2)) / denom).mean()
        pos_mean = ((loss_pos * valid).sum(dim=(1, 2)) / denom).mean()

        valid_t = valid_mask
        denom_t = valid_t.sum(dim=1) + 1e-6
        vel_x_mean = ((loss_vel[..., 0] * valid_t).sum(dim=1) / denom_t).mean()
        if self.output_dim >= 2:
            vel_y_mean = ((loss_vel[..., 1] * valid_t).sum(dim=1) / denom_t).mean()
        else:
            vel_y_mean = vel_x_mean.new_tensor(0.0)
        pos_x_mean = ((loss_pos[..., 0] * valid_t).sum(dim=1) / denom_t).mean()
        pos_y_mean = ((loss_pos[..., 1] * valid_t).sum(dim=1) / denom_t).mean()

        parts = {
            "loss_total": total_mean.detach(),
            "loss_vel": vel_mean.detach(),
            "loss_vel_x": vel_x_mean.detach(),
            "loss_vel_y": vel_y_mean.detach(),
            "loss_pos": pos_mean.detach(),
            "loss_pos_x": pos_x_mean.detach(),
            "loss_pos_y": pos_y_mean.detach(),
        }
        return total_mean, parts

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
        pred_vel_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_vel_norm = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_vel_cond)
            if self.x0_clip is not None:
                pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)

            pred_vel_cond = pred_vel_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t).prev_sample
        return pred_vel_cond

    def maybeVisualize(self, hist, hist_nbrs, temporal_mask, future, pred, valid_mask, stage,
                       pred_all=None, pred_best_idx=None):
        if not self.is_main_process: return
        if stage == "train":
            self.trainForwardCalls += 1
            if (not self.fut_enable_train_vis) or (self.trainForwardCalls % self.fut_vis_every_n != 0): return
        else:
            self.evalForwardCalls += 1
            if (not self.fut_enable_eval_vis) or (self.evalForwardCalls % self.fut_vis_every_n != 0): return

        vis_batch_idx = 0
        vis_ade, vis_fde = self.computeSingleAdeFde(pred, future, valid_mask, batch_idx=vis_batch_idx)
        metrics = {"ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
                   "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot}}
        visualize_batch_trajectories(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=pred,
            pred_all=pred_all,
            pred_best_idx=pred_best_idx,
            future_mask=valid_mask,
            batch_idx=vis_batch_idx,
            save_path=None,
            metrics=metrics,
            input_unit="ft",
            show_plot=True,
        )

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)

        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        # 物理法则转换：计算 GT 的真实物理帧间位移 (Velocity)
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys

        # 独立归一化：将物理速度变为完美的正态分布
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -10.0, 10.0)

        # 现在的加噪目标是：归一化的速度分布！
        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        hist_norm = self.norm(hist)
        context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context)
        context_aligned = self.context_proj(context)

        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_vel_cond)
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        # 网络输出：预测的归一化速度
        pred_vel_norm_t = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_vel_cond)

        # 传入双轨 Loss 防累积漂移
        loss, loss_parts = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            return_parts=True,
        )

        if self.fut_enable_train_vis:
            pred_vel_phys_t = pred_vel_norm_t[..., :2] * std_vel + mean_vel
            pred_pos_phys = torch.cumsum(pred_vel_phys_t, dim=1) + anchor_phys[..., :2]

            # 兼容多维度输出拼接
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            self.maybeVisualize(
                hist=hist,
                hist_nbrs=hist_nbrs,
                temporal_mask=temporal_mask,
                future=future,
                pred=pred_phys_abs,
                valid_mask=valid_mask,
                stage="train",
            )

        if return_components:
            return loss, loss_parts
        return loss

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)

        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        hist_norm = self.norm(hist)
        context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context)
        context_aligned = self.context_proj(context)

        infer_scheduler = self.buildInferenceScheduler()

        # 推断起点：一团关于速度的纯噪声
        x_t = torch.randn((bsz, t_len, self.output_dim), device=device)
        pred_vel_norm = self.rolloutFromXt(x_t, context_aligned, enc_emb, infer_scheduler)

        # 积分魔法还原：解归一化 -> 累加 -> 拼接绝对锚点
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)

        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]

        pred_phys_abs = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys

        # 构建 Eval Loss (假 Loss 用于记录)
        shifted_future = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_norm = future_phys.clone()
        target_vel_norm[..., :2] = ((future_phys[..., :2] - shifted_future[..., :2]) - mean_vel) / std_vel
        loss = self.computeLoss(
            pred_vel_norm,
            torch.clamp(target_vel_norm, -10.0, 10.0),
            future_phys,
            anchor_phys,
            valid_mask,
            return_parts=False,
        )

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        self.maybeVisualize(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=pred_phys_abs,
            valid_mask=valid_mask,
            stage="eval",
        )
        return loss, pred_phys_abs, ade, fde

    @torch.no_grad()
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5):
        """
        并行化提速版 SOTA 多模态评估 (minADE_K)
        通过在 Batch 维度展开，消除 for 循环，推理速度提升 K 倍。
        """
        bsz, t_len, _ = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        hist_norm = self.norm(hist)
        context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context)
        context_aligned = self.context_proj(context)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍
        context_aligned_k = context_aligned.repeat_interleave(K, dim=0)
        enc_emb_k = enc_emb.repeat_interleave(K, dim=0)

        infer_scheduler = self.buildInferenceScheduler()
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程
        x_t_k = torch.randn((bsz * K, t_len, self.output_dim), device=device)
        pred_vel_cond_k = torch.zeros((bsz * K, t_len, self.output_dim), device=device, dtype=x_t_k.dtype)

        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps_k = torch.full((bsz * K,), t_scalar, device=device, dtype=torch.long)

            pred_vel_norm_k = self.predictX0(x_t_k, timesteps_k, context_aligned_k, enc_emb_k, pred_vel_cond_k)
            if self.x0_clip is not None:
                pred_vel_norm_k = torch.clamp(pred_vel_norm_k, -self.x0_clip, self.x0_clip)

            pred_vel_cond_k = pred_vel_norm_k.detach()
            try:
                x_t_k = infer_scheduler.step(pred_vel_norm_k, t, x_t_k, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t_k = infer_scheduler.step(pred_vel_norm_k, t, x_t_k).prev_sample

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]
        pred_vel_norm = pred_vel_cond_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)
        std_vel_k = std_vel.unsqueeze(1)  # [1, 1, 1, 2]
        mean_vel_k = mean_vel.unsqueeze(1)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel_k + mean_vel_k

        anchor_phys_k = anchor_phys[..., :2].unsqueeze(1)  # [bsz, 1, 1, 2]
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys_k

        # 兼容多维度特征拼接
        all_preds = future_phys.unsqueeze(1).repeat(1, K, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys

        # 计算 minADE_K
        target_phys = future_phys[..., :2].unsqueeze(1)  # [bsz, 1, t_len, 2]
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)

        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        min_ade, best_k_idx = torch.min(ade_k, dim=1)

        # 提取 K 条中最贴合真实意图的那一条
        best_k_idx_exp = best_k_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, self.output_dim)
        best_pred_phys = all_preds.gather(1, best_k_idx_exp).squeeze(1)

        # 缓存多模态采样结果，供外部评估/可视化直接使用
        self.last_minade_all_preds = all_preds.detach()
        self.last_minade_best_idx = best_k_idx.detach()

        ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
        dummy_loss = torch.tensor(0.0, device=device)
        self.maybeVisualize(
            hist=hist,
            hist_nbrs=hist_nbrs,
            temporal_mask=temporal_mask,
            future=future,
            pred=best_pred_phys,
            valid_mask=valid_mask,
            stage="eval",
            pred_all=all_preds,
            pred_best_idx=best_k_idx,
        )

        return dummy_loss, best_pred_phys, ade_batch, fde_batch

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        return self.forwardTrain(
            hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=return_components
        )

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
