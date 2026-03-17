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
        if int(args.feature_dim) != 4:
            raise ValueError(
                "Current unified future branch requires feature_dim=4: "
                "[rel_x, rel_y, delta_x, delta_y]"
            )

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
        self.huber_delta = max(1e-4, float(args.fut_huber_delta))
        x0_clip_val = float(args.x0_clip)
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.input_embedding = nn.Linear(self.output_dim * 2, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)
        if int(getattr(self.hist_encoder, "hidden_dim", self.hidden_dim)) != self.hidden_dim:
            raise ValueError(
                f"HistEncoder hidden_dim must match hidden_dim_fut, got {getattr(self.hist_encoder, 'hidden_dim', None)} vs {self.hidden_dim}"
            )
        self.cond_projs = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.depth)])
        self.intent_endpoint_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        for proj in self.cond_projs:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0)
        for m in self.intent_endpoint_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

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
        # unified history 使用 anchor-relative 位置，均值按 0 处理。
        # std 先沿用位置尺度，后续可按新分布重统计。
        self.register_buffer("rel_pos_mean", torch.zeros(2, dtype=torch.float32), persistent=False)
        self.register_buffer("rel_pos_std", torch.tensor([8.8866, 68.8105], dtype=torch.float32), persistent=False)

        # 针对帧间相对位移 (Velocity) 的归一化参数
        vel_mean = torch.tensor([-0.004182, 5.041937], dtype=torch.float32)
        vel_std = torch.tensor([0.150222, 2.951254], dtype=torch.float32)

        self.register_buffer("vel_mean", vel_mean, persistent=False)
        self.register_buffer("vel_std", vel_std, persistent=False)

        self.loss_w_vel = max(0.0, float(args.fut_y_loss_weight))
        self.loss_w_pos = max(0.0, float(args.fut_pos_loss_weight))
        self.loss_w_end = max(0.0, float(getattr(args, "fut_end_loss_weight", 0.5)))
        self.loss_w_intent = max(0.0, float(getattr(args, "fut_intent_loss_weight", 0.2)))

    def computeLoss(
        self,
        pred_vel_norm,
        target_vel_norm,
        future_phys,
        anchor_phys,
        valid_mask,
        pred_intent_disp=None,
        return_components=False,
    ):
        # 统一损失：L_vel + lambda_pos * L_pos + lambda_end * L_end + lambda_intent * L_intent
        if pred_vel_norm.size(-1) != 2 or target_vel_norm.size(-1) != 2:
            raise ValueError(f"computeLoss currently expects dim=2, got pred={pred_vel_norm.size(-1)}, target={target_vel_norm.size(-1)}")

        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)

        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        gt_pos_phys = future_phys[..., :2]
        loss_pos = F.smooth_l1_loss(pred_pos_phys, gt_pos_phys, reduction="none", beta=self.huber_delta)

        pred_end, has_valid = self.gatherLastValid(pred_pos_phys, valid_mask)
        gt_end, _ = self.gatherLastValid(gt_pos_phys, valid_mask)
        loss_end_vec = F.smooth_l1_loss(pred_end, gt_end, reduction="none", beta=self.huber_delta)
        loss_end = self.maskedMean1d(loss_end_vec.mean(dim=-1), has_valid.float())

        target_intent_disp = gt_end - anchor_phys[..., :2].squeeze(1)
        if pred_intent_disp is None:
            loss_intent = torch.zeros_like(loss_end)
        else:
            loss_intent_vec = F.smooth_l1_loss(pred_intent_disp, target_intent_disp, reduction="none", beta=self.huber_delta)
            loss_intent = self.maskedMean1d(loss_intent_vec.mean(dim=-1), has_valid.float())

        loss_vel_mean = self.maskedMean3d(loss_vel, valid_mask)
        loss_pos_mean = self.maskedMean3d(loss_pos, valid_mask)
        loss = (
            self.loss_w_vel * loss_vel_mean
            + self.loss_w_pos * loss_pos_mean
            + self.loss_w_end * loss_end
            + self.loss_w_intent * loss_intent
        )
        if not return_components:
            return loss

        loss_metrics = self.summarizeLossForLog(
            loss_vel=loss_vel,
            loss_pos=loss_pos,
            loss_end=loss_end,
            loss_intent=loss_intent,
            valid_mask=valid_mask,
            total_loss=loss.detach(),
        )
        return loss, loss_metrics

    @staticmethod
    def maskedMean3d(loss_tensor, valid_mask):
        valid = valid_mask.unsqueeze(-1)
        numer = (loss_tensor * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

    @staticmethod
    def maskedMean2d(loss_tensor, valid_mask):
        denom = valid_mask.sum(dim=1) + 1e-6
        return ((loss_tensor * valid_mask).sum(dim=1) / denom).mean()

    @staticmethod
    def maskedMean1d(loss_tensor, valid_mask):
        numer = (loss_tensor * valid_mask).sum()
        denom = valid_mask.sum() + 1e-6
        return numer / denom

    @staticmethod
    def gatherLastValid(seq, valid_mask):
        # seq: [B, T, D], valid_mask: [B, T]
        t_idx = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(valid_mask)
        masked_idx = torch.where(valid_mask > 0, t_idx, t_idx.new_full(t_idx.shape, -1))
        last_idx = masked_idx.max(dim=1).values
        has_valid = last_idx >= 0
        safe_idx = last_idx.clamp(min=0)
        gather_idx = safe_idx.view(-1, 1, 1).expand(-1, 1, seq.size(-1))
        gathered = seq.gather(1, gather_idx).squeeze(1)
        return gathered, has_valid

    def summarizeLossForLog(self, loss_vel, loss_pos, loss_end, loss_intent, valid_mask, total_loss):
        # 在不参与反传的前提下，分别统计总损失/速度损失/位置损失及分量均值。
        with torch.no_grad():
            loss_vel_det = loss_vel.detach()
            loss_pos_det = loss_pos.detach()

            vel_x_mean = self.maskedMean2d(loss_vel_det[..., 0], valid_mask)
            vel_y_mean = self.maskedMean2d(loss_vel_det[..., 1], valid_mask)
            pos_x_mean = self.maskedMean2d(loss_pos_det[..., 0], valid_mask)
            pos_y_mean = self.maskedMean2d(loss_pos_det[..., 1], valid_mask)
            return {
                "loss_total": total_loss,
                "loss_vel": self.maskedMean3d(loss_vel_det, valid_mask),
                "loss_vel_x": vel_x_mean,
                "loss_vel_y": vel_y_mean,
                "loss_pos": self.maskedMean3d(loss_pos_det, valid_mask),
                "loss_pos_x": pos_x_mean,
                "loss_pos_y": pos_y_mean,
                "loss_end": loss_end.detach(),
                "loss_intent": loss_intent.detach(),
            }

    @staticmethod
    def computeAdeFde(pred, target, valid_mask):
        diff = pred[..., :2] - target[..., :2]
        dist = torch.norm(diff, dim=-1)
        ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)
        t_idx = torch.arange(dist.size(1), device=dist.device).unsqueeze(0).expand_as(dist)
        masked_idx = torch.where(valid_mask > 0, t_idx, t_idx.new_full(t_idx.shape, -1))
        last_idx = masked_idx.max(dim=1).values
        has_valid = last_idx >= 0
        final_dist = dist.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
        return ade, fde

    def buildUnifiedHistoryState(self, hist, hist_nbrs, temporal_mask):
        # 输入统一为 [rel_x, rel_y, delta_x, delta_y]
        # 先对 ego 做 anchor-relative；邻居轨迹按所属 ego 的 anchor 对齐。
        anchor = hist[:, -1:, :2]
        hist_pos = hist[..., :2] - anchor
        hist_prev = torch.cat((hist_pos[:, :1, :], hist_pos[:, :-1, :]), dim=1)
        hist_delta = hist_pos - hist_prev
        hist_state = torch.cat((hist_pos, hist_delta), dim=-1)

        nbr_pos = hist_nbrs[..., :2]
        if nbr_pos.size(0) > 0:
            temporal_occ = temporal_mask.view(
                temporal_mask.size(0),
                temporal_mask.size(1) * temporal_mask.size(2),
                temporal_mask.size(3),
            ).any(dim=-1)
            nbr_owner = temporal_occ.nonzero(as_tuple=False)[:, 0]
            if nbr_owner.numel() != nbr_pos.size(0):
                raise RuntimeError(
                    f"Neighbor count mismatch: owners={nbr_owner.numel()} vs nbr_tokens={nbr_pos.size(0)}"
                )
            nbr_anchor = anchor.index_select(0, nbr_owner)
            nbr_pos = nbr_pos - nbr_anchor
        nbr_prev = torch.cat((nbr_pos[:, :1, :], nbr_pos[:, :-1, :]), dim=1)
        nbr_delta = nbr_pos - nbr_prev
        nbr_state = torch.cat((nbr_pos, nbr_delta), dim=-1)
        return hist_state, nbr_state

    def normUnifiedState(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.rel_pos_mean) / self.rel_pos_std
        x_norm[..., 2:4] = (x[..., 2:4] - self.vel_mean) / self.vel_std
        x_norm = torch.clamp(x_norm, -10.0, 10.0)
        return x_norm

    def encodeHistoryCondition(self, hist, hist_nbrs, mask, temporal_mask):
        hist_state, nbr_state = self.buildUnifiedHistoryState(hist, hist_nbrs, temporal_mask)
        hist_state_norm = self.normUnifiedState(hist_state)
        nbr_state_norm = self.normUnifiedState(nbr_state)
        memory_tokens, z_intent, memory_mask = self.hist_encoder(hist_state_norm, nbr_state_norm, mask, temporal_mask)
        pred_intent_disp = self.intent_endpoint_head(z_intent)
        return memory_tokens, z_intent, memory_mask, pred_intent_disp

    def buildLayerCondition(self, timesteps, z_intent):
        t_emb = self.timestep_embedder(timesteps)
        return [t_emb + proj(z_intent) for proj in self.cond_projs]

    def predictX0(self, x_t, timesteps, memory_tokens, z_intent, memory_mask, pred_x0_cond):
        # 条件去噪：cross 读 memory token，AdaLN 读 z_intent 的逐层投影。
        y_layers = self.buildLayerCondition(timesteps, z_intent)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        return self.dit(x=input_embedded, y=y_layers, cross=memory_tokens, cross_attn_mask=memory_mask)

    def rolloutFromXt(self, x_t, memory_tokens, z_intent, memory_mask, infer_scheduler):
        # 核心功能：从初始噪声 x_t 出发，按推理调度器逐步去噪，得到最终的归一化速度预测。
        # 实现逻辑：每个时间步先用 predictX0 估计 x0（可选裁剪），再用 scheduler.step 反演到前一状态。
        # 作用：统一 train/eval 采样路径，保证单模态与多模态推理都复用同一去噪过程。
        bsz, t_len, _ = x_t.shape
        pred_vel_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            pred_vel_norm = self.predictX0(x_t, timesteps, memory_tokens, z_intent, memory_mask, pred_vel_cond)
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
            if not self.fut_enable_train_vis: return
        else:
            if not self.fut_enable_eval_vis: return

        vis_batch_idx = 0
        b_idx = min(max(int(vis_batch_idx), 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - future[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        vis_ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_idx = torch.nonzero(vm > 0, as_tuple=False).squeeze(-1)
        vis_fde = dist[valid_idx[-1]] if valid_idx.numel() > 0 else dist.new_tensor(0.0)
        metrics = {"ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
                   "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot}}
        visualize_batch_trajectories(
            hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask, future=future, pred=pred, pred_all=pred_all,
            pred_best_idx=pred_best_idx, future_mask=valid_mask, batch_idx=vis_batch_idx, save_path=None, metrics=metrics,
            input_unit="ft", show_plot=True,
        )

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        bsz, t_len, _ = future.shape
        # op_mask 与 future 对齐，默认 [B, T_f, D]，当前 D=2 且通道等价，直接取第 0 通道作为有效位。
        valid_mask = (op_mask[..., 0] > 0).float()

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

        memory_tokens, z_intent, memory_mask, pred_intent_disp = self.encodeHistoryCondition(
            hist, hist_nbrs, mask, temporal_mask
        )

        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(
                        x_t, timesteps, memory_tokens, z_intent, memory_mask, pred_vel_cond
                    )
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        # 网络输出：预测的归一化速度
        pred_vel_norm_t = self.predictX0(x_t, timesteps, memory_tokens, z_intent, memory_mask, pred_vel_cond)

        loss, loss_metrics = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            pred_intent_disp=pred_intent_disp,
            return_components=True,
        )

        if self.fut_enable_train_vis:
            pred_vel_phys_t = pred_vel_norm_t[..., :2] * std_vel + mean_vel
            pred_pos_phys = torch.cumsum(pred_vel_phys_t, dim=1) + anchor_phys[..., :2]

            # 兼容多维度输出拼接
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            self.maybeVisualize(hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask, future=future, pred=pred_phys_abs, valid_mask=valid_mask, stage="train")

        if return_components:
            return loss, loss_metrics
        return loss

    @torch.no_grad()
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()

        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        memory_tokens, z_intent, memory_mask, pred_intent_disp = self.encodeHistoryCondition(
            hist, hist_nbrs, mask, temporal_mask
        )

        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config, timestep_spacing=self.inference_timestep_spacing)
        infer_scheduler.set_timesteps(self.num_inference_steps)

        # 推断起点：一团关于速度的纯噪声
        x_t = torch.randn((bsz, t_len, self.output_dim), device=device)
        pred_vel_norm = self.rolloutFromXt(x_t, memory_tokens, z_intent, memory_mask, infer_scheduler)

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
            pred_intent_disp=pred_intent_disp,
        )

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        self.maybeVisualize(hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask, future=future, pred=pred_phys_abs, valid_mask=valid_mask, stage="eval")
        return loss, pred_phys_abs, ade, fde

    @torch.no_grad()
    def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5):
        """
        并行化提速版 SOTA 多模态评估 (minADE_K)
        通过在 Batch 维度展开，消除 for 循环，推理速度提升 K 倍。
        """
        bsz, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        memory_tokens, z_intent, memory_mask, _ = self.encodeHistoryCondition(hist, hist_nbrs, mask, temporal_mask)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍
        memory_tokens_k = memory_tokens.repeat_interleave(K, dim=0)
        z_intent_k = z_intent.repeat_interleave(K, dim=0)
        memory_mask_k = memory_mask.repeat_interleave(K, dim=0)

        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config, timestep_spacing=self.inference_timestep_spacing)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程
        x_t_k = torch.randn((bsz * K, t_len, self.output_dim), device=device)
        pred_vel_cond_k = torch.zeros((bsz * K, t_len, self.output_dim), device=device, dtype=x_t_k.dtype)

        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps_k = torch.full((bsz * K,), t_scalar, device=device, dtype=torch.long)

            pred_vel_norm_k = self.predictX0(
                x_t_k, timesteps_k, memory_tokens_k, z_intent_k, memory_mask_k, pred_vel_cond_k
            )
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
            hist=hist, hist_nbrs=hist_nbrs, temporal_mask=temporal_mask, future=future, pred=best_pred_phys, valid_mask=valid_mask,
            stage="eval", pred_all=all_preds, pred_best_idx=best_k_idx,
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
