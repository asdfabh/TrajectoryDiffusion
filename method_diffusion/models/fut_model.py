from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import torch
import torch.nn.functional as F
import os
from method_diffusion.models.hist_encoder import HistEncoder
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import visualize_batch_trajectories


class DiffusionFut(nn.Module):

    # 初始化FUT扩散模型结构、归一化参数与训练超参数。
    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args

        self.feature_dim = int(args.feature_dim_fut)
        self.input_dim = int(args.input_dim_fut)
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

        self.self_condition_prob = float(args.self_condition_prob)
        self.self_condition_prob = min(max(self.self_condition_prob, 0.0), 1.0)
        self.train_timestep_align_ratio = float(args.train_timestep_align_ratio)
        self.train_timestep_align_ratio = min(max(self.train_timestep_align_ratio, 0.0), 1.0)

        self.fut_loss_mode = str(args.fut_loss_mode)
        self.fut_loss_pos_weight = float(args.fut_loss_pos_weight)
        self.fut_loss_vel_weight = float(args.fut_loss_vel_weight)
        self.fut_y_loss_weight = max(1.0, float(args.fut_y_loss_weight))
        self.fut_time_weight_min = float(args.fut_time_weight_min)
        self.fut_time_weight_max = float(args.fut_time_weight_max)

        self.fut_pos_loss_type = str(args.fut_pos_loss_type)
        self.fut_huber_delta = max(1e-4, float(args.fut_huber_delta))
        self.fut_loss_acc_weight = float(args.fut_loss_acc_weight)
        self.fut_loss_endpoint_weight = float(args.fut_loss_endpoint_weight)
        self.fut_high_noise_threshold = min(max(float(args.fut_high_noise_threshold), 0.0), 1.0)
        self.fut_high_noise_weight = max(1.0, float(args.fut_high_noise_weight))

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.fut_vis_every_n = max(1, int(args.fut_vis_every_n))
        self.trainForwardCalls = 0
        self.evalForwardCalls = 0

        self.T = int(args.T_f)
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.input_embedding = nn.Linear(self.feature_dim + self.output_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)
        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim * 3, self.input_dim)
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
        self.dit = dit.DiT(dit_block=dit_block, final_layer=self.final_layer, depth=self.depth, model_type="x_start")

        align_scheduler = self.buildInferenceScheduler()
        align_ts = align_scheduler.timesteps.long().clone()
        if align_ts.ndim == 0:
            align_ts = align_ts.unsqueeze(0)
        self.register_buffer("aligned_train_timesteps", align_ts, persistent=False)

        self.register_buffer('pos_mean', torch.tensor([0.0330, -15.9150]).float(), persistent=False)
        self.register_buffer('pos_std', torch.tensor([8.8866, 68.8105]).float(), persistent=False)
        self.register_buffer('va_mean', torch.tensor([21.1503, 0.0060]).float(), persistent=False)
        self.register_buffer('va_std', torch.tensor([13.5983, 4.5057]).float(), persistent=False)

    @staticmethod
    # 将op_mask整理为与future长度一致的有效位掩码。
    def toValidMask(op_mask, seq_len, batch_size, device):
        if op_mask is None:
            return torch.ones((batch_size, seq_len), dtype=torch.float32, device=device)
        if op_mask.dim() == 3:
            valid = op_mask[..., 0]
        elif op_mask.dim() == 2:
            valid = op_mask
        else:
            raise ValueError(f"Unsupported op_mask shape: {tuple(op_mask.shape)}")
        valid = (valid > 0.5).float()
        if valid.size(1) > seq_len:
            valid = valid[:, :seq_len]
        elif valid.size(1) < seq_len:
            pad = torch.ones((batch_size, seq_len - valid.size(1)), dtype=valid.dtype, device=valid.device)
            valid = torch.cat([valid, pad], dim=1)
        return valid.to(device)

    @staticmethod
    # 按样本在掩码区域内计算逐样本平均损失。
    def maskedReducePerSample(value, mask):
        value = value * mask
        sum_dims = tuple(range(1, value.dim()))
        numer = value.sum(dim=sum_dims)
        denom = mask.sum(dim=sum_dims) + 1e-6
        return numer / denom

    @staticmethod
    # 对逐样本损失执行可选的样本权重加权平均。
    def weightedBatchReduce(loss_per_sample, sample_weight=None):
        if sample_weight is None:
            return loss_per_sample.mean()
        weight = torch.clamp(sample_weight, min=0.0)
        weight_sum = weight.sum()
        if weight_sum <= 1e-6:
            return loss_per_sample.new_tensor(0.0)
        return (loss_per_sample * weight).sum() / (weight_sum + 1e-6)

    # 采样训练用扩散时间步并可按推理步点分布对齐。
    def sampleTrainTimesteps(self, batch_size, device):
        low = 0
        uniform_t = torch.randint(low, self.num_train_timesteps, (batch_size,), device=device).long()
        if self.train_timestep_align_ratio <= 0.0:
            return uniform_t
        align_pool = self.aligned_train_timesteps.to(device)
        if align_pool.numel() == 0:
            return uniform_t
        align_ids = torch.randint(0, align_pool.numel(), (batch_size,), device=device)
        aligned_t = align_pool[align_ids].long()
        aligned_t = torch.clamp(aligned_t, min=low, max=self.num_train_timesteps - 1)
        choose_align = (torch.rand(batch_size, device=device) < self.train_timestep_align_ratio)
        return torch.where(choose_align, aligned_t, uniform_t)

    # 构建与当前推理参数一致的DDIM调度器。
    def buildInferenceScheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config,
                                                  timestep_spacing=self.inference_timestep_spacing)
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    # 根据噪声强度区间给训练样本分配额外权重。
    def buildNoiseSampleWeight(self, timesteps):
        weights = torch.ones_like(timesteps, dtype=torch.float32)
        if self.fut_high_noise_weight <= 1.0:
            return weights
        t_ratio = timesteps.float() / float(max(1, self.num_train_timesteps - 1))
        high_mask = t_ratio >= self.fut_high_noise_threshold
        high_value = torch.full_like(weights, self.fut_high_noise_weight)
        return torch.where(high_mask, high_value, weights)

    # 在给定条件下预测当前噪声状态对应的x0。
    def predictX0(self, x_t, timesteps, context, enc_emb, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        y = t_emb + enc_emb
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(combined_input)
        pred_x0 = self.dit(x=input_embedded, y=y, cross=context)
        return pred_x0

    # 生成DiT的全局条件向量（自车意图+场景上下文）。
    def encodeGlobalCondition(self, context, hist_enc):
        ego_intent = hist_enc[:, -1, :]
        scene_context = context.max(dim=1).values
        global_context = torch.cat([ego_intent, scene_context], dim=-1)
        in_dim = int(self.enc_embedding.in_features)
        cur_dim = int(global_context.size(-1))
        if cur_dim > in_dim:
            global_context = global_context[..., :in_dim]
        elif cur_dim < in_dim:
            pad = torch.zeros((global_context.size(0), in_dim - cur_dim), device=global_context.device,
                              dtype=global_context.dtype)
            global_context = torch.cat([global_context, pad], dim=-1)
        return self.enc_embedding(global_context)

    # 构建各向异性坐标损失权重向量。
    def buildAnisotropicAxisWeight(self, device, dtype):
        return torch.tensor([1.0, self.fut_y_loss_weight], device=device, dtype=dtype).view(1, 1, 2)

    # 从给定x_t执行完整DDIM反推获得最终预测轨迹。
    def rolloutFromXt(self, x_t, context, enc_emb, infer_scheduler):
        bsz, t_len, _ = x_t.shape
        pred_x0_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)
            pred_x0_norm = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
            if self.x0_clip is not None:
                pred_x0_norm = torch.clamp(pred_x0_norm, -self.x0_clip, self.x0_clip)
            pred_x0_cond = pred_x0_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_x0_norm, t, x_t).prev_sample
        return x_t

    # 用GT加噪初始化后完整反推计算训练诊断指标。
    def computeTrainRolloutMetricsFromGtNoise(self, context, enc_emb, future, future_norm, valid_mask):
        infer_scheduler = self.buildInferenceScheduler()
        start_t = int(infer_scheduler.timesteps[0].item()) if isinstance(infer_scheduler.timesteps[0],
                                                                         torch.Tensor) else int(
            infer_scheduler.timesteps[0])
        start_timesteps = torch.full((future_norm.size(0),), start_t, device=future_norm.device, dtype=torch.long)
        full_noise = torch.randn_like(future_norm)
        x_t_start = self.diffusion_scheduler.add_noise(future_norm, full_noise, start_timesteps)
        pred_norm = self.rolloutFromXt(x_t_start, context, enc_emb, infer_scheduler)
        pred = self.denorm(pred_norm)
        ade, fde = self.computeAdeFde(pred, future, valid_mask)
        return pred, ade, fde

    # 构建随时间步递增的损失权重曲线。
    def buildTimeWeights(self, seq_len, device, dtype):
        w_min = self.fut_time_weight_min
        w_max = self.fut_time_weight_max
        if w_max < w_min:
            w_min, w_max = w_max, w_min
        if seq_len <= 1:
            return torch.ones((1, seq_len, 1), device=device, dtype=dtype) * w_max
        weights = torch.linspace(w_min, w_max, seq_len, device=device, dtype=dtype).view(1, seq_len, 1)
        return weights

    # 计算主训练损失（归一化空间L1+速度项）。
    def computeLossL1TimeVel(self, pred, target, valid_mask, sample_weight=None):
        pred_pos = pred[..., :2]
        target_pos = target[..., :2]
        axis_weight = self.buildAnisotropicAxisWeight(pred.device, pred.dtype)
        mask_xy = valid_mask.unsqueeze(-1).expand_as(pred_pos)
        time_weight = self.buildTimeWeights(pred_pos.size(1), pred.device, pred.dtype)
        weighted_mask_xy = mask_xy * time_weight.expand(pred_pos.size(0), -1, pred_pos.size(2))

        pos_err = torch.abs(pred_pos - target_pos) * axis_weight
        loss_pos_ps = self.maskedReducePerSample(pos_err, weighted_mask_xy)

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        if pred_vel.numel() == 0:
            loss_vel_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_vel = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1).expand_as(pred_vel)
            time_weight_vel = 0.5 * (time_weight[:, 1:, :] + time_weight[:, :-1, :])
            weighted_valid_vel = valid_vel * time_weight_vel.expand(pred_pos.size(0), -1, pred_pos.size(2))
            vel_err = torch.abs(pred_vel - target_vel) * axis_weight
            loss_vel_ps = self.maskedReducePerSample(vel_err, weighted_valid_vel)

        total_ps = self.fut_loss_pos_weight * loss_pos_ps + self.fut_loss_vel_weight * loss_vel_ps
        return self.weightedBatchReduce(total_ps, sample_weight=sample_weight)

    # 计算包含加速度与终点项的legacy损失。
    def computeLossLegacy(self, pred, target, valid_mask, sample_weight=None):
        pred_pos = pred[..., :2]
        target_pos = target[..., :2]
        axis_weight = self.buildAnisotropicAxisWeight(pred.device, pred.dtype)
        mask_xy = valid_mask.unsqueeze(-1).expand_as(pred_pos)

        if self.fut_pos_loss_type == "huber":
            pos_err = F.smooth_l1_loss(pred_pos, target_pos, reduction="none", beta=self.fut_huber_delta)
        else:
            pos_err = torch.abs(pred_pos - target_pos)
        pos_err = pos_err * axis_weight
        loss_pos_ps = self.maskedReducePerSample(pos_err, mask_xy)

        pred_vel = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]
        target_vel = target_pos[:, 1:, :] - target_pos[:, :-1, :]
        if pred_vel.numel() == 0:
            loss_vel_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_vel = (valid_mask[:, 1:] * valid_mask[:, :-1]).unsqueeze(-1).expand_as(pred_vel)
            vel_err = torch.abs(pred_vel - target_vel) * axis_weight
            loss_vel_ps = self.maskedReducePerSample(vel_err, valid_vel)

        pred_acc = pred_vel[:, 1:, :] - pred_vel[:, :-1, :]
        target_acc = target_vel[:, 1:, :] - target_vel[:, :-1, :]
        if pred_acc.numel() == 0:
            loss_acc_ps = pred_pos.new_zeros(pred_pos.size(0))
        else:
            valid_acc = (valid_mask[:, 2:] * valid_mask[:, 1:-1] * valid_mask[:, :-2]).unsqueeze(-1).expand_as(pred_acc)
            acc_err = torch.abs(pred_acc - target_acc) * axis_weight
            loss_acc_ps = self.maskedReducePerSample(acc_err, valid_acc)

        valid_counts = valid_mask.sum(dim=1).long()
        has_valid = (valid_counts > 0).float()
        last_idx = torch.clamp(valid_counts - 1, min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, 2)
        pred_end = pred_pos.gather(1, gather_idx).squeeze(1)
        target_end = target_pos.gather(1, gather_idx).squeeze(1)
        if self.fut_pos_loss_type == "huber":
            end_err = F.smooth_l1_loss(pred_end, target_end, reduction="none", beta=self.fut_huber_delta)
        else:
            end_err = torch.abs(pred_end - target_end)
        end_axis_weight = axis_weight.squeeze(0)
        end_err = (end_err * end_axis_weight).mean(dim=-1)
        loss_end_ps = end_err * has_valid

        total_ps = (
                self.fut_loss_pos_weight * loss_pos_ps
                + self.fut_loss_vel_weight * loss_vel_ps
                + self.fut_loss_acc_weight * loss_acc_ps
                + self.fut_loss_endpoint_weight * loss_end_ps
        )
        return self.weightedBatchReduce(total_ps, sample_weight=sample_weight)

    # 根据配置选择当前使用的损失函数实现。
    def computeLoss(self, pred, target, valid_mask, sample_weight=None):
        if self.fut_loss_mode == "legacy":
            return self.computeLossLegacy(pred, target, valid_mask, sample_weight=sample_weight)
        return self.computeLossL1TimeVel(pred, target, valid_mask, sample_weight=sample_weight)

    @staticmethod
    # 计算batch级ADE和FDE指标。
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
    # 计算单条样本轨迹的ADE和FDE指标。
    def computeSingleAdeFde(pred, target, valid_mask, batch_idx=0):
        b_idx = min(max(batch_idx, 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - target[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_count = int(vm.sum().item())
        if valid_count > 0:
            fde = dist[valid_count - 1]
        else:
            fde = dist.new_tensor(0.0)
        return ade, fde

    # 在指定频率下可视化当前轨迹预测结果。
    def maybeVisualize(self, hist, future, pred, valid_mask, stage):
        if not self.is_main_process:
            return
        if stage == "train":
            self.trainForwardCalls += 1
            if (not self.fut_enable_train_vis) or (self.trainForwardCalls % self.fut_vis_every_n != 0):
                return
        else:
            self.evalForwardCalls += 1
            if (not self.fut_enable_eval_vis) or (self.evalForwardCalls % self.fut_vis_every_n != 0):
                return

        vis_batch_idx = 0
        vis_ade, vis_fde = self.computeSingleAdeFde(pred, future, valid_mask, batch_idx=vis_batch_idx)
        metrics = {
            "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
            "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        }
        visualize_batch_trajectories(
            hist=hist,
            hist_nbrs=None,
            temporal_mask=None,
            future=future,
            pred=pred,
            future_mask=valid_mask,
            batch_idx=vis_batch_idx,
            save_path=None,
            metrics=metrics,
            input_unit="ft",
            show_plot=True,
        )
        print(
            f"[{stage}][Vis Traj idx={vis_batch_idx}] ADE: {vis_ade.item():.4f} ft ({vis_ade.item() * self.meter_per_foot:.4f} m), "
            f"FDE: {vis_fde.item():.4f} ft ({vis_fde.item() * self.meter_per_foot:.4f} m)"
        )

    # 执行FUT训练前向并返回当前DiT预测结果的损失与指标。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, feat_dim = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
        future_norm = self.norm(future)
        x_start = future_norm
        noise = torch.randn_like(x_start)

        timesteps = self.sampleTrainTimesteps(bsz, device)
        x_t = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context, hist_enc)

        pred_x0_cond = torch.zeros((bsz, t_len, self.output_dim), device=device, dtype=x_t.dtype)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, device=device) < self.self_condition_prob).view(bsz, 1, 1).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
                pred_x0_cond = prev_pred.detach() * use_sc

        pred_x0_t = self.predictX0(x_t, timesteps, context, enc_emb, pred_x0_cond)
        sample_weight = self.buildNoiseSampleWeight(timesteps)
        total_loss = self.computeLoss(pred_x0_t, future_norm, valid_mask, sample_weight=sample_weight)

        pred = self.denorm(pred_x0_t)
        ade, fde = self.computeAdeFde(pred, future, valid_mask)
        self.maybeVisualize(hist, future, pred, valid_mask, stage="train")
        return total_loss, pred, ade, fde

    @torch.no_grad()
    # 执行FUT推理前向并返回多步反推后的预测结果与指标。
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        bsz, t_len, feat_dim = future.shape
        valid_mask = self.toValidMask(op_mask, t_len, bsz, device)

        x_t = torch.randn((bsz, t_len, feat_dim), device=device)
        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        enc_emb = self.encodeGlobalCondition(context, hist_enc)

        infer_scheduler = self.buildInferenceScheduler()
        x_t = self.rolloutFromXt(x_t, context, enc_emb, infer_scheduler)

        pred = self.denorm(x_t)
        pred_norm = self.norm(pred)
        future_norm = self.norm(future)
        loss = self.computeLoss(pred_norm, future_norm, valid_mask)
        ade, fde = self.computeAdeFde(pred, future, valid_mask)
        self.maybeVisualize(hist, future, pred, valid_mask, stage="eval")
        return loss, pred, ade, fde

    # 默认forward入口复用训练前向逻辑。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    # 对输入轨迹做归一化并裁剪到稳定数值范围。
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        channels = x_norm.shape[-1]
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -5.0, 5.0)
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -5.0, 5.0)
        return x_norm

    # 将归一化轨迹恢复到原始物理量纲。
    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
        channels = x.shape[-1]
        if channels >= 4:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean
        return x_denorm
