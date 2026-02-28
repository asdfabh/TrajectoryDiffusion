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
    # 初始化 Fut 扩散模型：定义网络结构、扩散调度器、归一化统计量与可视化/训练策略超参数。
    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        self.args = args

        self.hidden_dim = int(args.hidden_dim_fut)  # Fut DiT 隐藏维度。
        self.output_dim = int(args.output_dim_fut)  # Fut 输出通道维度（默认 2，对应 x/y）。
        self.heads = int(args.heads_fut)  # Fut DiT 多头注意力头数。
        self.dropout = float(args.dropout_fut)  # Fut DiT dropout 比例。
        self.depth = int(args.depth_fut)  # Fut DiT Block 堆叠层数。
        self.mlp_ratio = int(args.mlp_ratio_fut)  # Fut DiT MLP 扩展倍率。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)  # 训练扩散总步数。
        self.time_embedding_size = int(args.time_embedding_size_fut)  # 时间步正弦特征维度（再映射到 hidden_dim）。
        self.num_inference_steps = int(args.num_inference_steps)  # 推理 DDIM 采样步数。
        self.inference_timestep_spacing = str(args.inference_timestep_spacing)  # DDIM 时间步抽样方式（leading/trailing）。
        self.ddim_eta = float(args.ddim_eta)  # DDIM 随机性系数 eta。
        self.T = int(args.T_f)  # Fut 序列长度（用于 FinalLayer 配置）。

        self.self_condition_prob = min(max(float(args.self_condition_prob), 0.0), 1.0)  # 训练时 self-conditioning 触发概率。
        self.cfg_enabled = int(getattr(args, "cfg_enabled", 1)) > 0  # 是否启用 CFG 训练/推理分支。
        self.cfg_drop_prob = min(max(float(getattr(args, "cfg_drop_prob", 0.15)), 0.0), 1.0)  # 训练阶段条件丢弃概率。
        self.cfg_guidance_scale = max(0.0, float(getattr(args, "cfg_guidance_scale", 2.0)))  # 推理阶段 CFG 引导强度 w。
        self.y_loss_weight = max(1.0, float(args.fut_y_loss_weight))  # y 轴损失加权系数（>=1，强调纵向误差）。
        self.huber_delta = max(1e-4, float(args.fut_huber_delta))  # SmoothL1/Huber 的 beta(delta) 超参数。
        self.pos_loss_weight_min = max(0.0, float(getattr(args, "fut_pos_loss_weight_min", 0.0)))  # 位置损失 warmup 起始权重。
        self.pos_loss_weight_max = max(self.pos_loss_weight_min, float(getattr(args, "fut_pos_loss_weight_max", 1.0)))  # 位置损失 warmup 最终权重。
        self.pos_loss_warmup_ratio = min(max(float(getattr(args, "fut_pos_loss_warmup_ratio", 0.2)), 0.0), 1.0)  # 位置损失 warmup 占总 epoch 比例。
        self.train_epoch_idx = 0  # 当前训练 epoch（由训练脚本每轮同步）。
        self.train_total_epochs = 0  # 训练总 epoch（由训练脚本每轮同步）。
        x0_clip_val = float(args.x0_clip)  # 预测 x0 截断阈值（<=0 表示不截断）。
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None  # x0 截断阈值或 None。

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0  # 是否开启训练可视化。
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0  # 是否开启评估可视化。
        self.fut_vis_every_n = max(1, int(args.fut_vis_every_n))  # 每 N 次 forward 触发一次可视化。
        self.trainForwardCalls = 0  # 训练 forward 调用计数器。
        self.evalForwardCalls = 0  # 评估 forward 调用计数器。
        self.meter_per_foot = 0.3048  # 英尺到米的换算系数。
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0  # DDP 主进程标记（仅主进程可视化）。

        self.input_embedding = nn.Linear(self.output_dim * 2, self.hidden_dim)  # 输入投影层：拼接 [x_t, self_cond] 后映射到 hidden_dim。
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)  # Fut token 序列位置编码。
        self.hist_encoder = HistEncoder(args)  # 历史/社交条件编码器。

        context_dim = int(args.encoder_input_dim) * 2  # HistEncoder 输出上下文维度（拼接后 2 * encoder_input_dim）。
        self.enc_embedding = nn.Linear(context_dim, self.hidden_dim)  # 全局条件映射层（取 context 最后帧后再映射）。
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

        # 极其优雅的判定：同维度时省去参数与计算量
        if context_dim == self.hidden_dim:
            self.context_proj = nn.Identity()  # 上下文维度已对齐时直接透传。
        else:
            self.context_proj = nn.Linear(context_dim, self.hidden_dim)  # 上下文序列对齐到 DiT cross-attn 所需维度。
            nn.init.xavier_uniform_(self.context_proj.weight)
            nn.init.constant_(self.context_proj.bias, 0)

        self.null_global_emb = nn.Parameter(torch.zeros(1, self.hidden_dim))  # 无条件分支全局向量（CFG 用）。
        self.null_context = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))  # 无条件分支上下文序列模板（CFG 用）。

        self.timestep_embedder = dit.TimestepEmbedder(self.hidden_dim, self.time_embedding_size)  # 扩散时间步编码器。
        self.diffusion_scheduler = DDIMScheduler(  # 训练期前向加噪调度器（prediction_type=sample 即直接预测 x0）。
            num_train_timesteps=self.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.hidden_dim, self.heads, self.dropout, self.mlp_ratio)  # 单层 DiT Block 模板。
        final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)  # DiT 输出头（映射到 output_dim）。
        self.dit = dit.DiT(dit_block=dit_block, final_layer=final_layer, depth=self.depth, model_type="x_start")  # Fut 去噪主干。

        # ================= 双空间归一化参数 =================
        # 保留坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        self.register_buffer("pos_mean", torch.tensor([0.0330, -15.9150]).float(), persistent=False)  # 位置均值 [x_mean, y_mean]。
        self.register_buffer("pos_std", torch.tensor([8.8866, 68.8105]).float(), persistent=False)  # 位置标准差 [x_std, y_std]。
        self.register_buffer("va_mean", torch.tensor([21.1503, 0.0060]).float(), persistent=False)  # 速度/加速度均值 [v_mean, a_mean]。
        self.register_buffer("va_std", torch.tensor([13.5983, 4.5057]).float(), persistent=False)  # 速度/加速度标准差 [v_std, a_std]。

        # 核心新增：专门针对帧间相对位移 (Velocity) 的高斯先验参数
        # 横向 X 变道速度极小均值约 0，纵向 Y 每帧(0.2s)平均约 11ft
        self.register_buffer("vel_mean", torch.tensor([0.0066, 6.9922]).float(), persistent=False)  # 速度均值先验 [dx_mean, dy_mean]。
        self.register_buffer("vel_std", torch.tensor([0.1695, 3.0166]).float(), persistent=False)  # 速度标准差先验 [dx_std, dy_std]。

    # 将 op_mask 统一为有效帧掩码 valid_mask：支持 None/[B,T]/[B,T,C](默认 [B,T,2])，其中 1 表示有效(非掩码)、0 表示无效。
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

    # 基于训练调度器配置构建推理 DDIM 调度器，并设置推理步数与时间步间隔策略。
    def buildInferenceScheduler(self):
        try:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config,
                                                  timestep_spacing=self.inference_timestep_spacing)
        except Exception:
            scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        scheduler.set_timesteps(self.num_inference_steps)
        return scheduler

    # 从时序上下文中取最后一帧全局状态，并映射为 DiT 的全局条件向量。
    def encodeGlobalCondition(self, context):
        return self.enc_embedding(context[:, -1, :])

    # 同步训练进度：供位置损失 warmup 计算当前权重使用。
    def setTrainProgress(self, epoch_idx, total_epochs):
        self.train_epoch_idx = max(0, int(epoch_idx))
        self.train_total_epochs = max(0, int(total_epochs))

    # 计算当前 epoch 的位置损失权重 alpha，支持线性 warmup 到设定上限。
    def getPosLossWeight(self):
        if self.pos_loss_warmup_ratio <= 0.0 or self.train_total_epochs <= 0:
            return self.pos_loss_weight_max
        warmup_epochs = max(1, int(round(self.train_total_epochs * self.pos_loss_warmup_ratio)))
        if warmup_epochs <= 1:
            return self.pos_loss_weight_max
        progress_epoch = min(max(self.train_epoch_idx - 1, 0), warmup_epochs - 1)
        progress = float(progress_epoch) / float(warmup_epochs - 1)
        return self.pos_loss_weight_min + (self.pos_loss_weight_max - self.pos_loss_weight_min) * progress

    # 对条件编码执行 CFG 条件丢弃：drop_mask=1 时替换为可学习 null 条件，支持显式 mask 或按概率采样。
    def applyConditionDropout(self, context_aligned, enc_emb, drop_mask=None):
        bsz = enc_emb.size(0)
        if drop_mask is None:
            if (not self.cfg_enabled) or self.cfg_drop_prob <= 0.0:
                drop_mask = torch.zeros(bsz, dtype=torch.bool, device=enc_emb.device)
            else:
                drop_mask = torch.rand(bsz, device=enc_emb.device) < self.cfg_drop_prob
        else:
            drop_mask = drop_mask.to(enc_emb.device)
            if drop_mask.dim() > 1:
                drop_mask = drop_mask.view(bsz, -1)[:, 0]
            if drop_mask.dtype != torch.bool:
                drop_mask = drop_mask > 0.5

        if not drop_mask.any():
            return context_aligned, enc_emb, drop_mask

        drop_mask_global = drop_mask.view(bsz, 1).to(enc_emb.dtype)
        drop_mask_context = drop_mask.view(bsz, 1, 1).to(context_aligned.dtype)
        null_global = self.null_global_emb.to(enc_emb.dtype).expand(bsz, -1)
        null_context = self.null_context.to(context_aligned.dtype).expand(bsz, context_aligned.size(1), -1)

        dropped_enc_emb = enc_emb * (1.0 - drop_mask_global) + null_global * drop_mask_global
        dropped_context = context_aligned * (1.0 - drop_mask_context) + null_context * drop_mask_context
        return dropped_context, dropped_enc_emb, drop_mask

    # 单步预测 x0：输入 x_t + self-conditioning + 时间嵌入 + 历史条件，输出归一化速度(位移)。
    def predictX0(self, x_t, timesteps, context_aligned, enc_emb, pred_x0_cond):
        t_emb = self.timestep_embedder(timesteps)
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        return self.dit(x=input_embedded, y=t_emb + enc_emb, cross=context_aligned)

    # 计算双轨损失：速度域 SmoothL1 抑制抖动 + 位置域积分误差抑制累积漂移（含时间递增权重与有效掩码）。
    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask):
        """ 双轨闭环 Loss：微观治抖动 (Vel)，宏观防累积漂移 (Pos)"""
        weights = [1.0] * self.output_dim
        if self.output_dim >= 2: weights[1] = self.y_loss_weight
        axis_weight = torch.tensor(weights, device=pred_vel_norm.device, dtype=pred_vel_norm.dtype).view(1, 1,
                                                                                                         self.output_dim)

        # 1. 速度域微观约束 (Huber Loss)：在归一化空间算，梯度极度稳定，防高频抖动
        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)
        loss_vel = loss_vel * axis_weight

        # 2. 积分回物理位置域 (宏观绝对约束)：全局可导梯度反传，彻底打断累积误差！
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)

        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        # Cumsum 将速度积分还原为绝对坐标
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]

        # 将位置误差除以位置标准差，让宏观梯度的尺度与微观梯度保持对等公平
        std_pos = self.pos_std.view(1, 1, 2).to(pred_vel_norm.device)
        pos_diff = (pred_pos_phys - future_phys[..., :2]) / std_pos
        loss_pos = torch.abs(pos_diff)

        # 维度安全补齐 (应对 output_dim > 2 的情况)
        if self.output_dim > 2:
            pad = torch.zeros(loss_pos.shape[:-1] + (self.output_dim - 2,), device=loss_pos.device,
                              dtype=loss_pos.dtype)
            loss_pos = torch.cat([loss_pos, pad], dim=-1)

        # 施加时间递增权重，逼迫模型对远端 FDE 负责
        T_len = loss_pos.size(1)
        time_weights = torch.linspace(1.0, 2.0, T_len, device=loss_pos.device, dtype=loss_pos.dtype).view(1, T_len, 1)
        loss_pos = loss_pos * time_weights * axis_weight

        # 双轨合一
        pos_weight = self.getPosLossWeight()
        total_loss = loss_vel + pos_weight * loss_pos

        valid = valid_mask.unsqueeze(-1)
        numer = (total_loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

    # 计算批量 ADE/FDE：ADE 用所有有效点平均距离，FDE 取每条轨迹最后一个有效时刻距离。
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

    # 计算单条样本 ADE/FDE：用于可视化与调试时展示指定 batch 索引轨迹误差。
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

    # 执行 DDIM 去噪 rollout：支持标准条件采样与 CFG 双分支外推，并在每步使用 self-conditioning。
    def rolloutFromXt(self, x_t, context_aligned, enc_emb, infer_scheduler, use_cfg=False, guidance_scale=None):
        bsz, t_len, _ = x_t.shape
        guidance = self.cfg_guidance_scale if guidance_scale is None else float(guidance_scale)
        run_cfg = bool(self.cfg_enabled and use_cfg and guidance > 1.0)

        pred_vel_cond = torch.zeros((bsz, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        pred_vel_uncond = None
        if run_cfg:
            null_mask = torch.ones(bsz, dtype=torch.bool, device=x_t.device)
            null_context, null_emb, _ = self.applyConditionDropout(context_aligned, enc_emb, drop_mask=null_mask)
            pred_vel_uncond = torch.zeros_like(pred_vel_cond)

        pred_vel_final = pred_vel_cond
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((bsz,), t_scalar, device=x_t.device, dtype=torch.long)

            if run_cfg:
                pred_vel_norm_uncond = self.predictX0(x_t, timesteps, null_context, null_emb, pred_vel_uncond)
                pred_vel_norm_cond = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_vel_cond)
                if self.x0_clip is not None:
                    pred_vel_norm_uncond = torch.clamp(pred_vel_norm_uncond, -self.x0_clip, self.x0_clip)
                    pred_vel_norm_cond = torch.clamp(pred_vel_norm_cond, -self.x0_clip, self.x0_clip)
                pred_vel_norm = pred_vel_norm_uncond + guidance * (pred_vel_norm_cond - pred_vel_norm_uncond)
            else:
                pred_vel_norm = self.predictX0(x_t, timesteps, context_aligned, enc_emb, pred_vel_cond)
                if self.x0_clip is not None:
                    pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)

            if run_cfg and self.x0_clip is not None:
                pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)

            if run_cfg:
                pred_vel_cond = pred_vel_norm_cond.detach()
                pred_vel_uncond = pred_vel_norm_uncond.detach()
            else:
                pred_vel_cond = pred_vel_norm.detach()
            pred_vel_final = pred_vel_norm.detach()

            try:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t).prev_sample
        return pred_vel_final

    # 按训练/评估阶段与频率条件执行轨迹可视化，默认仅主进程启用以避免 DDP 重复绘图。
    def maybeVisualize(self, hist, future, pred, valid_mask, stage):
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
        visualize_batch_trajectories(hist=hist, hist_nbrs=None, temporal_mask=None, future=future, pred=pred,
                                     future_mask=valid_mask, batch_idx=vis_batch_idx, save_path=None, metrics=metrics,
                                     input_unit="ft", show_plot=True)

    # 训练前向：构造速度监督并加噪、提取条件编码、预测 x0(速度)、计算双轨损失并可选可视化。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
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
        context_aligned_used, enc_emb_used, _ = self.applyConditionDropout(context_aligned, enc_emb)

        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(bsz, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(x_t, timesteps, context_aligned_used, enc_emb_used, pred_vel_cond)
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        # 网络输出：预测的归一化速度
        pred_vel_norm_t = self.predictX0(x_t, timesteps, context_aligned_used, enc_emb_used, pred_vel_cond)

        # 传入双轨 Loss 防累积漂移
        loss = self.computeLoss(pred_vel_norm_t, target_vel_norm, future_phys, anchor_phys, valid_mask)

        if self.fut_enable_train_vis:
            pred_vel_phys_t = pred_vel_norm_t[..., :2] * std_vel + mean_vel
            pred_pos_phys = torch.cumsum(pred_vel_phys_t, dim=1) + anchor_phys[..., :2]

            # 兼容多维度输出拼接
            pred_phys_abs = future_phys.clone()
            pred_phys_abs[..., :2] = pred_pos_phys
            self.maybeVisualize(hist, future, pred_phys_abs, valid_mask, stage="train")

        return loss

    # 评估前向：从纯噪声 rollout 生成未来轨迹，返回评估损失及 ADE/FDE 指标。
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
        use_cfg = self.cfg_enabled and self.cfg_guidance_scale > 1.0
        pred_vel_norm = self.rolloutFromXt(
            x_t,
            context_aligned,
            enc_emb,
            infer_scheduler,
            use_cfg=use_cfg,
            guidance_scale=self.cfg_guidance_scale,
        )

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
        loss = self.computeLoss(pred_vel_norm, torch.clamp(target_vel_norm, -10.0, 10.0), future_phys, anchor_phys,
                                valid_mask)

        ade, fde = self.computeAdeFde(pred_phys_abs, future, valid_mask)
        self.maybeVisualize(hist, future, pred_phys_abs, valid_mask, stage="eval")
        return loss, pred_phys_abs, ade, fde

    @torch.no_grad()
    # def forwardEval_minADE(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, K=5):
    #     """ OOM-Safe SOTA 多模态评估 (minADE_K)"""
    #     bsz, t_len, _ = future.shape
    #     valid_mask = self.toValidMask(op_mask, t_len, bsz, device)
    #     anchor_phys = hist[..., -1:, :self.output_dim]
    #     future_phys = future[..., :self.output_dim]
    #
    #     hist_norm = self.norm(hist)
    #     context, _ = self.hist_encoder(hist_norm, self.norm(hist_nbrs), mask, temporal_mask)
    #     enc_emb = self.encodeGlobalCondition(context)
    #     context_aligned = self.context_proj(context)
    #
    #     infer_scheduler = self.buildInferenceScheduler()
    #     std_vel = self.vel_std.view(1, 1, 2).to(device)
    #     mean_vel = self.vel_mean.view(1, 1, 2).to(device)
    #
    #     all_preds = []
    #     for _ in range(K):
    #         # 每次给予完全独立的随机速度噪声，产生平行宇宙分化
    #         x_t = torch.randn((bsz, t_len, self.output_dim), device=device)
    #         pred_vel_norm = self.rolloutFromXt(x_t, context_aligned, enc_emb, infer_scheduler)
    #         pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
    #         pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
    #
    #         pred_phys_abs = future_phys.clone()
    #         pred_phys_abs[..., :2] = pred_pos_phys
    #         all_preds.append(pred_phys_abs)
    #
    #     all_preds = torch.stack(all_preds, dim=1)  # [B, K, T, dim]
    #
    #     # 计算 minADE_K
    #     target_phys = future_phys[..., :2].unsqueeze(1)
    #     diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)  # [B, K, T]
    #     valid_mask_exp = valid_mask.unsqueeze(1)  # [B, 1, T]
    #
    #     ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)  # [B, K]
    #     min_ade, best_k_idx = torch.min(ade_k, dim=1)  # [B]
    #
    #     best_k_idx_exp = best_k_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, all_preds.size(-1))
    #     best_pred_phys = all_preds.gather(1, best_k_idx_exp).squeeze(1)  # [B, T, dim]
    #
    #     ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
    #     dummy_loss = torch.tensor(0.0, device=device)
    #     self.maybeVisualize(hist, future, best_pred_phys, valid_mask, stage="eval")
    #
    #     return dummy_loss, best_pred_phys, ade_batch, fde_batch

    # 多模态评估(minADE_K)：并行采样 K 条候选轨迹并选最小 ADE 的一条用于打分。
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

        # 一次性生成 bsz * K 份随机噪声，并行执行完整去噪（可选 CFG 引导）。
        x_t_k = torch.randn((bsz * K, t_len, self.output_dim), device=device)
        use_cfg = self.cfg_enabled and self.cfg_guidance_scale > 1.0
        pred_vel_norm_k = self.rolloutFromXt(
            x_t_k,
            context_aligned_k,
            enc_emb_k,
            infer_scheduler,
            use_cfg=use_cfg,
            guidance_scale=self.cfg_guidance_scale,
        )

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

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

        ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
        dummy_loss = torch.tensor(0.0, device=device)
        self.maybeVisualize(hist, future, best_pred_phys, valid_mask, stage="eval")

        return dummy_loss, best_pred_phys, ade_batch, fde_batch

    # 统一前向入口：默认复用训练路径 forwardTrain。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device)

    # 输入归一化：位置通道按 pos 统计量归一化，若存在 VA 通道则按 va 统计量归一化并裁剪。
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -10.0, 10.0)
        channels = x_norm.shape[-1]
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -10.0, 10.0)
        return x_norm

    # 输入反归一化：将位置/VA 通道还原到物理量纲空间。
    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
        channels = x.shape[-1]
        if channels >= 4:
            x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean
        return x_denorm
