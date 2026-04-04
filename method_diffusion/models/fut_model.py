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
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()

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

        # 扩散与推理参数：控制训练时间步、推理步数和 DDIM 采样行为。
        self.num_train_timesteps = int(args.num_train_timesteps_fut)
        self.num_inference_steps = int(args.num_inference_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.x0_clip = float(args.x0_clip) if float(args.x0_clip) > 0 else None

        # 输入编码模块：分别处理 future 噪声序列和 history context。
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.context_embedding = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)

        # DiT 主干与扩散调度器：负责时间嵌入、去噪建模和 DDIM 调度。
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

        # 双空间归一化参数
        # 物理坐标归一化参数 (给历史轨迹编码和宏观位置 Loss 使用)
        if self.dataset_name == "ngsim":
            self.register_buffer("pos_mean", torch.tensor([0.05076411229651117, -31.318518632454474], dtype=torch.float32), persistent=False)
            self.register_buffer("pos_std", torch.tensor([9.67614343193339, 59.53730335210165], dtype=torch.float32), persistent=False)
            self.register_buffer("va_mean", torch.tensor([21.150308365503957, 0.006041414014469039], dtype=torch.float32), persistent=False)
            self.register_buffer("va_std", torch.tensor([13.598306447881924, 4.505736504111998], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_mean", torch.tensor([-0.004181504611623526, 5.041936610524995], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.1502223350250087, 2.951254134709027], dtype=torch.float32), persistent=False)
        elif self.dataset_name == "highd":
            self.register_buffer("pos_mean", torch.tensor([-0.39106148272179536, -115.63853904936501], dtype=torch.float32), persistent=False)
            self.register_buffer("pos_std", torch.tensor([9.266579303046143, 98.49671326349531], dtype=torch.float32), persistent=False)
            self.register_buffer("va_mean", torch.tensor([78.09292302772707, -0.04991240184019581], dtype=torch.float32), persistent=False)
            self.register_buffer("va_std", torch.tensor([29.215909315170098, 1.1700240860076556], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_mean", torch.tensor([0.004845835373614644, 17.01558226555126], dtype=torch.float32), persistent=False)
            self.register_buffer("vel_std", torch.tensor([0.10621210903901461, 4.838376260255577], dtype=torch.float32), persistent=False)
        else:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}' for fut normalization. Supported: highd, ngsim"
            )

    # 基于当前噪声状态与 history context 预测归一化速度 x0。
    def predictX0(self, x_t, timesteps, context_tokens):
        t_emb = self.timestep_embedder(timesteps)
        input_embedded = self.input_embedding(x_t) + self.pos_embedding(x_t)
        cross_encoded = self.context_embedding(context_tokens)
        return self.dit(x=input_embedded, t_cond=t_emb, cross=cross_encoded)

    def encodeContext(self, hist, hist_nbrs, mask, temporal_mask):
        hist_norm = self.normalize(hist)
        hist_nbrs_norm = self.normalize(hist_nbrs)
        context, _ = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        return context

    def computeLoss(self, pred_vel_norm, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=False):
        # Velocity loss (speed loss)
        loss_vel = F.l1_loss(pred_vel_norm, target_vel_norm, reduction="none")

        # Position loss - convert normalized velocity to physical coordinates and compute position loss
        std_vel = self.vel_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        loss_pos = F.l1_loss(pred_pos_phys, future_phys[..., :2], reduction="none")

        # Combined loss: velocity + 0.5 * position (fixed weight, not from config)
        pos_loss_weight = 0.5
        total_loss = loss_vel + pos_loss_weight * loss_pos

        valid = valid_mask.unsqueeze(-1)
        numer = (total_loss * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        total_mean = (numer / denom).mean()

        if not return_parts:
            return total_mean

        vel_mean = ((loss_vel * valid).sum(dim=(1, 2)) / denom).mean()
        pos_mean = ((loss_pos * valid).sum(dim=(1, 2)) / denom).mean()
        parts = {
            "loss_total": total_mean.detach(),
            "loss_diffusion": total_mean.detach(),
            "loss_x0": total_mean.detach(),
            "loss_vel": vel_mean.detach(),
            "loss_pos": pos_mean.detach(),
        }
        return total_mean, parts

    # 执行推理阶段的 DDIM 采样并输出最终预测结果（归一化速度 x0）。
    def sampleFromXt(self, x_t, context_tokens, infer_scheduler):
        pred_x0 = None
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((x_t.size(0),), t_scalar, device=x_t.device, dtype=torch.long)
            pred_x0 = self.predictX0(x_t, timesteps, context_tokens)
            if self.x0_clip is not None:
                pred_x0 = torch.clamp(pred_x0, -self.x0_clip, self.x0_clip)
            try:
                x_t = infer_scheduler.step(pred_x0, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_x0, t, x_t).prev_sample
        return pred_x0

    # 统一准备评估阶段所需的条件编码、掩码和调度器参数。
    def prepareEvalInputs(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        bsz, t_len, _ = future.shape
        anchor_phys = hist[:, -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        context_tokens = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)
        infer_scheduler = DDIMScheduler.from_config(self.diffusion_scheduler.config)
        infer_scheduler.set_timesteps(self.num_inference_steps)
        return bsz, t_len, anchor_phys, future_phys, context_tokens, infer_scheduler

    # 执行单个 batch 的 fut 训练前向与损失计算。
    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        bsz, t_len, _ = future.shape  # [B, T_f, D]
        valid_mask = (op_mask[..., 0] > 0.5).float().to(device)  # [B, T_f]

        anchor_phys = hist[:, -1:, :self.output_dim]  # [B, 1, 2]
        future_phys = future[..., :self.output_dim]  # [B, T_f, 2]

        # 坐标空间转换：计算 GT 的真实物理帧间位移 (Velocity)。
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys  # [B, T_f, 2]

        # 归一化：将物理速度变为正态分布。
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -5.0, 5.0)

        # 加噪过程。
        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (bsz,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        context_tokens = self.encodeContext(hist, hist_nbrs, mask, temporal_mask)

        # 网络输出：预测的归一化速度 x0。
        pred_x0 = self.predictX0(x_t, timesteps, context_tokens)

        loss, loss_parts = self.computeLoss(pred_x0, target_vel_norm, future_phys, anchor_phys, valid_mask, return_parts=True)

        if return_components:
            return loss, loss_parts
        return loss

    @torch.no_grad()
    # 执行单模态推理评估并返回预测轨迹。
    def forwardEval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        bsz, t_len, anchor_phys, future_phys, context_tokens, infer_scheduler = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, device)

        x_t = torch.randn((bsz, t_len, self.input_dim), device=device)
        pred_vel_norm = self.sampleFromXt(x_t, context_tokens, infer_scheduler)

        # 积分还原：解归一化 -> 累加 -> 拼接绝对锚点。
        std_vel = self.vel_std.view(1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        pred_phys_abs = future_phys.clone()
        pred_phys_abs[..., :2] = pred_pos_phys

        return pred_phys_abs

    @torch.no_grad()
    # 执行并行多模态推理评估并返回多模态预测轨迹集合。
    def forwardEvalMulti(self, hist, hist_nbrs, mask, temporal_mask, future, device, K=5):
        bsz, t_len, anchor_phys, future_phys, context_tokens, infer_scheduler = self.prepareEvalInputs(hist, hist_nbrs, mask, temporal_mask, future, device)

        # 核心提速优化：在 Batch 维度上并行展开 K 倍。
        context_tokens_k = context_tokens.repeat_interleave(K, dim=0)

        # 一次性生成 bsz * K 份随机噪声，并行执行完整的去噪过程。
        x_t_k = torch.randn((bsz * K, t_len, self.input_dim), device=device)
        pred_vel_norm_k = self.sampleFromXt(x_t_k, context_tokens_k, infer_scheduler)

        # 把并发结果 Reshape 回 [bsz, K, t_len, dim]。
        pred_vel_norm = pred_vel_norm_k.view(bsz, K, t_len, self.output_dim)

        # 积分还原物理坐标 (张量广播)。
        std_vel = self.vel_std.view(1, 1, 1, 2).to(device)
        mean_vel = self.vel_mean.view(1, 1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        anchor_phys_k = anchor_phys[..., :2].unsqueeze(1)
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys_k

        # 兼容多维度输出，前两维写入还原后的物理坐标。
        all_preds = future_phys.unsqueeze(1).repeat(1, K, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys

        return all_preds

    # 统一前向入口，默认复用训练路径。
    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=False):
        return self.forwardTrain(hist, hist_nbrs, mask, temporal_mask, future, op_mask, device, return_components=return_components)

    # 对历史输入的坐标与运动学特征做归一化。
    def normalize(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
        x_norm[..., 0:2] = torch.clamp(x_norm[..., 0:2], -10.0, 10.0)
        channels = x_norm.shape[-1]
        if channels >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 2:4] = torch.clamp(x_norm[..., 2:4], -10.0, 10.0)
        return x_norm
