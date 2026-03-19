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
    """future 轨迹扩散模型，使用显式意图条件引导去噪。"""

    def __init__(self, args):
        """初始化历史编码器、意图条件和 DiT 去噪器。

        Args:
            args: future 分支的配置对象，包含模型规模、采样器和损失权重等参数。
        """
        super(DiffusionFut, self).__init__()
        self.args = args
        if int(args.feature_dim) != 4:
            raise ValueError("Current future branch requires feature_dim=4: [rel_x, rel_y, v, a]")

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
        self.loss_w_vel = 1.0
        self.loss_w_pos = max(0.0, float(args.fut_pos_loss_weight))
        self.loss_w_lat = max(0.0, float(getattr(args, "intent_loss_weight_lat", 0.20)))
        self.loss_w_lon = max(0.0, float(getattr(args, "intent_loss_weight_lon", 0.20)))

        x0_clip_val = float(args.x0_clip)
        self.x0_clip = x0_clip_val if x0_clip_val > 0 else None

        self.fut_enable_train_vis = int(args.fut_enable_train_vis) > 0
        self.fut_enable_eval_vis = int(args.fut_enable_eval_vis) > 0
        self.meter_per_foot = 0.3048
        self.is_main_process = int(os.environ.get("RANK", "0")) == 0

        self.register_buffer("lat_class_weight", torch.ones(3, dtype=torch.float32))
        self.register_buffer("lon_class_weight", torch.ones(3, dtype=torch.float32))
        for name, values in {
            "hist_pos_mean": [0.05130798, -35.39044909],
            "hist_pos_std": [9.63184438, 60.19290744],
            "hist_va_mean": [23.92449619, 0.04203195],
            "hist_va_std": [13.34587118, 4.61229342],
            "fut_delta_mean": [-0.00407632, 5.53086274],
            "fut_delta_std": [0.15694872, 2.88311335],
        }.items():
            self.register_buffer(name, torch.tensor(values, dtype=torch.float32), persistent=False)

        self.input_embedding = nn.Linear(self.output_dim * 2, self.hidden_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.hidden_dim)
        self.hist_encoder = HistEncoder(args)
        if int(getattr(self.hist_encoder, "hidden_dim", self.hidden_dim)) != self.hidden_dim:
            raise ValueError(
                f"HistEncoder hidden_dim must match hidden_dim_fut, got {getattr(self.hist_encoder, 'hidden_dim', None)} vs {self.hidden_dim}"
            )

        self.cond_projs = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.depth)])

        self.lat_emb = nn.Embedding(3, self.hidden_dim)
        self.lon_emb = nn.Embedding(3, self.hidden_dim)
        self.intent_fuse = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

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
        self.initModelWeights()

    def initModelWeights(self):
        """统一初始化 future 主模型内的显式参数层。"""
        init_linear = lambda layer: (nn.init.xavier_uniform_(layer.weight), nn.init.zeros_(layer.bias))
        for layer in (self.input_embedding, *self.cond_projs): init_linear(layer)
        self.intent_fuse.apply(lambda layer: init_linear(layer) if isinstance(layer, nn.Linear) else None)
        for embedding in (self.lat_emb, self.lon_emb): nn.init.normal_(embedding.weight, std=0.02)

    def set_intent_class_weights(self, lat_weight, lon_weight):
        """更新横纵向意图损失的类别权重。

        Args:
            lat_weight: 横向意图类别权重，形状为 `[3]`。
            lon_weight: 纵向意图类别权重，形状为 `[3]`。
        """
        self.lat_class_weight.copy_(lat_weight.detach().to(self.lat_class_weight.device, dtype=self.lat_class_weight.dtype))
        self.lon_class_weight.copy_(lon_weight.detach().to(self.lon_class_weight.device, dtype=self.lon_class_weight.dtype))

    def normalizeExtras(self, extras, hist, hist_nbrs, device):
        """补齐 future 分支需要的附加监督与几何字段。

        Args:
            extras: 调用方传入的附加字段字典。
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            device: 目标设备。

        Returns:
            包含车道、距离和意图监督标签的完整字典。
        """
        if extras is None:
            extras = {}
        batch_size, hist_len, _ = hist.shape
        nbr_total = hist_nbrs.size(0)
        default = {
            "ego_lane": torch.zeros(batch_size, hist_len, 1, device=device, dtype=hist.dtype),
            "nbr_lane": torch.zeros(nbr_total, hist_len, 1, device=device, dtype=hist.dtype),
            "lat_gt": None,
            "lon_gt": None,
        }
        merged = {}
        for key, value in default.items():
            merged[key] = extras.get(key, value)
        return merged

    def resolveForwardInputs(self, hist, hist_nbrs, extras=None, device=None):
        """统一 device 与 extras 的默认值。

        Args:
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            extras: 可选附加字段字典。
            device: 可选目标设备。

        Returns:
            一个二元组：
            - extras: 补齐后的附加字段字典。
            - device: 最终使用的设备对象。
        """
        if device is None:
            device = hist.device
        extras = self.normalizeExtras(extras, hist, hist_nbrs, device)
        return extras, device

    def buildTargetVelNorm(self, future_phys, anchor_phys, device):
        """将未来绝对位置转换为归一化速度增量监督。

        Args:
            future_phys: 未来绝对位置序列。
            anchor_phys: 历史最后时刻的锚点位置。
            device: 目标设备。

        Returns:
            一个二元组：
            - target_vel_norm: 归一化后的速度增量监督。
            - target_vel_phys: 物理量空间下的速度增量监督。
        """
        shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
        target_vel_phys = future_phys - shifted_future_phys
        std_vel = self.fut_delta_std.view(1, 1, 2).to(device)
        mean_vel = self.fut_delta_mean.view(1, 1, 2).to(device)
        target_vel_norm = target_vel_phys.clone()
        target_vel_norm[..., :2] = (target_vel_phys[..., :2] - mean_vel) / std_vel
        target_vel_norm[..., :2] = torch.clamp(target_vel_norm[..., :2], -10.0, 10.0)
        return target_vel_norm, target_vel_phys

    def buildIntentCondition(self, lat_logits, lon_logits, lat_gt=None, lon_gt=None, epoch=None, training=False):
        """构造训练或推理阶段使用的离散意图条件。

        Args:
            lat_logits: 横向意图 logits。
            lon_logits: 纵向意图 logits。
            lat_gt: 可选横向真实标签。
            lon_gt: 可选纵向真实标签。
            epoch: 当前训练轮次。
            training: 是否处于训练阶段。

        Returns:
            注入 DiT AdaLN 的离散意图条件向量。
        """
        if training and lat_gt is not None and lon_gt is not None:
            tf_epochs = int(getattr(self.args, "intent_teacher_forcing_epochs", 10))
            if tf_epochs <= 0:
                use_gt = torch.zeros(lat_logits.size(0), dtype=torch.bool, device=lat_logits.device)
            else:
                p_gt = max(0.0, 1.0 - float(epoch or 1) / float(tf_epochs))
                use_gt = torch.rand(lat_logits.size(0), device=lat_logits.device) < p_gt
            lat_pred = lat_logits.argmax(dim=-1)
            lon_pred = lon_logits.argmax(dim=-1)
            lat_idx = torch.where(use_gt, lat_gt, lat_pred)
            lon_idx = torch.where(use_gt, lon_gt, lon_pred)
        else:
            lat_idx = lat_logits.argmax(dim=-1)
            lon_idx = lon_logits.argmax(dim=-1)
        return self.intent_fuse(self.lat_emb(lat_idx) + self.lon_emb(lon_idx))

    def computeLoss(
        self,
        pred_vel_norm,
        target_vel_norm,
        future_phys,
        anchor_phys,
        valid_mask,
        lat_logits=None,
        lon_logits=None,
        lat_gt=None,
        lon_gt=None,
        return_components=False,
    ):
        """计算速度、位置和显式意图的联合损失。

        Args:
            pred_vel_norm: 预测的归一化速度增量。
            target_vel_norm: 真实的归一化速度增量。
            future_phys: 真实未来绝对轨迹。
            anchor_phys: 历史最后时刻的锚点位置。
            valid_mask: 未来有效位掩码。
            lat_logits: 可选的横向意图 logits。
            lon_logits: 可选的纵向意图 logits。
            lat_gt: 可选的横向意图标签。
            lon_gt: 可选的纵向意图标签。
            return_components: 是否同时返回日志分量。

        Returns:
            当 `return_components=False` 时返回总损失；
            否则返回 `(loss, loss_metrics)`。
        """
        if pred_vel_norm.size(-1) != 2 or target_vel_norm.size(-1) != 2:
            raise ValueError(
                f"computeLoss currently expects dim=2, got pred={pred_vel_norm.size(-1)}, target={target_vel_norm.size(-1)}"
            )

        loss_vel = F.smooth_l1_loss(pred_vel_norm, target_vel_norm, reduction="none", beta=self.huber_delta)

        std_vel = self.fut_delta_std.view(1, 1, 2).to(pred_vel_norm.device)
        mean_vel = self.fut_delta_mean.view(1, 1, 2).to(pred_vel_norm.device)
        pred_vel_phys = pred_vel_norm * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=1) + anchor_phys[..., :2]
        gt_pos_phys = future_phys[..., :2]
        loss_pos = F.smooth_l1_loss(pred_pos_phys, gt_pos_phys, reduction="none", beta=self.huber_delta)

        loss_vel_mean = self.maskedMean3d(loss_vel, valid_mask)
        loss_pos_mean = self.maskedMean3d(loss_pos, valid_mask)

        zero_scalar = pred_vel_norm.new_tensor(0.0)
        loss_lat = zero_scalar
        loss_lon = zero_scalar
        acc_lat = zero_scalar
        acc_lon = zero_scalar

        if lat_logits is not None and lat_gt is not None:
            loss_lat = F.cross_entropy(lat_logits, lat_gt, weight=self.lat_class_weight.to(lat_logits.device))
            acc_lat = (lat_logits.argmax(dim=-1) == lat_gt).float().mean()
        if lon_logits is not None and lon_gt is not None:
            loss_lon = F.cross_entropy(lon_logits, lon_gt, weight=self.lon_class_weight.to(lon_logits.device))
            acc_lon = (lon_logits.argmax(dim=-1) == lon_gt).float().mean()

        loss = (
            self.loss_w_vel * loss_vel_mean
            + self.loss_w_pos * loss_pos_mean
            + self.loss_w_lat * loss_lat
            + self.loss_w_lon * loss_lon
        )
        if not return_components:
            return loss

        loss_metrics = self.summarizeLossForLog(
            loss_vel=loss_vel,
            loss_pos=loss_pos,
            valid_mask=valid_mask,
            loss_total=loss.detach(),
            loss_lat=loss_lat.detach(),
            loss_lon=loss_lon.detach(),
            acc_lat=acc_lat.detach(),
            acc_lon=acc_lon.detach(),
        )
        return loss, loss_metrics

    @staticmethod
    def maskedMean3d(loss_tensor, valid_mask):
        """按有效未来点对三维损失张量做均值。

        Args:
            loss_tensor: 形状为 `[B, T, C]` 的损失张量。
            valid_mask: 形状为 `[B, T]` 的有效位掩码。

        Returns:
            全 batch 的标量平均损失。
        """
        valid = valid_mask.unsqueeze(-1)
        numer = (loss_tensor * valid).sum(dim=(1, 2))
        denom = valid.sum(dim=(1, 2)) + 1e-6
        return (numer / denom).mean()

    def summarizeLossForLog(self, loss_vel, loss_pos, valid_mask, loss_total, loss_lat, loss_lon, acc_lat, acc_lon):
        """整理训练日志需要的损失与精度标量。

        Args:
            loss_vel: 速度损失张量。
            loss_pos: 位置损失张量。
            valid_mask: 有效位掩码。
            loss_total: 总损失标量。
            loss_lat: 横向意图损失标量。
            loss_lon: 纵向意图损失标量。
            acc_lat: 横向意图精度。
            acc_lon: 纵向意图精度。

        Returns:
            记录日志所需的指标字典。
        """
        with torch.no_grad():
            return {
                "loss_total": loss_total,
                "loss_vel": self.maskedMean3d(loss_vel.detach(), valid_mask),
                "loss_pos": self.maskedMean3d(loss_pos.detach(), valid_mask),
                "loss_lat": loss_lat,
                "loss_lon": loss_lon,
                "acc_lat": acc_lat,
                "acc_lon": acc_lon,
            }

    @staticmethod
    def computeAdeFde(pred, target, valid_mask):
        """计算 batch 级别的 ADE / FDE。

        Args:
            pred: 预测轨迹。
            target: 真实轨迹。
            valid_mask: 有效位掩码。

        Returns:
            一个二元组 `(ade, fde)`。
        """
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

    def normHistoryInput(self, x):
        """标准化历史主干输入 `[x, y, v, a]`。

        Args:
            x: 原始历史状态张量。

        Returns:
            标准化后的历史状态张量。
        """
        x_norm = x.clone()
        if x_norm.numel() == 0:
            return x_norm
        x_norm[..., 0:2] = (x[..., 0:2] - self.hist_pos_mean) / self.hist_pos_std
        x_norm[..., 2:4] = (x[..., 2:4] - self.hist_va_mean) / self.hist_va_std
        x_norm = torch.clamp(x_norm, -10.0, 10.0)
        return x_norm

    def encodeHistoryCondition(self, hist, hist_nbrs, mask, temporal_mask, extras):
        """编码历史交互上下文并输出显式意图 logits。

        Args:
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            mask: 邻车存在性掩码。
            temporal_mask: temporal 分支使用的邻车掩码。
            extras: 车道、距离和监督标签字典。

        Returns:
            一个四元组：
            - memory_tokens: cross-attn 使用的上下文 token。
            - memory_mask: 上下文 token 的 padding mask。
            - lat_logits: 横向意图 logits。
            - lon_logits: 纵向意图 logits。
        """
        hist_state_norm = self.normHistoryInput(hist)
        nbr_state_norm = self.normHistoryInput(hist_nbrs)
        memory_tokens, memory_mask, lat_logits, lon_logits = self.hist_encoder(
            hist_state_norm,
            nbr_state_norm,
            mask,
            temporal_mask,
            extras["ego_lane"],
            extras["nbr_lane"],
            ego_state_raw=hist,
            nbr_state_raw=hist_nbrs,
        )
        return memory_tokens, memory_mask, lat_logits, lon_logits

    def predictX0(self, x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_x0_cond):
        """在给定交互记忆和意图条件下预测 x0。

        Args:
            x_t: 当前噪声状态。
            timesteps: 当前扩散步编号。
            memory_tokens: 历史交互记忆 token。
            intent_cond: 意图条件向量。
            memory_mask: 历史记忆的 padding mask。
            pred_x0_cond: self-conditioning 使用的前一步预测。

        Returns:
            当前步预测得到的 `x0`。
        """
        t_emb = self.timestep_embedder(timesteps)
        y_layers = [t_emb + proj(intent_cond) for proj in self.cond_projs]
        combined_input = torch.cat([x_t, pred_x0_cond], dim=-1)
        input_embedded = self.input_embedding(combined_input) + self.pos_embedding(x_t)
        return self.dit(x=input_embedded, y=y_layers, cross=memory_tokens, cross_attn_mask=memory_mask)

    def rolloutFromXt(self, x_t, memory_tokens, intent_cond, memory_mask, infer_scheduler):
        """从纯噪声迭代回滚到最终速度增量预测。

        Args:
            x_t: 初始噪声张量。
            memory_tokens: 历史交互记忆 token。
            intent_cond: 意图条件向量。
            memory_mask: 历史记忆的 padding mask。
            infer_scheduler: 推理阶段使用的 DDIM 调度器。

        Returns:
            最终预测得到的归一化速度增量。
        """
        batch_size, t_len, _ = x_t.shape
        pred_vel_cond = torch.zeros((batch_size, t_len, self.output_dim), device=x_t.device, dtype=x_t.dtype)
        for t in infer_scheduler.timesteps:
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            timesteps = torch.full((batch_size,), t_scalar, device=x_t.device, dtype=torch.long)
            pred_vel_norm = self.predictX0(x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_vel_cond)
            if self.x0_clip is not None:
                pred_vel_norm = torch.clamp(pred_vel_norm, -self.x0_clip, self.x0_clip)
            pred_vel_cond = pred_vel_norm.detach()
            try:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t, eta=self.ddim_eta).prev_sample
            except TypeError:
                x_t = infer_scheduler.step(pred_vel_norm, t, x_t).prev_sample
        return pred_vel_cond

    def maybeVisualize(self, hist, hist_nbrs, temporal_mask, future, pred, valid_mask, stage, pred_all=None, pred_best_idx=None):
        """按配置决定是否绘制训练或评估样本。

        Args:
            hist: ego 历史轨迹。
            hist_nbrs: 邻车历史轨迹。
            temporal_mask: 邻车恢复与可视化使用的掩码。
            future: 真实未来轨迹。
            pred: 预测未来轨迹。
            valid_mask: 未来有效位掩码。
            stage: 当前阶段，取值为 `train` 或 `eval`。
            pred_all: 可选的多模态候选预测。
            pred_best_idx: 可选的最佳候选索引。

        Returns:
            无返回值。
        """
        if not self.is_main_process:
            return
        if stage == "train":
            if not self.fut_enable_train_vis:
                return
        else:
            if not self.fut_enable_eval_vis:
                return

        vis_batch_idx = 0
        b_idx = min(max(int(vis_batch_idx), 0), pred.size(0) - 1)
        diff = pred[b_idx, :, :2] - future[b_idx, :, :2]
        dist = torch.norm(diff, dim=-1)
        vm = valid_mask[b_idx]
        vis_ade = (dist * vm).sum() / (vm.sum() + 1e-6)
        valid_idx = torch.nonzero(vm > 0, as_tuple=False).squeeze(-1)
        vis_fde = dist[valid_idx[-1]] if valid_idx.numel() > 0 else dist.new_tensor(0.0)
        metrics = {
            "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * self.meter_per_foot},
            "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * self.meter_per_foot},
        }
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

    def forwardTrain(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, epoch=None, return_components=False):
        """执行 future 分支的训练前向与联合损失计算。

        Args:
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            mask: 邻车存在性掩码。
            temporal_mask: 邻车可视化掩码。
            future: 真实未来轨迹。
            op_mask: 未来有效位掩码。
            extras: 车道、距离与意图标签字典。
            device: 目标设备。
            epoch: 当前训练轮次，用于意图 teacher forcing 调度。
            return_components: 是否返回日志分量。

        Returns:
            当 `return_components=False` 时返回总损失；
            否则返回 `(loss, loss_metrics)`。
        """
        extras, device = self.resolveForwardInputs(hist, hist_nbrs, extras, device)
        batch_size = future.shape[0]
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]
        target_vel_norm, _ = self.buildTargetVelNorm(future_phys, anchor_phys, device)

        noise = torch.randn_like(target_vel_norm)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device).long()
        x_t = self.diffusion_scheduler.add_noise(target_vel_norm, noise, timesteps)

        memory_tokens, memory_mask, lat_logits, lon_logits = self.encodeHistoryCondition(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            extras,
        )
        intent_cond = self.buildIntentCondition(
            lat_logits,
            lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
            epoch=epoch,
            training=True,
        )

        pred_vel_cond = torch.zeros_like(x_t)
        if self.self_condition_prob > 0.0:
            use_sc = (torch.rand(batch_size, 1, 1, device=device) < self.self_condition_prob).float()
            if use_sc.any():
                with torch.no_grad():
                    prev_pred_vel = self.predictX0(
                        x_t,
                        timesteps,
                        memory_tokens,
                        intent_cond,
                        memory_mask,
                        pred_vel_cond,
                    )
                pred_vel_cond = prev_pred_vel.detach() * use_sc

        pred_vel_norm_t = self.predictX0(x_t, timesteps, memory_tokens, intent_cond, memory_mask, pred_vel_cond)
        loss, loss_metrics = self.computeLoss(
            pred_vel_norm_t,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
            return_components=True,
        )

        if self.fut_enable_train_vis:
            std_vel = self.fut_delta_std.view(1, 1, 2).to(device)
            mean_vel = self.fut_delta_mean.view(1, 1, 2).to(device)
            pred_pos_phys = torch.cumsum(pred_vel_norm_t[..., :2] * std_vel + mean_vel, dim=1) + anchor_phys[..., :2]
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
            return loss, loss_metrics
        return loss

    @torch.no_grad()
    def forwardEvalMultiSample(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, K=5):
        """执行多样本采样评估，并返回 oracle 最优样本结果。

        Args:
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            mask: 邻车存在性掩码。
            temporal_mask: 邻车可视化掩码。
            future: 真实未来轨迹。
            op_mask: 未来有效位掩码。
            extras: 车道、距离与意图标签字典。
            device: 目标设备。
            K: 采样次数。

        Returns:
            一个四元组：
            - loss: 以 oracle 最优样本为基准计算的总损失。
            - best_pred_phys: oracle 最优样本对应的未来预测轨迹。
            - ade_batch: oracle 最优样本的 batch ADE。
            - fde_batch: oracle 最优样本的 batch FDE。
        """
        extras, device = self.resolveForwardInputs(hist, hist_nbrs, extras, device)
        batch_size, t_len, _ = future.shape
        valid_mask = (op_mask[..., 0] > 0).float()
        anchor_phys = hist[..., -1:, :self.output_dim]
        future_phys = future[..., :self.output_dim]

        memory_tokens, memory_mask, lat_logits, lon_logits = self.encodeHistoryCondition(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            extras,
        )
        intent_cond = self.buildIntentCondition(lat_logits, lon_logits, training=False)
        infer_scheduler = DDIMScheduler.from_config(
            self.diffusion_scheduler.config,
            timestep_spacing=self.inference_timestep_spacing,
        )
        infer_scheduler.set_timesteps(self.num_inference_steps)
        k_eff = max(1, int(K))
        pred_vel_norm_list = []
        for _ in range(k_eff):
            x_t = torch.randn((batch_size, t_len, self.output_dim), device=device)
            pred_vel_norm_list.append(self.rolloutFromXt(x_t, memory_tokens, intent_cond, memory_mask, infer_scheduler))
        pred_vel_norm = torch.stack(pred_vel_norm_list, dim=1)
        std_vel = self.fut_delta_std.view(1, 1, 1, 2).to(device)
        mean_vel = self.fut_delta_mean.view(1, 1, 1, 2).to(device)
        pred_vel_phys = pred_vel_norm[..., :2] * std_vel + mean_vel
        pred_pos_phys = torch.cumsum(pred_vel_phys, dim=2) + anchor_phys[..., :2].unsqueeze(1)

        all_preds = future_phys.unsqueeze(1).repeat(1, k_eff, 1, 1).clone()
        all_preds[..., :2] = pred_pos_phys
        target_phys = future_phys[..., :2].unsqueeze(1)
        diff = torch.norm(all_preds[..., :2] - target_phys, dim=-1)
        valid_mask_exp = valid_mask.unsqueeze(1)
        ade_k = (diff * valid_mask_exp).sum(dim=2) / (valid_mask_exp.sum(dim=2) + 1e-6)
        _, best_k_idx = torch.min(ade_k, dim=1)

        best_pred_idx = best_k_idx.view(batch_size, 1, 1, 1).expand(batch_size, 1, t_len, self.output_dim)
        best_pred_phys = all_preds.gather(1, best_pred_idx).squeeze(1)
        best_pred_vel_norm = pred_vel_norm.gather(1, best_pred_idx).squeeze(1)
        target_vel_norm, _ = self.buildTargetVelNorm(future_phys, anchor_phys, device)
        loss = self.computeLoss(
            best_pred_vel_norm,
            target_vel_norm,
            future_phys,
            anchor_phys,
            valid_mask,
            lat_logits=lat_logits,
            lon_logits=lon_logits,
            lat_gt=extras.get("lat_gt"),
            lon_gt=extras.get("lon_gt"),
        )

        self.last_multisample_all_preds = all_preds.detach()
        self.last_multisample_best_idx = best_k_idx.detach()
        self.last_multisample_intent_pairs = None
        self.last_multisample_intent_prob = None

        ade_batch, fde_batch = self.computeAdeFde(best_pred_phys, future, valid_mask)
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
        return loss, best_pred_phys, ade_batch, fde_batch

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, op_mask, extras=None, device=None, epoch=1, return_components=False):
        """对齐 `nn.Module` 默认接口，训练时转发到 `forwardTrain`。

        Args:
            hist: ego 历史状态。
            hist_nbrs: 邻车历史状态。
            mask: 邻车存在性掩码。
            temporal_mask: 邻车可视化掩码。
            future: 真实未来轨迹。
            op_mask: 未来有效位掩码。
            extras: 车道、距离与意图标签字典。
            device: 目标设备。
            epoch: 兼容旧训练入口的 epoch 参数。
            return_components: 是否返回日志分量。

        Returns:
            与 `forwardTrain` 保持一致的返回值。
        """
        return self.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            future,
            op_mask,
            extras,
            device,
            epoch=epoch,
            return_components=return_components,
        )
