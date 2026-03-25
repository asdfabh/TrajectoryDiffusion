from method_diffusion.models import dit_hist as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import torch
import torch.nn.functional as F
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import maybe_visualize_hist_reconstruction


_HIST_NORMALIZATION_PRESETS = {
    "ngsim": {
        "pos_mean": [0.049275586306451916, -37.23250272465904],
        "pos_std": [0.8998403636714408, 33.7460443040218],
        "va_mean": [24.77257929333611, 0.08585249191100579],
        "va_std": [14.214159448243082, 4.626550411651881],
    },
    # highD 先保留占位值，收到统计参数后直接替换这一组常量即可。
    "highd": {
        "pos_mean": [-0.03517636323581014, -127.6782104010702],
        "pos_std": [0.8697201631848155, 89.44988951201532],
        "va_mean": [85.15262720834814, -0.060102165990297766],
        "va_std": [24.390956800543215, 1.0738574975912054],
    },
}


class DiffusionPast(nn.Module):

    def __init__(self, args):
        super(DiffusionPast, self).__init__()
        # Net parameters
        self.args = args
        self.dataset_name = str(getattr(args, "dataset", "ngsim")).strip().lower()
        self.feature_dim = 2 * int(args.feature_dim) + 1  # 输入特征维度 default: 6 (x, y, v, a, laneID, class)
        self.input_dim = int(args.input_dim)   # 输入到Dit的维度 default: 128
        self.hidden_dim = int(args.hidden_dim)
        self.output_dim = int(args.output_dim)
        self.heads = int(args.heads)
        self.dropout = args.dropout
        self.depth = int(args.depth)
        self.mlp_ratio = args.mlp_ratio
        self.num_train_timesteps = args.num_train_timesteps
        self.time_embedding_size = args.time_embedding_size
        self.num_inference_steps = args.num_inference_steps
        self.T = int(args.T)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.input_embedding = nn.Linear(self.feature_dim, self.input_dim)
        self.pos_embedding = SequentialPositionalEncoding(self.input_dim)

        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.final_layer = dit.FinalLayer(self.hidden_dim, self.T, self.output_dim)
        self.dit = dit.DiT(
            dit_block=dit_block,
            final_layer=self.final_layer,
            time_embedder=self.timestep_embedder,
            depth=self.depth,
            model_type="x_start"
        )

        norm_params = self.load_normalization_params()
        self.register_buffer("pos_mean", norm_params["pos_mean"], persistent=False)
        self.register_buffer("pos_std", norm_params["pos_std"], persistent=False)
        self.register_buffer("va_mean", norm_params["va_mean"], persistent=False)
        self.register_buffer("va_std", norm_params["va_std"], persistent=False)
        self.hist_dt = 0.2  # NGSIM 10Hz and current hist pipeline uses d_s=2.
        self.loss_weights = {
            "xy_unknown": 1.00,
            "xy_known": 0.20,
            "va_unknown": 0.45,
            "va_known": 0.10,
            "dxy": 0.00,
            "v_cons": 0.00,
            "a_cons": 0.00,
        }

    def load_normalization_params(self):
        params = _HIST_NORMALIZATION_PRESETS.get(self.dataset_name)
        if params is None:
            supported = ", ".join(sorted(_HIST_NORMALIZATION_PRESETS.keys()))
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}' for hist normalization. Supported: {supported}"
            )
        return {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in params.items()
        }

    def masked_l1(self, pred, target, mask=None):
        loss = F.l1_loss(pred, target, reduction="none")
        if mask is None:
            return loss.mean()
        mask = mask.to(pred.device).float()
        while mask.dim() < loss.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(loss)
        return (loss * mask).sum() / (mask.sum() + 1e-6)

    def compute_motion_loss(self, pred, target, mask, return_parts=False):
        """
        pred: [B, T, D]
        target: [B, T, D]
        mask: [B, T, 1] (1 for known/observed, 0 for unknown/masked)
        """
        known_mask = mask.view(mask.shape[0], mask.shape[1], 1).to(pred.device).float()
        unknown_mask = 1.0 - known_mask

        loss_xy_unknown = self.masked_l1(pred[..., :2], target[..., :2], unknown_mask)
        loss_xy_known = self.masked_l1(pred[..., :2], target[..., :2], known_mask)

        zero = pred.new_tensor(0.0)
        loss_va_unknown = zero
        loss_va_known = zero
        loss_dxy = zero
        loss_v_cons = zero
        loss_a_cons = zero

        if pred.shape[-1] >= 4:
            loss_va_unknown = self.masked_l1(pred[..., 2:4], target[..., 2:4], unknown_mask)
            loss_va_known = self.masked_l1(pred[..., 2:4], target[..., 2:4], known_mask)

            pred_phys = self.denorm(pred)
            target_phys = self.denorm(target)

            pred_dxy = pred_phys[:, 1:, :2] - pred_phys[:, :-1, :2]
            target_dxy = target_phys[:, 1:, :2] - target_phys[:, :-1, :2]
            pair_unknown = torch.maximum(unknown_mask[:, 1:, :], unknown_mask[:, :-1, :])
            # loss_dxy: 约束相邻时刻的位置增量 delta(x, y) 与 GT 一致，衡量局部轨迹形状是否平滑且正确。
            loss_dxy = self.masked_l1(pred_dxy, target_dxy, pair_unknown)

            pred_v = pred_phys[..., 2]
            pred_a = pred_phys[..., 3]
            pred_v_from_y = pred_dxy[..., 1] / self.hist_dt
            pred_a_from_v = (pred_v[:, 1:] - pred_v[:, :-1]) / self.hist_dt
            # loss_v_cons: 用位置序列的纵向差分速度约束输出 v，衡量预测位置与预测速度是否自洽。
            loss_v_cons = self.masked_l1(pred_v_from_y, pred_v[:, 1:], None)
            # loss_a_cons: 用速度序列的一阶差分约束输出 a，衡量预测速度与预测加速度是否自洽。
            loss_a_cons = self.masked_l1(pred_a_from_v, pred_a[:, 1:], None)

        total_loss = (
            self.loss_weights["xy_unknown"] * loss_xy_unknown
            + self.loss_weights["xy_known"] * loss_xy_known
            + self.loss_weights["va_unknown"] * loss_va_unknown
            + self.loss_weights["va_known"] * loss_va_known
            + self.loss_weights["dxy"] * loss_dxy
            + self.loss_weights["v_cons"] * loss_v_cons
            + self.loss_weights["a_cons"] * loss_a_cons
        )

        if not return_parts:
            return total_loss

        parts = {
            "loss_total": total_loss.detach(),
            "loss_xy_unknown": loss_xy_unknown.detach(),
            "loss_xy_known": loss_xy_known.detach(),
            "loss_va_unknown": loss_va_unknown.detach(),
            "loss_va_known": loss_va_known.detach(),
            "loss_dxy": loss_dxy.detach(),
            "loss_v_cons": loss_v_cons.detach(),
            "loss_a_cons": loss_a_cons.detach(),
        }
        return total_loss, parts

    # hist: [B, T, dim], hist_masked: [B, T, dim+1]
    def forward_train(self, hist, hist_masked, device, return_components=False):
        B, T,  _ = hist_masked.shape

        hist_mask = hist_masked[..., -1:].float()  # [B, T, 1]
        hist_masked_value = hist_masked[..., :-1]  # [B, T, dim]
        cond = torch.cat([self.norm(hist_masked_value), hist_mask], dim=-1)  # [B, T, dim+1]

        # 训练加噪对象改为 full x0，条件仍为 masked 轨迹
        x_start = self.norm(hist)  # [B, T, dim]
        timesteps = torch.randint(0, self.num_train_timesteps, (B,), device=device)
        noise = torch.randn_like(x_start)
        x_noisy = self.diffusion_scheduler.add_noise(x_start, noise, timesteps)
        model_input = torch.cat([x_noisy, cond], dim=-1)  # [B, T, 2*dim+1]

        input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
        pred_x0 = self.dit(x=input_embedded, t=timesteps)

        loss, loss_parts = self.compute_motion_loss(pred_x0, x_start, hist_mask, return_parts=True)

        pred = self.denorm(pred_x0)
        maybe_visualize_hist_reconstruction(
            hist=hist,
            hist_masked=hist_masked,
            pred=pred,
            stage="train",
            enable_train_vis=int(getattr(self.args, "hist_enable_train_vis", 0)) > 0,
            enable_eval_vis=int(getattr(self.args, "hist_enable_eval_vis", 0)) > 0,
        )

        if return_components:
            return loss, pred, loss_parts
        return loss, pred

    @torch.no_grad()
    def forward_eval(self, hist, hist_masked, device):
        B, T, _ = hist_masked.shape

        hist_mask = hist_masked[..., -1:].float()
        mask_unknown = 1.0 - hist_mask
        hist_masked_value = hist_masked[..., :-1]
        cond = torch.cat([self.norm(hist_masked_value), hist_mask], dim=-1)
        known_x0 = self.norm(hist_masked_value)

        # Hybrid 初始化：已知位置来自 q(x_t | x0_known)，未知位置使用纯噪声
        self.diffusion_scheduler.set_timesteps(self.num_inference_steps)
        infer_timesteps = self.diffusion_scheduler.timesteps
        init_t = int(infer_timesteps[0])
        init_t_batch = torch.full((B,), init_t, device=device, dtype=torch.long)
        known_noise = torch.randn_like(known_x0)
        known_x_t = self.diffusion_scheduler.add_noise(known_x0, known_noise, init_t_batch)
        unknown_x_t = torch.randn_like(known_x0)
        x_t = hist_mask * known_x_t + mask_unknown * unknown_x_t

        for idx, t in enumerate(infer_timesteps):
            t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
            model_input = torch.cat((x_t, cond), dim=-1)
            input_embedded = self.input_embedding(model_input) + self.pos_embedding(model_input)
            pred_x0_norm = self.dit(x=input_embedded, t=t_batch)

            # 采样阶段强约束已观测轨迹，模型重点修复缺失段
            pred_x0_norm = hist_mask * known_x0 + mask_unknown * pred_x0_norm
            x_t = self.diffusion_scheduler.step(pred_x0_norm, t, x_t).prev_sample

            # 准备下一步：将已知位置重新投影到对应噪声层
            if idx < len(infer_timesteps) - 1:
                next_t = int(infer_timesteps[idx + 1])
                next_t_batch = torch.full((B,), next_t, device=device, dtype=torch.long)
                known_x_next = self.diffusion_scheduler.add_noise(known_x0, known_noise, next_t_batch)
                x_t = hist_mask * known_x_next + mask_unknown * x_t

        final_pred_norm = hist_mask * known_x0 + mask_unknown * x_t
        final_pred = self.denorm(final_pred_norm)
        loss = torch.nn.functional.mse_loss(final_pred, hist)
        maybe_visualize_hist_reconstruction(
            hist=hist,
            hist_masked=hist_masked,
            pred=final_pred,
            stage="eval",
            enable_train_vis=int(getattr(self.args, "hist_enable_train_vis", 0)) > 0,
            enable_eval_vis=int(getattr(self.args, "hist_enable_eval_vis", 0)) > 0,
        )

        return loss, final_pred

    def forward(self, hist, hist_masked, device, return_components=False):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_masked, device, return_components=return_components)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        if x_norm.size(-1) >= 2:
            x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        if x_norm.size(-1) >= 4:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        if x_denorm.size(-1) >= 2:
            x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        if x_denorm.size(-1) >= 4:
            x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean  # v, a
        return x_denorm
