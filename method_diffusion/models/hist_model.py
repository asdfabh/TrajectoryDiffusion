from method_diffusion.models import dit_hist as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import torch
import torch.nn.functional as F
from method_diffusion.utils.position_encoding import SequentialPositionalEncoding
from method_diffusion.utils.visualization import visualize_hist_reconstruction


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

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="sample",
            clip_sample=False,
        )

        dit_block = dit.DiTBlock(self.input_dim, self.heads, self.dropout, self.mlp_ratio)
        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)
        self.final_layer = dit.FinalLayer(self.input_dim, self.output_dim)
        self.dit = dit.DiT(dit_block=dit_block, final_layer=self.final_layer, depth=self.depth)

        self.init_params()
        self.loss_weights = self.init_loss_weights()

    def init_params(self):
        if self.dataset_name == "ngsim":
            pos_mean = [0.049275586306451916, -37.23250272465904]
            pos_std = [0.8998403636714408, 33.7460443040218]
            va_mean = [24.77257929333611, 0.08585249191100579]
            va_std = [14.214159448243082, 4.626550411651881]
            lane_center = [4.0]
            lane_scale = [3.0]   # laneID: [1, 7]
            class_center = [2.0]
            class_scale = [1.0]  # class: [1, 3]
            lane_min = [1.0]
            lane_max = [7.0]
            class_min = [1.0]
            class_max = [3.0]
        elif self.dataset_name == "highd":
            pos_mean = [-0.03517636323581014, -127.6782104010702]
            pos_std = [0.8697201631848155, 89.44988951201532]
            va_mean = [85.15262720834814, -0.060102165990297766]
            va_std = [24.390956800543215, 1.0738574975912054]
            lane_center = [2.5]
            lane_scale = [1.5]   # laneID: [1, 4]
            class_center = [2.0]
            class_scale = [1.0]  # class 恒为 2，保持映射到 0
            lane_min = [1.0]
            lane_max = [4.0]
            class_min = [2.0]
            class_max = [2.0]
        else:
            raise ValueError(
                f"Unsupported dataset '{self.dataset_name}' for hist normalization. Supported: highd, ngsim"
            )

        self.register_buffer("pos_mean", torch.tensor(pos_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("pos_std", torch.tensor(pos_std, dtype=torch.float32), persistent=False)
        self.register_buffer("va_mean", torch.tensor(va_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("va_std", torch.tensor(va_std, dtype=torch.float32), persistent=False)
        self.register_buffer("lane_center", torch.tensor(lane_center, dtype=torch.float32), persistent=False)
        self.register_buffer("lane_scale", torch.tensor(lane_scale, dtype=torch.float32), persistent=False)
        self.register_buffer("class_center", torch.tensor(class_center, dtype=torch.float32), persistent=False)
        self.register_buffer("class_scale", torch.tensor(class_scale, dtype=torch.float32), persistent=False)
        self.register_buffer("lane_min", torch.tensor(lane_min, dtype=torch.float32), persistent=False)
        self.register_buffer("lane_max", torch.tensor(lane_max, dtype=torch.float32), persistent=False)
        self.register_buffer("class_min", torch.tensor(class_min, dtype=torch.float32), persistent=False)
        self.register_buffer("class_max", torch.tensor(class_max, dtype=torch.float32), persistent=False)

    def init_loss_weights(self):
        return {
            "xy_unknown": 1.50,
            "xy_known": 0.40,
            "va_unknown": 1.50,
            "va_known": 0.20,
            "discrete_unknown": 1.00,
            "discrete_known": 0.20,
        }

    def masked_loss_mean(self, loss_map, mask):
        mask = mask.to(loss_map.device).float()
        while mask.dim() < loss_map.dim():
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(loss_map)
        numer = (loss_map * mask).sum(dim=tuple(range(1, loss_map.dim())))
        denom = mask.sum(dim=tuple(range(1, mask.dim()))) + 1e-6
        return (numer / denom).mean()

    def compute_motion_loss(self, pred, target, mask):
        """
        pred: [B, T, D]
        target: [B, T, D]
        mask: [B, T, 1] (1 for known/observed, 0 for unknown/masked)
        """
        known_mask = mask.view(mask.shape[0], mask.shape[1], 1).to(pred.device).float()
        unknown_mask = 1.0 - known_mask

        loss_xy_map = F.l1_loss(pred[..., :2], target[..., :2], reduction="none")
        loss_v_map = F.l1_loss(pred[..., 2:3], target[..., 2:3], reduction="none")
        loss_a_map = F.smooth_l1_loss(pred[..., 3:4], target[..., 3:4], reduction="none")

        loss_xy_unknown = self.masked_loss_mean(loss_xy_map, unknown_mask)
        loss_xy_known = self.masked_loss_mean(loss_xy_map, known_mask)
        loss_v_unknown = self.masked_loss_mean(loss_v_map, unknown_mask)
        loss_v_known = self.masked_loss_mean(loss_v_map, known_mask)
        loss_a_unknown = self.masked_loss_mean(loss_a_map, unknown_mask)
        loss_a_known = self.masked_loss_mean(loss_a_map, known_mask)
        loss_lane_unknown = pred.new_tensor(0.0)
        loss_lane_known = pred.new_tensor(0.0)
        loss_class_unknown = pred.new_tensor(0.0)
        loss_class_known = pred.new_tensor(0.0)

        if pred.size(-1) == 6:
            loss_lane_map = F.mse_loss(pred[..., 4:5], target[..., 4:5], reduction="none")
            loss_class_map = F.mse_loss(pred[..., 5:6], target[..., 5:6], reduction="none")
            loss_lane_unknown = self.masked_loss_mean(loss_lane_map, unknown_mask)
            loss_lane_known = self.masked_loss_mean(loss_lane_map, known_mask)
            loss_class_unknown = self.masked_loss_mean(loss_class_map, unknown_mask)
            loss_class_known = self.masked_loss_mean(loss_class_map, known_mask)

        total_loss = (
            self.loss_weights["xy_unknown"] * loss_xy_unknown
            + self.loss_weights["xy_known"] * loss_xy_known
            + self.loss_weights["va_unknown"] * loss_v_unknown
            + self.loss_weights["va_known"] * loss_v_known
            + self.loss_weights["va_unknown"] * loss_a_unknown
            + self.loss_weights["va_known"] * loss_a_known
            + self.loss_weights["discrete_unknown"] * loss_lane_unknown
            + self.loss_weights["discrete_known"] * loss_lane_known
            + self.loss_weights["discrete_unknown"] * loss_class_unknown
            + self.loss_weights["discrete_known"] * loss_class_known
        )

        parts = {
            "loss_total": total_loss.detach(),
            "loss_xy_unknown": loss_xy_unknown.detach(),
            "loss_xy_known": loss_xy_known.detach(),
            "loss_v_unknown": loss_v_unknown.detach(),
            "loss_v_known": loss_v_known.detach(),
            "loss_a_unknown": loss_a_unknown.detach(),
            "loss_a_known": loss_a_known.detach(),
            "loss_lane_unknown": loss_lane_unknown.detach(),
            "loss_lane_known": loss_lane_known.detach(),
            "loss_class_unknown": loss_class_unknown.detach(),
            "loss_class_known": loss_class_known.detach(),
        }
        return total_loss, parts

    # hist: [B, T, dim], hist_masked: [B, T, dim+1]
    def forward_train(self, hist, hist_masked, device):
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
        t_cond = self.timestep_embedder(timesteps)
        pred_x0 = self.dit(input_embedded, t_cond)

        loss, loss_parts = self.compute_motion_loss(pred_x0, x_start, hist_mask)

        pred = self.denorm(pred_x0)
        visualize_hist_reconstruction(
            hist=hist,
            hist_masked=hist_masked,
            pred=pred,
            stage="train",
            enable_train_vis=int(getattr(self.args, "hist_enable_train_vis", 0)) > 0,
            enable_eval_vis=int(getattr(self.args, "hist_enable_eval_vis", 0)) > 0,
        )

        return loss, pred, loss_parts

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
            t_cond = self.timestep_embedder(t_batch)
            pred_x0_norm = self.dit(input_embedded, t_cond)

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
        if final_pred.shape[-1] == 6:
            final_pred[..., 4:5] = torch.clamp(torch.round(final_pred[..., 4:5]), self.lane_min, self.lane_max)
            final_pred[..., 5:6] = torch.clamp(torch.round(final_pred[..., 5:6]), self.class_min, self.class_max)
        loss = torch.nn.functional.mse_loss(final_pred, hist)
        visualize_hist_reconstruction(
            hist=hist,
            hist_masked=hist_masked,
            pred=final_pred,
            stage="eval",
            enable_train_vis=int(getattr(self.args, "hist_enable_train_vis", 0)) > 0,
            enable_eval_vis=int(getattr(self.args, "hist_enable_eval_vis", 0)) > 0,
        )

        return loss, final_pred

    def forward(self, hist, hist_masked, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_masked, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        if x_norm.size(-1) == 4:
            x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
        elif x_norm.size(-1) == 6:
            x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std
            x_norm[..., 4:5] = (x[..., 4:5] - self.lane_center) / self.lane_scale
            x_norm[..., 5:6] = (x[..., 5:6] - self.class_center) / self.class_scale
        else:
            raise ValueError(f"Unsupported hist feature dim {x_norm.size(-1)}. Expected 4 or 6.")
        x_norm = torch.clamp(x_norm, -10.0, 10.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        if x_denorm.size(-1) == 4:
            x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
            x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean
        elif x_denorm.size(-1) == 6:
            x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean
            x_denorm[..., 2:4] = x[..., 2:4] * self.va_std + self.va_mean
            x_denorm[..., 4:5] = x[..., 4:5] * self.lane_scale + self.lane_center
            x_denorm[..., 5:6] = x[..., 5:6] * self.class_scale + self.class_center
        else:
            raise ValueError(f"Unsupported hist feature dim {x_denorm.size(-1)}. Expected 4 or 6.")
        return x_denorm
