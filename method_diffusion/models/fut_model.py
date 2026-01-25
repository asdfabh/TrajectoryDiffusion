from method_diffusion.models import dit_fut as dit
from torch import nn
from diffusers.schedulers import DDIMScheduler
import numpy as np
import torch
from method_diffusion.models.hist_encoder import HistEncoder
import torch.nn.functional as F
from pathlib import Path
from method_diffusion.utils.visualization import visualize_batch_trajectories, plot_traj_with_mask, plot_traj
import math

def gen_sineembed_for_position(pos_tensor, hidden_dim=128):
    # Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[..., 0] * scale
    y_embed = pos_tensor[..., 1] * scale
    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat((pos_y, pos_x), dim=-1)
    return pos

class DiffusionFut(nn.Module):

    def __init__(self, args):
        super(DiffusionFut, self).__init__()
        # Net parameters
        self.args = args
        self.feature_dim = int(args.feature_dim_fut) # 输入特征维度 default: 6 (x, y, v, a, laneID, class)
        self.input_dim = int(args.input_dim_fut)   # 输入到Dit的维度 default: 128
        self.hidden_dim = int(args.hidden_dim_fut)
        self.output_dim = int(args.output_dim_fut)
        self.heads = int(args.heads_fut)
        self.dropout = args.dropout_fut
        self.depth = int(args.depth_fut)
        self.mlp_ratio = args.mlp_ratio_fut
        self.num_train_timesteps = args.num_train_timesteps_fut
        self.time_embedding_size = args.time_embedding_size_fut
        self.num_inference_steps = args.num_inference_steps
        self.num_modes = args.num_modes  # 聚类数
        self.T = int(args.T_f)

        # 输入嵌入层和位置编码，相加得到Dit的输入
        self.query_encoder = nn.Sequential(nn.Linear(self.input_dim, self.input_dim), nn.SiLU(),
            nn.Linear(self.input_dim, self.input_dim), nn.LayerNorm(self.input_dim))
        self.timestep_embedder = dit.TimestepEmbedder(self.input_dim, self.time_embedding_size)

        self.hist_encoder = HistEncoder(args)
        self.enc_embedding = nn.Linear(self.args.encoder_input_dim, self.input_dim)
        nn.init.xavier_uniform_(self.enc_embedding.weight)
        nn.init.constant_(self.enc_embedding.bias, 0)

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
            depth=self.depth,
            model_type="x_start"
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.SiLU(),
            nn.Linear(self.input_dim, self.num_modes)
        )

        self.anchor_path = './method_diffusion/dataset/anchors_ngsim.npy'  # 确保路径正确
        if Path(self.anchor_path).exists():
            anchors = np.load(self.anchor_path)  # [K, T, 2]
            self.register_buffer('anchors', torch.from_numpy(anchors).float())
        else:
            print(f"Warning: Anchor file not found at {self.anchor_path}, using zeros.")
            self.register_buffer('anchors', torch.zeros(self.num_modes, self.T, 2))

        self.register_buffer('pos_mean', torch.tensor([0.0, 0.0]).float(), persistent=False)
        self.register_buffer('pos_std', torch.tensor([30, 200]).float(), persistent=False)
        self.register_buffer('va_mean', torch.tensor([20, 0.01]).float(), persistent=False)
        self.register_buffer('va_std', torch.tensor([20, 8]).float(), persistent=False)


    def get_closest_anchor(self, future):
        """
        计算 GT 与所有 Anchor 的距离，返回最近 Anchor 的索引
        future: [B, T, 2]
        anchors: [K, T, 2]
        return: [B] (indices)
        """
        # [B, 1, T, 2] - [1, K, T, 2] -> [B, K, T, 2]
        # 使用 denorm 后的距离或者 norm 后的距离都可以，这里用 norm 后的
        future_norm = self.norm(future)[..., :2]
        anchors_norm = self.norm(self.anchors)[..., :2]

        diff = future_norm.unsqueeze(1) - anchors_norm.unsqueeze(0)
        dist = torch.norm(diff, dim=-1).mean(dim=-1)  # [B, K]
        min_dist, min_idx = torch.min(dist, dim=1)  # [B]
        return min_idx

    def compute_loss(self, pred_x0, target_norm, cls_logits, target_mode_idx):
        """
        pred_x0: [B, T, 2] 预测的轨迹
        target_norm: [B, T, 2] GT
        cls_logits: [B, K] 分类 logits
        target_mode_idx: [B] GT 对应的最近 anchor 索引
        """
        # Regression Loss (只计算 Best Anchor 对应的 loss，这里 pred_x0 已经是对应 Best Anchor 的预测了)
        reg_loss = F.l1_loss(pred_x0, target_norm)  # 或者 MSE
        # Classification Loss
        cls_loss = F.cross_entropy(cls_logits, target_mode_idx)

        total_loss = reg_loss + 0.5 * cls_loss
        return total_loss, reg_loss, cls_loss

    def forward_train(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]  # [B, D]
        enc_emb = self.enc_embedding(global_context)  # [B, D]

        target_mode_idx = self.get_closest_anchor(future)  # [B]
        cls_logits = self.cls_head(enc_emb)  # [B, K]

        # 截断扩散 (Truncated Diffusion): 只在 0 - 50 步之间训练
        max_train_step = 50
        timesteps = torch.randint(0, max_train_step, (B,), device=device).long()
        target_norm = self.norm(future)[..., :2]

        # 加噪: Standard Diffusion 是在 GT 上加噪
        noise = torch.randn_like(target_norm)
        x_noisy = self.diffusion_scheduler.add_noise(target_norm, noise, timesteps)

        x_sine = gen_sineembed_for_position(x_noisy, hidden_dim=self.input_dim)
        x_input = self.query_encoder(x_sine)  # [B, T, input_dim]
        t_emb = self.timestep_embedder(timesteps)
        y = t_emb + enc_emb

        pred_x0 = self.dit(x=x_input, y=y, cross=context)
        loss, reg_loss, cls_loss = self.compute_loss(pred_x0, target_norm, cls_logits, target_mode_idx)

        # For logging
        pred = self.denorm(pred_x0)
        diff = pred[..., :2] - future[..., :2]
        ade = torch.norm(diff, dim=-1).mean()
        fde = torch.norm(diff, dim=-1)[:, -1].mean()

        return loss, pred, ade, fde

    @torch.no_grad()
    def forward_eval(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        B, T, _ = future.shape
        K = self.num_modes

        hist_norm = self.norm(hist)
        hist_nbrs_norm = self.norm(hist_nbrs)
        context, hist_enc = self.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
        global_context = hist_enc[:, -1, :]
        enc_emb = self.enc_embedding(global_context)  # [B, D]

        cls_logits = self.cls_head(enc_emb)
        cls_probs = F.softmax(cls_logits, dim=-1)  # [B, K]

        # Context Expand: [B, T, D] -> [B*K, T, D], 并行生成 K 条轨迹
        context_expanded = context.repeat_interleave(K, dim=0)
        enc_emb_expanded = enc_emb.repeat_interleave(K, dim=0)

        # Anchors: [K, T, 2] -> [B, K, T, 2] -> [B*K, T, 2]
        batch_anchors = self.anchors.unsqueeze(0).repeat(B, 1, 1, 1)
        batch_anchors_norm = self.norm(batch_anchors)[..., :2]
        x_t = batch_anchors_norm.view(B * K, T, 2)

        start_step = 20  # 例如从第 20 步开始 (总共1000)
        noise = torch.randn_like(x_t)

        # 给 Anchor 加噪，作为起始点
        x_t = self.diffusion_scheduler.add_noise(
            x_t, noise,
            torch.full((B * K,), start_step, device=device, dtype=torch.long)
        )

        inference_timesteps = [20, 10, 0]

        for t_val in inference_timesteps:
            timesteps = torch.full((B * K,), t_val, device=device, dtype=torch.long)
            t_emb = self.timestep_embedder(timesteps)
            y = t_emb + enc_emb_expanded

            x_sine = gen_sineembed_for_position(x_t, hidden_dim=self.input_dim)
            x_input = self.query_encoder(x_sine)
            pred_x0 = self.dit(x=x_input, y=y, cross=context_expanded)

            if t_val > 0:
                output = self.diffusion_scheduler.step(pred_x0, t_val, x_t)
                x_t = output.prev_sample
            else:
                x_t = pred_x0  # 最后一步直接取预测结果

        pred_trajs = self.denorm(x_t).view(B, K, T, 2)  # [B, K, T, 2]

        gt = future[..., :2].unsqueeze(1)  # [B, 1, T, 2]
        diff = pred_trajs - gt
        dist = torch.norm(diff, dim=-1)  # [B, K, T]

        ade_per_mode = dist.mean(dim=-1)  # [B, K]
        fde_per_mode = dist[..., -1]  # [B, K]

        min_ade = ade_per_mode.min(dim=1)[0].mean()
        min_fde = fde_per_mode.min(dim=1)[0].mean()

        # 也可以返回基于分类置信度最高的轨迹作为单模态输出
        # best_mode_idx = cls_probs.argmax(dim=-1)  # [B]

        loss = torch.tensor(0.0)  # Eval 阶段 loss 仅供参考

        return loss, pred_trajs, min_ade, min_fde

    def forward(self, hist, hist_nbrs, mask, temporal_mask, future, device):
        """Standard forward method for DDP compatibility"""
        return self.forward_train(hist, hist_nbrs, mask, temporal_mask, future, device)

    # hist = [B, T, dim], nbrs = [N_total, T, dim]. dim = x, y, v, a, laneID, class
    def norm(self, x):
        x_norm = x.clone()
        x_norm[..., 0:2] = (x[..., 0:2] - self.pos_mean) / self.pos_std  # x, y
        C = x_norm.shape[-1]
        if C == 3:
            x_norm[..., 2:4] = (x[..., 2:4] - self.va_mean) / self.va_std  # v, a
        x_norm = torch.clamp(x_norm, -5.0, 5.0)
        return x_norm

    def denorm(self, x):
        x_denorm = x.clone()
        x_denorm[..., 0:2] = x[..., 0:2] * self.pos_std + self.pos_mean  # x, y
        C = x.shape[-1]
        if C == 3:
            x_denorm[..., 2:4] = (x[..., 2:4] * self.va_std) + self.va_mean  # v, a
        return x_denorm

