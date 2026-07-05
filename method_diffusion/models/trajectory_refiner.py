import math

import torch
from torch import nn

from method_diffusion.utils.trajectory_kinematics import recompute_theta_v_from_xy


HIST_SEQUENCE_FEATURE_DIM = 10
TRAJ_SEQUENCE_FEATURE_DIM = 11
MODE_CONTEXT_FEATURE_DIM = 12


class TemporalBasisResidualRefiner(nn.Module):
    """TABR: 编码 history / candidate 序列，预测 temporal-basis 轨迹残差。

    支持两种模式：
    - state_dim=2: 仅预测 XY 残差（原有模式）
    - state_dim=4: 预测全状态残差 (x, y, theta, v)
    """

    def __init__(self, hidden_dim=128, max_delta=5.0, time_power=2.0, num_basis=4,
                 state_dim=4, max_delta_theta=0.35, max_delta_v=3.0):
        super().__init__()
        hidden = int(hidden_dim)
        self.max_delta = float(max_delta)
        self.time_power = float(time_power)
        self.num_basis = max(int(num_basis), 2)
        self.state_dim = int(state_dim)
        self.max_delta_theta = float(max_delta_theta)
        self.max_delta_v = float(max_delta_v)

        self.hist_encoder = nn.GRU(
            input_size=HIST_SEQUENCE_FEATURE_DIM,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.traj_encoder = nn.GRU(
            input_size=TRAJ_SEQUENCE_FEATURE_DIM,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.mode_context = nn.Sequential(
            nn.LayerNorm(MODE_CONTEXT_FEATURE_DIM),
            nn.Linear(MODE_CONTEXT_FEATURE_DIM, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        if self.state_dim == 2:
            head_out_dim = self.num_basis * 2 + 1  # XY control + shared gate
        elif self.state_dim == 4:
            head_out_dim = self.num_basis * 4 + 4  # Full-state control + per-dim gate
        else:
            raise ValueError(f"Unsupported state_dim: {self.state_dim}. Must be 2 or 4.")

        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 3),
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, head_out_dim),
        )

    @staticmethod
    def prepend_zero_step(xy):
        step = xy.new_zeros(*xy.shape[:-2], xy.size(-2), 2)
        if xy.size(-2) > 1:
            step[..., 1:, :] = xy[..., 1:, :] - xy[..., :-1, :]
        return step

    @staticmethod
    def sequence_acc(step):
        acc = step.new_zeros(step.shape)
        if step.size(-2) > 1:
            acc[..., 1:, :] = step[..., 1:, :] - step[..., :-1, :]
        return acc

    @staticmethod
    def sequence_motion_state(step, dt):
        step_dt = max(float(dt), 1e-6)
        theta = torch.atan2(step[..., 1], step[..., 0]).unsqueeze(-1)
        speed = torch.linalg.norm(step, dim=-1, keepdim=True) / step_dt
        return torch.cat([theta, speed], dim=-1)

    @staticmethod
    def sequence_features(traj, include_time=False, dt=1.0):
        xy = traj[..., :2]
        step = TemporalBasisResidualRefiner.prepend_zero_step(xy)
        acc = TemporalBasisResidualRefiner.sequence_acc(step)
        step_norm = torch.linalg.norm(step, dim=-1, keepdim=True)
        acc_norm = torch.linalg.norm(acc, dim=-1, keepdim=True)
        theta_v = TemporalBasisResidualRefiner.sequence_motion_state(step, dt)
        features = [xy, step, step_norm, acc, acc_norm, theta_v]
        if not include_time:
            return torch.cat(features, dim=-1)

        time = torch.linspace(
            1.0 / max(traj.size(-2), 1),
            1.0,
            traj.size(-2),
            device=traj.device,
            dtype=traj.dtype,
        )
        time = time.view(*((1,) * (traj.dim() - 2)), traj.size(-2), 1)
        time = time.expand(*traj.shape[:-1], 1)
        return torch.cat([*features, time], dim=-1)

    @staticmethod
    def mode_context_features(traj):
        xy = traj[..., :2]
        bsz, k_size, _, _ = xy.shape
        endpoint = xy[:, :, -1]
        endpoint_mean = endpoint.mean(dim=1, keepdim=True)
        endpoint_std = endpoint.std(dim=1, keepdim=True, unbiased=False)
        endpoint_centered = endpoint - endpoint_mean
        mean_xy = xy.mean(dim=2)
        step = TemporalBasisResidualRefiner.prepend_zero_step(xy)
        final_step = step[:, :, -1]
        endpoint_norm = torch.linalg.norm(endpoint, dim=-1, keepdim=True)
        if k_size > 1:
            mode_id = torch.linspace(0.0, 1.0, k_size, device=traj.device, dtype=traj.dtype)
        else:
            mode_id = traj.new_zeros(1)
        mode_id = mode_id.view(1, k_size, 1).expand(bsz, -1, -1)
        return torch.cat(
            [
                endpoint,
                endpoint_centered,
                endpoint_std.expand(-1, k_size, -1),
                mean_xy,
                final_step,
                endpoint_norm,
                mode_id,
            ],
            dim=-1,
        )

    def temporal_basis(self, t_len, device, dtype):
        degree = self.num_basis - 1
        time = torch.linspace(
            1.0 / max(t_len, 1),
            1.0,
            t_len,
            device=device,
            dtype=dtype,
        ).pow(max(self.time_power, 1e-6))
        terms = []
        one_minus = 1.0 - time
        for idx in range(self.num_basis):
            coeff = float(math.comb(degree, idx))
            terms.append(coeff * time.pow(idx) * one_minus.pow(degree - idx))
        return torch.stack(terms, dim=-1)

    def forward(self, hist, traj, dt):
        squeeze_candidate = False
        if traj.dim() == 3:
            traj = traj.unsqueeze(1)
            squeeze_candidate = True
        if traj.dim() != 4:
            raise ValueError(f"Expected traj shape [B,T,D] or [B,K,T,D], got {tuple(traj.shape)}")
        if traj.size(-1) < 4:
            raise ValueError(f"Expected trajectory feature dim >= 4, got {traj.size(-1)}")

        bsz, k_size, t_len, _ = traj.shape
        hist_feat = self.sequence_features(hist, include_time=False, dt=dt)
        _, hist_hidden = self.hist_encoder(hist_feat)
        hist_embed = hist_hidden[-1].unsqueeze(1).expand(-1, k_size, -1)

        traj_feat = self.sequence_features(traj, include_time=True, dt=dt)
        traj_feat_flat = traj_feat.reshape(bsz * k_size, t_len, TRAJ_SEQUENCE_FEATURE_DIM)
        _, traj_hidden = self.traj_encoder(traj_feat_flat)
        traj_embed = traj_hidden[-1].view(bsz, k_size, -1)

        mode_embed = self.mode_context(self.mode_context_features(traj))
        features = torch.cat([hist_embed, traj_embed, mode_embed], dim=-1)

        if self.state_dim == 2:
            # XY-only mode (legacy)
            raw = self.head(features.reshape(bsz * k_size, -1)).view(bsz, k_size, self.num_basis * 2 + 1)
            control = torch.tanh(raw[..., :self.num_basis * 2]).view(bsz, k_size, self.num_basis, 2)
            control = control * self.max_delta
            gate = torch.sigmoid(raw[..., -1:])
            basis = self.temporal_basis(t_len, traj.device, traj.dtype)
            delta_xy = torch.einsum("tc,bkcd->bktd", basis, control)
            delta_xy = gate.unsqueeze(2) * delta_xy
            refined_xy = traj[..., :2] + delta_xy
            refined = torch.cat([refined_xy, traj[..., 2:]], dim=-1)
            refined = recompute_theta_v_from_xy(refined, dt)
            delta_end = delta_xy[:, :, -1]
        else:
            # Full-state mode: refine x, y, theta, v with per-dimension gate
            raw = self.head(features.reshape(bsz * k_size, -1)).view(
                bsz, k_size, self.num_basis * self.state_dim + self.state_dim
            )
            control_raw = torch.tanh(raw[..., :self.num_basis * self.state_dim]).view(
                bsz, k_size, self.num_basis, self.state_dim
            )
            # Scale each dimension separately (avoid inplace)
            scale = torch.ones_like(control_raw)
            scale[..., :2] = self.max_delta
            scale[..., 2] = self.max_delta_theta
            scale[..., 3] = self.max_delta_v
            control = control_raw * scale

            gate = torch.sigmoid(raw[..., self.num_basis * self.state_dim:])  # [B, K, state_dim]
            basis = self.temporal_basis(t_len, traj.device, traj.dtype)
            delta = torch.einsum("tc,bkcd->bktd", basis, control)  # [B, K, T, 4]
            delta = gate.unsqueeze(2) * delta  # [B, K, T, 4] * [B, K, 1, 4] -> [B, K, T, 4]

            refined = traj + delta
            # Normalize theta to [-pi, pi] (avoid inplace)
            theta_normalized = torch.atan2(torch.sin(refined[..., 2]), torch.cos(refined[..., 2]))
            # Ensure v is non-negative (avoid inplace)
            v_clamped = refined[..., 3].clamp(min=0.0)
            # Create new tensor instead of inplace modification
            refined = torch.stack([refined[..., 0], refined[..., 1], theta_normalized, v_clamped], dim=-1)
            delta_end = delta[:, :, -1]

        if squeeze_candidate:
            refined = refined.squeeze(1)
            delta_end = delta_end.squeeze(1)
            gate = gate.squeeze(1)
            control = control.squeeze(1)
        control_norm = torch.linalg.norm(control, dim=-1).mean(dim=-1)
        return refined, {
            "delta_end": delta_end,
            "gate": gate,
            "control_points": control,
            "delta_regularizer": control_norm,
        }


def build_trajectory_refiner(args):
    return TemporalBasisResidualRefiner(
        hidden_dim=args.fut_refiner_hidden_dim,
        max_delta=args.fut_refiner_max_delta,
        time_power=args.fut_refiner_time_power,
        num_basis=args.fut_refiner_num_basis,
        state_dim=getattr(args, "fut_refiner_state_dim", 4),
        max_delta_theta=getattr(args, "fut_refiner_max_delta_theta", 0.35),
        max_delta_v=getattr(args, "fut_refiner_max_delta_v", 3.0),
    )
