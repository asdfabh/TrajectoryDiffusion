import torch

from method_diffusion.utils.fut_utils import normalize_traj_valid_mask, wrap_angle


def recompute_theta_v_from_xy(traj, dt):
    """按 future 构造口径，从 xy 后向差分重算 theta/v。"""
    if traj.size(-1) != 4:
        raise ValueError(f"Expected trajectory feature dim 4, got {traj.size(-1)}")

    step_dt = max(float(dt), 1e-6)
    xy = traj[..., :2]
    origin = xy.new_zeros(*xy.shape[:-2], 1, 2)
    xy_prev = torch.cat([origin, xy[..., :-1, :]], dim=-2)
    delta = xy - xy_prev
    theta = torch.atan2(delta[..., 1], delta[..., 0]).unsqueeze(-1)
    speed = (torch.linalg.norm(delta, dim=-1) / step_dt).unsqueeze(-1)
    return torch.cat([xy, theta, speed], dim=-1)


def compute_kinematic_residual(traj, dt):
    """计算轨迹的运动学残差 (xy与theta/v的一致性)。

    对于轨迹 [x, y, theta, v]，计算：
        xy_kin[t] = xy[t-1] + v[t] * [cos(theta[t]), sin(theta[t])] * dt
        residual[t] = xy[t] - xy_kin[t]

    Args:
        traj: [..., T, 4] 轨迹 (x, y, theta, v)，支持 [B,T,4] 或 [B,K,T,4]
        dt: 时间步长 (秒)

    Returns:
        residual: 与traj相同shape的运动学残差 [..., T, 2]
    """
    if traj.size(-1) != 4:
        raise ValueError(f"Expected trajectory feature dim 4, got {traj.size(-1)}")

    step_dt = max(float(dt), 1e-6)
    xy = traj[..., :2]
    theta = traj[..., 2]
    v = traj[..., 3]

    # xy_prev: [origin, xy[0], xy[1], ..., xy[T-2]]
    origin = xy.new_zeros(*xy.shape[:-2], 1, 2)
    xy_prev = torch.cat([origin, xy[..., :-1, :]], dim=-2)

    # 从theta/v积分得到的xy
    xy_kin = xy_prev + torch.stack([
        v * torch.cos(theta) * step_dt,
        v * torch.sin(theta) * step_dt,
    ], dim=-1)

    return xy - xy_kin


class PhysicalDiagnostics:
    """累计基于 xy 的物理诊断指标。"""

    def __init__(
        self,
        dt,
        acc_limit=8.0,
        jerk_limit=20.0,
        yaw_rate_limit=1.5,
        curvature_limit=0.5,
        curvature_speed_min=1.0,
    ):
        self.dt = max(float(dt), 1e-6)
        self.acc_limit = float(acc_limit)
        self.jerk_limit = float(jerk_limit)
        self.yaw_rate_limit = float(yaw_rate_limit)
        self.curvature_limit = float(curvature_limit)
        self.curvature_speed_min = float(curvature_speed_min)
        self.totals = {
            "acc_sum": 0.0,
            "acc_max": 0.0,
            "acc_violation": 0.0,
            "acc_count": 0.0,
            "jerk_sum": 0.0,
            "jerk_max": 0.0,
            "jerk_violation": 0.0,
            "jerk_count": 0.0,
            "yaw_rate_sum": 0.0,
            "yaw_rate_max": 0.0,
            "yaw_rate_violation": 0.0,
            "yaw_rate_count": 0.0,
            "curvature_sum": 0.0,
            "curvature_max": 0.0,
            "curvature_violation": 0.0,
            "curvature_count": 0.0,
            "kin_res_sum": 0.0,
            "kin_res_max": 0.0,
            "kin_res_count": 0.0,
        }

    @staticmethod
    def _update_scalar(totals, prefix, values, limit=None):
        if values.numel() == 0:
            return
        values = values.detach()
        totals[f"{prefix}_sum"] += float(values.sum().item())
        totals[f"{prefix}_max"] = max(totals[f"{prefix}_max"], float(values.max().item()))
        totals[f"{prefix}_count"] += float(values.numel())
        if limit is not None:
            totals[f"{prefix}_violation"] += float((values > float(limit)).sum().item())

    def update(self, pred, valid_mask=None):
        pred = pred.detach()
        valid = normalize_traj_valid_mask(valid_mask, pred).bool()
        xy = pred[..., :2]

        if xy.size(1) >= 2:
            delta = xy[:, 1:] - xy[:, :-1]
            vel_mask = valid[:, 1:] & valid[:, :-1]
            speed = torch.linalg.norm(delta, dim=-1) / self.dt
            if pred.size(-1) >= 4:
                theta = pred[:, :-1, 2]
                v = pred[:, :-1, 3]
                kin_step = torch.stack(
                    (v * torch.cos(theta) * self.dt, v * torch.sin(theta) * self.dt),
                    dim=-1,
                )
                kin_res = torch.linalg.norm(delta - kin_step, dim=-1)
                self._update_scalar(self.totals, "kin_res", kin_res[vel_mask])
        else:
            delta = None
            speed = None

        if xy.size(1) >= 3:
            acc_delta = xy[:, 2:] - 2.0 * xy[:, 1:-1] + xy[:, :-2]
            acc_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
            acc = torch.linalg.norm(acc_delta, dim=-1) / (self.dt ** 2)
            self._update_scalar(self.totals, "acc", acc[acc_mask], self.acc_limit)

            yaw = torch.atan2(delta[..., 1], delta[..., 0])
            yaw_delta = wrap_angle(yaw[:, 1:] - yaw[:, :-1]).abs() / self.dt
            self._update_scalar(self.totals, "yaw_rate", yaw_delta[acc_mask], self.yaw_rate_limit)

            mid_speed = speed[:, 1:]
            curvature_mask = acc_mask & (mid_speed > self.curvature_speed_min)
            curvature = yaw_delta / mid_speed.clamp(min=1e-3)
            self._update_scalar(self.totals, "curvature", curvature[curvature_mask], self.curvature_limit)

        if xy.size(1) >= 4:
            jerk_delta = xy[:, 3:] - 3.0 * xy[:, 2:-1] + 3.0 * xy[:, 1:-2] - xy[:, :-3]
            jerk_mask = valid[:, 3:] & valid[:, 2:-1] & valid[:, 1:-2] & valid[:, :-3]
            jerk = torch.linalg.norm(jerk_delta, dim=-1) / (self.dt ** 3)
            self._update_scalar(self.totals, "jerk", jerk[jerk_mask], self.jerk_limit)

    def summary(self):
        out = {}
        for prefix in ("acc", "jerk", "yaw_rate", "curvature", "kin_res"):
            count = max(self.totals[f"{prefix}_count"], 1.0)
            out[f"{prefix}_mean"] = self.totals[f"{prefix}_sum"] / count
            out[f"{prefix}_max"] = self.totals[f"{prefix}_max"]
            if f"{prefix}_violation" in self.totals:
                out[f"{prefix}_violation_rate"] = self.totals[f"{prefix}_violation"] / count
        return out


def print_kinematic_diagnostics(summary, title):
    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    print(f"{'Metric':<24} | {'Mean':<12} | {'Max':<12} | {'Violation':<12}")
    print("-" * 65)
    rows = [
        ("acc", "acceleration"),
        ("jerk", "jerk"),
        ("yaw_rate", "yaw_rate"),
        ("curvature", "curvature"),
        ("kin_res", "kinematic_residual"),
    ]
    for key, label in rows:
        violation_key = f"{key}_violation_rate"
        violation = summary.get(violation_key, None)
        violation_text = "-" if violation is None else f"{violation:<12.6f}"
        print(
            f"{label:<24} | "
            f"{summary[f'{key}_mean']:<12.6f} | "
            f"{summary[f'{key}_max']:<12.6f} | "
            f"{violation_text}"
        )
    print("=" * 65)
