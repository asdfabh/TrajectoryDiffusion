"""Shared future-only cubic SG derivatives and refiner kinematic helpers."""

import math

import torch


MOTION_PHYSICS_VERSION = "future_xy_sg7p3_derivative_v1"
MOTION_WINDOW = 7
MOTION_POLYORDER = 3
MOTION_SPEED_MIN = 0.1

def get_motion_method(window_length=MOTION_WINDOW):
    if window_length != MOTION_WINDOW:
        raise ValueError("Motion derivatives are fixed to SG7/P3 (window_length=7)")
    window_length = int(window_length)
    return {
        "definition": f"future_xy_sg{window_length}p3_derivative_v1",
        "window_length": window_length,
        "polyorder": MOTION_POLYORDER,
        "derivative_method": "local polynomial fit, direct analytic derivatives at window center",
        "edge_policy": "complete symmetric support only; no padding or extrapolation",
        "future_only": True,
    }


MOTION_METHOD = get_motion_method()


def _broadcast_mask(mask, shape, device):
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    value = torch.as_tensor(mask, device=device)
    if tuple(value.shape) != tuple(shape) and value.ndim >= 3 and value.shape[-1] == 1:
        value = value.squeeze(-1)
    if len(shape) == 3 and value.ndim == 2:
        value = value.unsqueeze(1)
    return torch.broadcast_to(value > 0.5, shape)


def compute_future_motion(traj, dt, valid_mask=None, speed_min=MOTION_SPEED_MIN, edge_valid=None, window_length=MOTION_WINDOW):
    """Estimate v/a/jerk with the fixed 7-point cubic fit (SG7/P3).

    Inputs: [B,T,F] or [B,K,T,F], XY in meters, dt in seconds. Each
    derivative uses the same future positions; the first/last window_length//2
    points have no output. Any invalid position or discontinuous edge
    invalidates the whole window. No history/origin is prepended.
    Longitudinal/lateral acceleration follows the fitted velocity direction;
    below speed_min that projection is excluded. Vector jerk remains valid.
    Float64 arithmetic preserves gradients for possible training losses.
    """
    if traj.ndim not in (3, 4) or traj.size(-1) < 2:
        raise ValueError("Expected trajectory [B,T,F] or [B,K,T,F], F >= 2")
    if not math.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(float(speed_min)) or float(speed_min) <= 0:
        raise ValueError("speed_min must be finite and positive")
    window_length = get_motion_method(window_length)["window_length"]

    xy = traj[..., :2].to(torch.float64)
    valid = _broadcast_mask(valid_mask, xy.shape[:-1], xy.device) & torch.isfinite(xy).all(dim=-1)
    edges = _broadcast_mask(edge_valid, (*xy.shape[:-2], max(xy.size(-2) - 1, 0)), xy.device)
    if xy.size(-2) < window_length:
        empty = xy[..., :0, 0]
        mask = valid[..., :0]
        return {
            "longitudinal_acceleration": empty, "lateral_acceleration": empty,
            "acceleration_valid": mask, "acceleration_position_valid": mask,
            "jerk": empty, "jerk_valid": mask,
        }

    xy = torch.where(valid.unsqueeze(-1), xy, torch.zeros_like(xy))
    xy = xy - xy[..., :1, :]  # Translation cancels analytically; recenter for numerical precision.
    # Exact dot-product weights for analytic derivatives of the cubic fit.
    weights = [
        [22, -67, -58, 0, 58, 67, -22],
        [5, 0, -3, -4, -3, 0, 5],
        [-1, 1, 1, 0, -1, -1, 1],
    ]
    divisors = [252, 42, 6]
    kernels = xy.new_tensor(weights) / xy.new_tensor([
        divisor * float(dt) ** order for order, divisor in enumerate(divisors, 1)
    ])[:, None]
    velocity, acceleration, jerk_vector = (xy.unfold(-2, window_length, 1) @ kernels.T).unbind(-1)
    support = valid.unfold(-1, window_length, 1).all(dim=-1)
    support = support & edges.unfold(-1, window_length - 1, 1).all(dim=-1)
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    direction = velocity / speed.clamp(min=float(speed_min)).unsqueeze(-1)
    return {
        "longitudinal_acceleration": (acceleration * direction).sum(dim=-1),
        "lateral_acceleration": acceleration[..., 1] * direction[..., 0] - acceleration[..., 0] * direction[..., 1],
        "acceleration_valid": support & (speed >= float(speed_min)),
        "acceleration_position_valid": support,
        "jerk": torch.linalg.vector_norm(jerk_vector, dim=-1),
        "jerk_valid": support,
    }
