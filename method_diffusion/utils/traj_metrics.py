from typing import Optional

import torch


def _build_valid_mask(
    op_mask: Optional[torch.Tensor],
    batch_size: int,
    t_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if op_mask is None:
        return torch.ones(batch_size, t_len, device=device, dtype=dtype)

    if op_mask.dim() == 3:
        valid_mask = op_mask[:, :t_len, 0]
    elif op_mask.dim() == 2:
        valid_mask = op_mask[:, :t_len]
    else:
        raise ValueError(f"Unsupported op_mask dim: {op_mask.dim()}")

    return valid_mask.to(device=device, dtype=dtype).clamp(0, 1)


def _compute_step_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    op_mask: Optional[torch.Tensor] = None,
    t_max: Optional[int] = None,
):
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    t_len = min(pred_xy.shape[1], target_xy.shape[1])
    if t_max is not None:
        t_len = min(t_len, int(t_max))
    if t_len <= 0:
        raise ValueError("Trajectory length must be positive after alignment.")

    pred_xy = pred_xy[:, :t_len]
    target_xy = target_xy[:, :t_len]
    batch_size = pred_xy.shape[0]

    valid_mask = _build_valid_mask(
        op_mask=op_mask,
        batch_size=batch_size,
        t_len=t_len,
        device=pred_xy.device,
        dtype=pred_xy.dtype,
    )

    diff = pred_xy - target_xy
    dist_sq = torch.sum(diff * diff, dim=-1)
    dist = torch.sqrt(dist_sq.clamp(min=1e-12))

    se_sum_t = torch.sum(dist_sq * valid_mask, dim=0)
    de_sum_t = torch.sum(dist * valid_mask, dim=0)
    count_t = torch.sum(valid_mask, dim=0)
    return se_sum_t, de_sum_t, count_t


def _compute_ade_fde_from_step_sums(de_sum_t: torch.Tensor, count_t: torch.Tensor):
    total_de = de_sum_t.sum()
    total_count = count_t.sum().clamp(min=1.0)
    ade = total_de / total_count

    valid_steps = torch.nonzero(count_t > 0, as_tuple=True)[0]
    if valid_steps.numel() > 0:
        last_idx = valid_steps[-1]
        fde = de_sum_t[last_idx] / count_t[last_idx].clamp(min=1.0)
    else:
        last_idx = torch.tensor(0, device=de_sum_t.device, dtype=torch.long)
        fde = torch.zeros((), device=de_sum_t.device, dtype=de_sum_t.dtype)
    return ade, fde, last_idx


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    op_mask: Optional[torch.Tensor] = None,
    t_max: Optional[int] = None,
    unit_conversion: float = 1.0,
):
    """
    Compute ADE/FDE/RMSE in the same aggregation style as TAME evaluate:
    - ADE: sum of DE over all valid points / number of valid points.
    - FDE: DE at the last valid timestep aggregated over valid samples at that timestep.
    """
    se_sum_t, de_sum_t, count_t = _compute_step_sums(
        pred=pred,
        target=target,
        op_mask=op_mask,
        t_max=t_max,
    )
    ade, fde, last_idx = _compute_ade_fde_from_step_sums(de_sum_t, count_t)

    counts_clamped = count_t.clamp(min=1.0)
    rmse_per_step = torch.sqrt(se_sum_t / counts_clamped)
    de_per_step = de_sum_t / counts_clamped

    scale = float(unit_conversion)
    scale_sq = scale * scale
    return {
        "se_sum_t": se_sum_t,
        "de_sum_t": de_sum_t,
        "count_t": count_t,
        "rmse_per_step": rmse_per_step * scale,
        "de_per_step": de_per_step * scale,
        "ade": ade * scale,
        "fde": fde * scale,
        "mse": (se_sum_t.sum() / count_t.sum().clamp(min=1.0)) * scale_sq,
        "last_valid_t": int(last_idx.item()),
    }


class TrajectoryMetricsAccumulator:
    def __init__(self, t_max: int, device: torch.device, unit_conversion: float = 0.3048):
        self.t_max = int(t_max)
        self.device = device
        self.unit_conversion = float(unit_conversion)

        self.total_se = torch.zeros(self.t_max, device=device)
        self.total_de = torch.zeros(self.t_max, device=device)
        self.total_counts = torch.zeros(self.t_max, device=device)

    def update(self, pred: torch.Tensor, target: torch.Tensor, op_mask: Optional[torch.Tensor] = None):
        batch = compute_batch_metrics(
            pred=pred,
            target=target,
            op_mask=op_mask,
            t_max=self.t_max,
            unit_conversion=1.0,
        )
        self.total_se += batch["se_sum_t"]
        self.total_de += batch["de_sum_t"]
        self.total_counts += batch["count_t"]

    def get_summary(self):
        counts = self.total_counts.clamp(min=1.0)
        rmse_per_step = torch.sqrt(self.total_se / counts) * self.unit_conversion
        de_per_step = (self.total_de / counts) * self.unit_conversion

        ade, fde, _ = _compute_ade_fde_from_step_sums(self.total_de, self.total_counts)
        overall_mse = (self.total_se.sum() / self.total_counts.sum().clamp(min=1.0)) * (
            self.unit_conversion ** 2
        )
        return {
            "rmse_per_step": rmse_per_step,
            "de_per_step": de_per_step,
            "overall_ade": float((ade * self.unit_conversion).item()),
            "overall_fde": float((fde * self.unit_conversion).item()),
            "overall_mse": float(overall_mse.item()),
        }

    def running_ade(self) -> float:
        ade, _, _ = _compute_ade_fde_from_step_sums(self.total_de, self.total_counts)
        return float((ade * self.unit_conversion).item())

    def running_fde(self) -> float:
        _, fde, _ = _compute_ade_fde_from_step_sums(self.total_de, self.total_counts)
        return float((fde * self.unit_conversion).item())
