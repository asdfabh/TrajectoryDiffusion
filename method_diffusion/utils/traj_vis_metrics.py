from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import torch

from method_diffusion.utils.traj_metrics import compute_batch_metrics

# Try to switch from a non-interactive backend so plt.show() can pop up windows.
# This must happen before importing pyplot.
if matplotlib.get_backend().lower() == "agg":
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        # Keep current backend if GUI backend is unavailable.
        pass

from matplotlib import pyplot as plt


def _as_cpu_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def _valid_mask_1d(op_mask_sample: Optional[torch.Tensor], t_len: int) -> torch.Tensor:
    if op_mask_sample is None:
        return torch.ones(t_len, dtype=torch.bool)

    if op_mask_sample.dim() == 2:
        if op_mask_sample.size(-1) > 1:
            valid = op_mask_sample[:t_len, 0]
        else:
            valid = op_mask_sample[:t_len, 0]
    elif op_mask_sample.dim() == 1:
        valid = op_mask_sample[:t_len]
    else:
        raise ValueError(f"Unsupported op_mask_sample dim: {op_mask_sample.dim()}")
    return valid > 0


def compute_single_traj_ade_fde(
    pred_sample: torch.Tensor,
    fut_sample: torch.Tensor,
    op_mask_sample: Optional[torch.Tensor] = None,
    feet_to_meter: float = 0.3048,
) -> dict:
    t_len = min(pred_sample.shape[0], fut_sample.shape[0])
    if t_len <= 0:
        return {
            "ade_ft": 0.0,
            "fde_ft": 0.0,
            "ade_m": 0.0,
            "fde_m": 0.0,
        }

    pred_b = pred_sample[:t_len].unsqueeze(0)
    fut_b = fut_sample[:t_len].unsqueeze(0)

    op_mask_b = None
    if op_mask_sample is not None:
        valid_mask = _valid_mask_1d(op_mask_sample, t_len).to(dtype=pred_b.dtype)
        op_mask_b = valid_mask.unsqueeze(0)

    metrics = compute_batch_metrics(
        pred=pred_b,
        target=fut_b,
        op_mask=op_mask_b,
        t_max=t_len,
        unit_conversion=1.0,
    )
    ade_ft = metrics["ade"]
    fde_ft = metrics["fde"]

    return {
        "ade_ft": float(ade_ft.item()),
        "fde_ft": float(fde_ft.item()),
        "ade_m": float((ade_ft * feet_to_meter).item()),
        "fde_m": float((fde_ft * feet_to_meter).item()),
    }


def _extract_sample_neighbors(
    nbrs: Optional[torch.Tensor],
    sample_index: int,
    hist_len: int,
    temporal_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    nbrs = _as_cpu_tensor(nbrs)
    temporal_mask = _as_cpu_tensor(temporal_mask)
    if nbrs is None:
        return None

    # Preferred path: dense neighbor tensor.
    if nbrs.dim() == 4:
        # [B, T, N, D]
        if nbrs.shape[1] == hist_len:
            sample = nbrs[sample_index]
            return sample if sample.dim() == 3 else None
        # [B, N, T, D]
        if nbrs.shape[2] == hist_len:
            sample = nbrs[sample_index].permute(1, 0, 2).contiguous()
            return sample if sample.dim() == 3 else None
        return None

    # Compact format fallback: [N_total, T, D]
    if nbrs.dim() == 3 and nbrs.shape[1] == hist_len:
        if temporal_mask is None or temporal_mask.dim() < 4:
            return nbrs.permute(1, 0, 2).contiguous()
        cell_mask = temporal_mask[..., 0].reshape(temporal_mask.shape[0], -1).bool()
        nbr_counts = cell_mask.sum(dim=1).long()
        start_idx = int(nbr_counts[:sample_index].sum().item())
        end_idx = int(start_idx + nbr_counts[sample_index].item())
        if end_idx <= start_idx:
            return None
        return nbrs[start_idx:end_idx].permute(1, 0, 2).contiguous()

    return None


def _extract_sample_trajectory(x: torch.Tensor, sample_index: int) -> torch.Tensor:
    if x.dim() == 3:
        return x[sample_index]
    if x.dim() == 2:
        return x
    raise ValueError(f"Unsupported trajectory shape: {tuple(x.shape)}")


def visualize_hist_nbrs_fut_pred(
    hist: torch.Tensor,
    nbrs: Optional[torch.Tensor],
    fut: torch.Tensor,
    pred: torch.Tensor,
    op_mask: Optional[torch.Tensor] = None,
    sample_index: int = 0,
    save_path: Optional[str] = None,
    title_prefix: str = "",
    feet_to_meter: float = 0.3048,
    temporal_mask: Optional[torch.Tensor] = None,
) -> Tuple[dict, Optional[str]]:
    hist = _as_cpu_tensor(hist)
    fut = _as_cpu_tensor(fut)
    pred = _as_cpu_tensor(pred)
    op_mask = _as_cpu_tensor(op_mask)

    if hist is None or fut is None or pred is None:
        raise ValueError("hist, fut and pred must not be None.")

    if hist.dim() == 3:
        sample_index = max(0, min(int(sample_index), hist.shape[0] - 1))
    else:
        sample_index = 0

    hist_sample = _extract_sample_trajectory(hist, sample_index)
    fut_sample = _extract_sample_trajectory(fut, sample_index)
    pred_sample = _extract_sample_trajectory(pred, sample_index)

    op_mask_sample = None
    if op_mask is not None:
        if op_mask.dim() == 3:
            op_mask_sample = op_mask[sample_index]
        elif op_mask.dim() == 2:
            op_mask_sample = op_mask[sample_index]
        elif op_mask.dim() == 1:
            op_mask_sample = op_mask

    metrics = compute_single_traj_ade_fde(
        pred_sample=pred_sample,
        fut_sample=fut_sample,
        op_mask_sample=op_mask_sample,
        feet_to_meter=feet_to_meter,
    )

    nbrs_sample = _extract_sample_neighbors(
        nbrs=nbrs,
        sample_index=sample_index,
        hist_len=hist_sample.shape[0],
        temporal_mask=temporal_mask,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    if nbrs_sample is not None and nbrs_sample.dim() == 3:
        for nbr_idx in range(nbrs_sample.shape[1]):
            nbr_traj = nbrs_sample[:, nbr_idx, :2]
            if torch.max(torch.abs(nbr_traj)).item() < 1e-6:
                continue
            ax.plot(
                nbr_traj[:, 1],
                nbr_traj[:, 0],
                color="orange",
                linewidth=1.0,
                alpha=0.6,
            )

    ax.plot(hist_sample[:, 1], hist_sample[:, 0], "o-", color="blue", linewidth=2.0, markersize=3, label="hist")
    ax.plot(fut_sample[:, 1], fut_sample[:, 0], "o-", color="green", linewidth=2.0, markersize=3, label="fut")
    ax.plot(pred_sample[:, 1], pred_sample[:, 0], "o-", color="red", linewidth=2.0, markersize=3, label="pred")

    for y_val in [18, 6, -6, -18]:
        ax.axhline(y=y_val, color="gray", linestyle=":", linewidth=0.7)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlabel("Longitudinal")
    ax.set_ylabel("Lateral")

    title_base = (
        f"ADE {metrics['ade_ft']:.4f} ft ({metrics['ade_m']:.4f} m) | "
        f"FDE {metrics['fde_ft']:.4f} ft ({metrics['fde_m']:.4f} m)"
    )
    if title_prefix:
        ax.set_title(f"{title_prefix} | {title_base}")
    else:
        ax.set_title(title_base)
    ax.legend(loc="best")
    plt.tight_layout()

    saved_file = None
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path_obj), dpi=160)
        saved_file = str(save_path_obj)
        plt.close(fig)
    else:
        # Block here so each forward_train/forward_eval call visibly shows a window.
        backend_name = matplotlib.get_backend()
        if backend_name.lower() == "agg":
            print(
                "[traj_vis_metrics] matplotlib backend is 'agg' (non-interactive), "
                "cannot open GUI window. Set MPLBACKEND=TkAgg or run with GUI display."
            )
        try:
            plt.show(block=True)
        except TypeError:
            plt.show()
        plt.close(fig)

    metrics_out = dict(metrics)
    metrics_out["sample_index"] = int(sample_index)
    return metrics_out, saved_file
