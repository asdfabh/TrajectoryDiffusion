from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

MODE_COLOR_PALETTE = [
    "#E41A1C",
    "#377EB8",
    "#4DAF4A",
    "#FF7F00",
    "#984EA3",
    "#A65628",
    "#F781BF",
    "#17BECF",
    "#BCBD22",
]


def _to_numpy(data):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _compute_vis_metrics(pred_traj, target_traj, valid_mask):
    pred_arr = _to_numpy(pred_traj)
    target_arr = _to_numpy(target_traj)
    mask_arr = _to_numpy(valid_mask)
    if pred_arr is None or target_arr is None:
        return None

    pred_arr = np.asarray(pred_arr)
    target_arr = np.asarray(target_arr)
    if pred_arr.ndim != 2 or target_arr.ndim != 2 or pred_arr.shape[-1] < 2 or target_arr.shape[-1] < 2:
        return None

    time_steps = min(pred_arr.shape[0], target_arr.shape[0])
    if time_steps <= 0:
        return None

    if mask_arr is None:
        valid = np.ones(time_steps, dtype=bool)
    else:
        mask_arr = np.asarray(mask_arr).squeeze()
        if mask_arr.ndim > 1:
            mask_arr = mask_arr[..., 0]
        valid = mask_arr[:time_steps] > 0.5

    if not np.any(valid):
        return None

    diff = pred_arr[:time_steps, :2] - target_arr[:time_steps, :2]
    dist = np.linalg.norm(diff, axis=-1)
    dist_valid = dist[valid]
    if dist_valid.size == 0:
        return None

    return {
        "ade": float(dist_valid.mean()),
        "fde": float(dist_valid[-1]),
    }


def _get_mode_color(mode_idx):
    return MODE_COLOR_PALETTE[int(mode_idx) % len(MODE_COLOR_PALETTE)]


def plot_hist_reconstruction(hist_original, hist_masked, hist_pred, fig_num1=3, fig_num2=3, input_unit="ft"):
    """绘制 hist 重建结果：原始轨迹、掩码位置、预测轨迹。"""
    num_samples = len(hist_original)
    fig_num1 = num_samples // fig_num2 + (num_samples % fig_num2 > 0)
    figsize = (fig_num2 * 6, fig_num1 * 5)

    fig, axes = plt.subplots(fig_num1, fig_num2, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i in range(num_samples):
        ax = axes_flat[i]

        hist_orig = np.asarray(hist_original[i])
        hist_p = np.asarray(hist_pred[i])
        obs_mask = np.asarray(hist_masked[i])[:, -1].astype(bool)
        masked_indices = ~obs_mask

        ax.plot(hist_orig[:, 1], hist_orig[:, 0], 'o-', color='blue',
                markersize=5, linewidth=1.5, label='Original', zorder=4)
        if masked_indices.any():
            masked_pos = hist_orig[masked_indices]
            ax.plot(masked_pos[:, 1], masked_pos[:, 0], 'rx',
                    markersize=8, markeredgewidth=2, label='Masked', zorder=5)
        ax.plot(hist_p[:, 1], hist_p[:, 0], 'o-', color='green',
                markersize=3, linewidth=1.5, label='Predicted', zorder=6)

        ax.set_ylim(-20, 20)
        for y in [18, 6, -6, -18]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=1, zorder=1)

        x_lim = ax.get_xlim()
        x_range = x_lim[1] - x_lim[0]
        scale = x_range / 1400 if abs(x_range) > 1e-6 else 1.0
        rect = patches.Rectangle(
            (-40 * scale, -1.5), 10 * scale, 2 * scale,
            linewidth=2, edgecolor='red', facecolor='red', zorder=5
        )
        ax.add_patch(rect)

        ax.set_xlabel(f'Lateral Position ({input_unit})')
        ax.set_ylabel(f'Longitudinal Position ({input_unit})')
        ax.set_title(f'Sample {i + 1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(num_samples, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_hist_reconstruction(hist, hist_masked, pred, stage, enable_train_vis=False, enable_eval_vis=False,
                                  batch_idx=0, input_unit="ft"):
    """按配置开关控制 hist 重建可视化。"""
    if stage == "train":
        if not enable_train_vis:
            return
    elif not enable_eval_vis:
        return

    hist_vis = hist[batch_idx:batch_idx + 1, :, :2].detach().cpu().numpy()
    hist_masked_vis = hist_masked[batch_idx:batch_idx + 1].detach().cpu().numpy()
    pred_vis = pred[batch_idx:batch_idx + 1, :, :2].detach().cpu().numpy()

    plot_hist_reconstruction(
        hist_original=hist_vis,
        hist_masked=hist_masked_vis,
        hist_pred=pred_vis,
        fig_num1=1,
        fig_num2=1,
        input_unit=input_unit,
    )


def _visualize_scene_prediction(hist=None, hist_nbrs=None, temporal_mask=None, future=None, pred=None,
                                pred_all=None, pred_best_idx=None, future_mask=None,
                                batch_idx=0, metrics=None, input_unit="ft",
                                title=None, highlight_label="Best",
                                hist_masked=None, hist_reconstructed=None):
    """绘制 fut model 最终预测结果。"""

    def safe_get_batch(data, b_idx, batch_ndim=3):
        arr = _to_numpy(data)
        if arr is None:
            return None
        if arr.ndim >= batch_ndim and arr.shape[0] > b_idx:
            return arr[b_idx]
        return arr

    def normalize_traj2d(arr):
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        if arr.ndim != 2 or arr.shape[-1] < 2:
            return None
        return arr

    def reconstruct_nbrs_from_mask(nbrs_flat, tmask, b_idx):
        nbrs_arr = _to_numpy(nbrs_flat)
        mask_arr = _to_numpy(tmask)
        if nbrs_arr is None or mask_arr is None:
            return None
        if nbrs_arr.ndim != 3 or mask_arr.ndim != 4:
            return None
        if b_idx < 0 or b_idx >= mask_arr.shape[0]:
            return None

        occ = mask_arr[..., 0] if mask_arr.shape[-1] > 0 else mask_arr[..., -1]
        occ = occ.astype(bool)
        occ_flat = occ.reshape(occ.shape[0], -1)

        counts = occ_flat.sum(axis=1).astype(int)
        start = int(counts[:b_idx].sum())
        cur_count = int(counts[b_idx])
        if cur_count <= 0 or start >= nbrs_arr.shape[0]:
            return None

        end = min(start + cur_count, nbrs_arr.shape[0])
        cur_nbrs = nbrs_arr[start:end]
        occ_ids = np.where(occ_flat[b_idx])[0]
        take = min(len(occ_ids), cur_nbrs.shape[0])
        if take <= 0:
            return None

        n_grid = occ_flat.shape[1]
        t_len, feat_dim = cur_nbrs.shape[1], cur_nbrs.shape[2]
        out = np.zeros((n_grid, t_len, feat_dim), dtype=cur_nbrs.dtype)
        out[occ_ids[:take], :, :] = cur_nbrs[:take, :, :]
        return out

    def resolve_best_idx(best_idx_data, b_idx, num_modes):
        if best_idx_data is None or num_modes <= 0:
            return None
        arr = _to_numpy(best_idx_data)
        if arr is None:
            return None
        if np.isscalar(arr):
            idx_val = int(arr)
        elif arr.ndim == 0:
            idx_val = int(arr.item())
        elif arr.ndim >= 1 and arr.shape[0] > b_idx:
            idx_val = int(np.asarray(arr[b_idx]).reshape(-1)[0])
        else:
            return None
        return max(0, min(num_modes - 1, idx_val))

    def select_best_by_rmse(pred_modes, gt_traj, fut_mask_arr):
        if pred_modes is None or pred_modes.ndim != 3 or pred_modes.shape[0] == 0:
            return None
        if gt_traj is None or gt_traj.ndim != 2 or gt_traj.shape[-1] < 2:
            return 0

        valid = np.ones(gt_traj.shape[0], dtype=bool)
        if fut_mask_arr is not None:
            fm = np.asarray(fut_mask_arr).squeeze()
            if fm.ndim > 1:
                fm = fm[..., 0]
            valid = fm[:gt_traj.shape[0]] > 0.5
            if not np.any(valid):
                valid = np.ones(gt_traj.shape[0], dtype=bool)

        best_k = 0
        best_rmse = float("inf")
        for k in range(pred_modes.shape[0]):
            traj = pred_modes[k]
            if traj.ndim != 2 or traj.shape[-1] < 2:
                continue
            t_len = min(traj.shape[0], gt_traj.shape[0], valid.shape[0])
            if t_len <= 0:
                continue
            vm = valid[:t_len]
            if not np.any(vm):
                continue
            diff = traj[:t_len, :2] - gt_traj[:t_len, :2]
            rmse = float(np.sqrt(np.sum(diff[vm] ** 2, axis=-1).mean()))
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
        return best_k

    idx = int(batch_idx)
    gt_hist = normalize_traj2d(safe_get_batch(hist, idx, batch_ndim=3))
    gt_fut = normalize_traj2d(safe_get_batch(future, idx, batch_ndim=3))
    pred_single = normalize_traj2d(safe_get_batch(pred, idx, batch_ndim=3))
    fut_mask_arr = safe_get_batch(future_mask, idx, batch_ndim=3)
    hist_masked_arr = safe_get_batch(hist_masked, idx, batch_ndim=3)
    hist_recon = normalize_traj2d(safe_get_batch(hist_reconstructed, idx, batch_ndim=3))
    pred_modes = safe_get_batch(pred_all, idx, batch_ndim=4)

    if pred_modes is not None:
        pred_modes = np.asarray(pred_modes)
        if pred_modes.ndim == 2 and pred_modes.shape[-1] >= 2:
            pred_modes = pred_modes[None, ...]
        if pred_modes.ndim != 3 or pred_modes.shape[-1] < 2:
            pred_modes = None

    best_k = resolve_best_idx(pred_best_idx, idx, 0 if pred_modes is None else pred_modes.shape[0])
    if best_k is None and pred_modes is not None:
        best_k = select_best_by_rmse(pred_modes, gt_fut, fut_mask_arr)

    pred_best = pred_single
    if pred_best is None and pred_modes is not None and pred_modes.shape[0] > 0:
        pred_best = normalize_traj2d(pred_modes[0 if best_k is None else best_k])

    nbrs_vis = reconstruct_nbrs_from_mask(hist_nbrs, temporal_mask, idx)

    fig, ax = plt.subplots(figsize=(16, 9))
    lane_lines = (-18.0, -6.0, 6.0, 18.0)

    for lane_y in lane_lines:
        ax.axhline(
            y=lane_y,
            color='gray',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            zorder=0,
        )

    if nbrs_vis is not None:
        first_nbr = True
        for n_idx in range(nbrs_vis.shape[0]):
            traj = np.asarray(nbrs_vis[n_idx])
            if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[-1] < 2:
                continue
            valid = np.any(np.abs(traj[:, :2]) > 1e-4, axis=1)
            if np.count_nonzero(valid) < 2:
                continue
            traj_valid = traj[valid]
            ax.plot(
                traj_valid[:, 1], traj_valid[:, 0],
                color='#D8B365', alpha=0.55, linewidth=1.1, linestyle='-',
                label='Neighbors' if first_nbr else None, zorder=1
            )
            first_nbr = False

    if gt_hist is not None:
        ax.plot(
            gt_hist[:, 1], gt_hist[:, 0],
            color='#1F77B4',
            linestyle='-',
            marker='o',
            markersize=4,
            linewidth=2.0,
            alpha=0.85,
            label='Original Hist',
            zorder=8,
        )

    if hist_masked_arr is not None and gt_hist is not None:
        hist_mask_arr = np.asarray(hist_masked_arr)
        if hist_mask_arr.ndim == 2 and hist_mask_arr.shape[-1] >= 1:
            obs_mask = hist_mask_arr[:, -1] > 0.5
            masked_idx = ~obs_mask
            if masked_idx.any():
                masked_points = gt_hist[:masked_idx.shape[0]][masked_idx]
                if masked_points.size > 0:
                    ax.scatter(
                        masked_points[:, 1],
                        masked_points[:, 0],
                        marker='x',
                        s=64,
                        linewidths=1.8,
                        color='#D62728',
                        label='Masked Hist Points',
                        zorder=6,
                    )

    if hist_recon is not None:
        ax.plot(
            hist_recon[:, 1], hist_recon[:, 0],
            color='#2CA02C',
            linestyle='-',
            marker='o',
            markersize=4,
            linewidth=2.2,
            alpha=0.9,
            label='Reconstructed Hist',
            zorder=5,
        )

    if gt_fut is not None:
        ax.plot(
            gt_fut[:, 1], gt_fut[:, 0],
            color='#00A65A',
            linestyle='-',
            marker='o',
            markersize=4,
            linewidth=2.2,
            alpha=0.9,
            label='GT Future',
            zorder=9,
        )
    if pred_modes is not None:
        for k in range(pred_modes.shape[0]):
            traj = normalize_traj2d(pred_modes[k])
            if traj is None:
                continue
            is_best = (best_k is not None and k == best_k)
            mode_color = _get_mode_color(k)
            mode_num = k + 1
            label = f"Mode {mode_num}"
            if is_best:
                label = f"{label} ({highlight_label})"
            ax.plot(
                traj[:, 1], traj[:, 0],
                linestyle='-',
                marker='o',
                markersize=4,
                markeredgewidth=1.0,
                linewidth=2.6 if is_best else 1.6,
                alpha=0.95 if is_best else 0.8,
                color=mode_color,
                label=label,
                zorder=4 if is_best else 3
            )
            end_y, end_x = traj[-1, 0], traj[-1, 1]
            ax.scatter(
                [end_x],
                [end_y],
                s=36 if is_best else 18,
                color=mode_color,
                marker='*' if is_best else 'o',
                zorder=4 if is_best else 3,
            )
            end_text = f"{mode_num} {highlight_label}" if is_best else f"{mode_num}"
            ax.text(
                end_x + 3.0,
                end_y + (0.25 if is_best else 0.15),
                end_text,
                color=mode_color,
                fontsize=9,
                weight='bold' if is_best else 'normal',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor=mode_color),
                zorder=4,
            )
    elif pred_best is not None:
        ax.plot(
            pred_best[:, 1], pred_best[:, 0],
            color='#E41A1C',
            linestyle='-',
            marker='o',
            markersize=4,
            linewidth=2.0,
            alpha=0.9,
            label='Pred',
            zorder=4,
        )

    unit = input_unit or "ft"
    title = "Trajectory Visualization" if title is None else str(title)
    if metrics:
        metric_parts = []
        for metric_name, metric_val in metrics.items():
            if isinstance(metric_val, dict) and unit in metric_val:
                metric_parts.append(f"{metric_name}: {metric_val[unit]:.3f} {unit}")
        if metric_parts:
            title = title + "\n" + " | ".join(metric_parts)
    ax.set_title(title)
    ax.set_xlabel(f"Lateral Position ({unit})")
    ax.set_ylabel(f"Longitudinal Position ({unit})")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('auto')
    ax.set_ylim(-18, 18)
    plt.tight_layout()
    plt.show()


def visualize_scene_prediction(hist, hist_nbrs, temporal_mask, future, pred, valid_mask,
                               pred_all=None, pred_best_idx=None,
                               meter_per_foot=0.3048, batch_idx=0, title=None,
                               highlight_label="Best", hist_masked=None,
                               hist_reconstructed=None):
    """统一场景可视化：hist、mask、重建、周车、future 与预测结果。"""
    vis_metrics = _compute_vis_metrics(
        pred_traj=pred[batch_idx],
        target_traj=future[batch_idx],
        valid_mask=valid_mask[batch_idx],
    )
    metrics = {
        "ADE": {"ft": 0.0, "m": 0.0},
        "FDE": {"ft": 0.0, "m": 0.0},
    }
    if vis_metrics is not None:
        metrics["ADE"] = {"ft": vis_metrics["ade"], "m": vis_metrics["ade"] * meter_per_foot}
        metrics["FDE"] = {"ft": vis_metrics["fde"], "m": vis_metrics["fde"] * meter_per_foot}
    _visualize_scene_prediction(
        hist=hist,
        hist_nbrs=hist_nbrs,
        temporal_mask=temporal_mask,
        future=future,
        pred=pred,
        pred_all=pred_all,
        pred_best_idx=pred_best_idx,
        future_mask=valid_mask,
        batch_idx=batch_idx,
        metrics=metrics,
        input_unit="ft",
        title=title,
        highlight_label=highlight_label,
        hist_masked=hist_masked,
        hist_reconstructed=hist_reconstructed,
    )
