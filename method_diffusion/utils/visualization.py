from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


def plot_traj_with_mask(hist_original, hist_masked, hist_pred, fig_num1=3, fig_num2=3, input_unit="ft"):
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


def maybe_visualize_hist_reconstruction(hist, hist_masked, pred, stage, enable_train_vis=False, enable_eval_vis=False,
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

    plot_traj_with_mask(
        hist_original=hist_vis,
        hist_masked=hist_masked_vis,
        hist_pred=pred_vis,
        fig_num1=1,
        fig_num2=1,
        input_unit=input_unit,
    )


def visualize_batch_trajectories(hist=None, hist_nbrs=None, temporal_mask=None, future=None, pred=None,
                                 pred_all=None, pred_best_idx=None,
                                 future_mask=None, batch_idx=0, metrics=None, input_unit="ft"):
    """绘制 future 预测结果，支持单模态与多模态最佳轨迹高亮。"""

    def to_numpy(data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def safe_get_batch(data, b_idx, batch_ndim=3):
        arr = to_numpy(data)
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
        nbrs_arr = to_numpy(nbrs_flat)
        mask_arr = to_numpy(tmask)
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
        arr = to_numpy(best_idx_data)
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

    def select_best_by_ade(pred_modes, gt_traj, fut_mask_arr):
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
        best_ade = float("inf")
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
            ade = float(np.linalg.norm(diff, axis=-1)[vm].mean())
            if ade < best_ade:
                best_ade = ade
                best_k = k
        return best_k

    idx = int(batch_idx)
    gt_hist = normalize_traj2d(safe_get_batch(hist, idx, batch_ndim=3))
    gt_fut = normalize_traj2d(safe_get_batch(future, idx, batch_ndim=3))
    pred_single = normalize_traj2d(safe_get_batch(pred, idx, batch_ndim=3))
    fut_mask_arr = safe_get_batch(future_mask, idx, batch_ndim=3)
    pred_modes = safe_get_batch(pred_all, idx, batch_ndim=4)

    if pred_modes is not None:
        pred_modes = np.asarray(pred_modes)
        if pred_modes.ndim == 2 and pred_modes.shape[-1] >= 2:
            pred_modes = pred_modes[None, ...]
        if pred_modes.ndim != 3 or pred_modes.shape[-1] < 2:
            pred_modes = None

    best_k = resolve_best_idx(pred_best_idx, idx, 0 if pred_modes is None else pred_modes.shape[0])
    if best_k is None and pred_modes is not None:
        best_k = select_best_by_ade(pred_modes, gt_fut, fut_mask_arr)

    pred_best = pred_single
    if pred_best is None and pred_modes is not None and pred_modes.shape[0] > 0:
        pred_best = normalize_traj2d(pred_modes[0 if best_k is None else best_k])

    nbrs_vis = reconstruct_nbrs_from_mask(hist_nbrs, temporal_mask, idx)

    fig, ax = plt.subplots(figsize=(10, 10))

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
        ax.plot(gt_hist[:, 1], gt_hist[:, 0], 'b-o', label='Hist', markersize=4, linewidth=2, alpha=0.8)
    if gt_fut is not None:
        ax.plot(gt_fut[:, 1], gt_fut[:, 0], 'g-o', label='GT Future', markersize=4, linewidth=2, alpha=0.8)
    if pred_modes is not None:
        non_best_labeled = False
        for k in range(pred_modes.shape[0]):
            traj = normalize_traj2d(pred_modes[k])
            if traj is None:
                continue
            is_best = (best_k is not None and k == best_k)
            label = None
            if is_best:
                label = 'Pred Best'
            elif not non_best_labeled:
                label = 'Pred Modes'
                non_best_labeled = True
            ax.plot(
                traj[:, 1], traj[:, 0],
                linestyle='-',
                marker='o' if is_best else '+',
                markersize=4 if is_best else 6,
                markeredgewidth=1.2 if not is_best else 1.0,
                linewidth=1.4 if not is_best else 2.0,
                alpha=0.65 if not is_best else 0.95,
                color='#FF0000' if is_best else '#5DADE2',
                label=label,
                zorder=3 if is_best else 2
            )
    elif pred_best is not None:
        ax.plot(pred_best[:, 1], pred_best[:, 0], 'r-o', label='Pred', markersize=4, linewidth=2, alpha=0.9)

    unit = input_unit or "ft"
    title = "Trajectory Visualization"
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
    ax.axis('equal')
    plt.tight_layout()
    plt.show()


def maybe_visualize_future_prediction(hist, hist_nbrs, temporal_mask, future, pred, valid_mask, stage,
                                      enable_train_vis=False, enable_eval_vis=False,
                                      pred_all=None, pred_best_idx=None, meter_per_foot=0.3048, batch_idx=0):
    """按配置开关控制 future 预测可视化。"""
    if stage == "train":
        if not enable_train_vis:
            return
    elif not enable_eval_vis:
        return

    diff = pred[batch_idx, :, :2] - future[batch_idx, :, :2]
    dist = torch.norm(diff, dim=-1)
    vis_mask = valid_mask[batch_idx]
    vis_ade = (dist * vis_mask).sum() / (vis_mask.sum() + 1e-6)
    valid_count = int(vis_mask.sum().item())
    vis_fde = dist[valid_count - 1] if valid_count > 0 else dist.new_tensor(0.0)
    metrics = {
        "ADE(vis traj)": {"ft": vis_ade.item(), "m": vis_ade.item() * meter_per_foot},
        "FDE(vis traj)": {"ft": vis_fde.item(), "m": vis_fde.item() * meter_per_foot},
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
        batch_idx=batch_idx,
        metrics=metrics,
        input_unit="ft",
    )
