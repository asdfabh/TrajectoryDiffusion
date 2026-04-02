from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch


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

    coord_sq = diff[valid] ** 2
    return {
        "ade": float(dist_valid.mean()),
        "fde": float(dist_valid[-1]),
        "rmse": float(np.sqrt(coord_sq.mean())),
    }


def _format_prob_list(prob_values):
    prob_arr = _to_numpy(prob_values)
    if prob_arr is None:
        return "[]"
    prob_arr = np.asarray(prob_arr).reshape(-1)
    return "[" + ", ".join(f"{float(v):.2f}" for v in prob_arr) + "]"


def _format_joint_label(joint_idx, num_lon_classes=3):
    if joint_idx is None:
        return "-"
    joint_val = int(joint_idx)
    lat_idx = joint_val // int(num_lon_classes)
    lon_idx = joint_val % int(num_lon_classes)
    return f"J{joint_val + 1}(lat_{lat_idx + 1}+lon_{lon_idx + 1})"


def _format_optional_index(name, value, one_based=False):
    arr = _to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return None
    idx_val = int(arr[0])
    if one_based:
        idx_val += 1
    return f"{name}={idx_val}"


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
                                 pred_all=None, pred_best_idx=None, pred_selected_idx=None,
                                 anchor_all=None,
                                 future_mask=None, pred_instant=None, intent_probs=None, intent_meta=None,
                                 batch_idx=0, metrics=None, input_unit="ft"):
    """绘制 future 预测结果，支持单模态与多模态最佳轨迹高亮。"""

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
    anchor_modes = safe_get_batch(anchor_all, idx, batch_ndim=4)
    pred_instant_vis = normalize_traj2d(safe_get_batch(pred_instant, idx, batch_ndim=3))
    intent_lat = safe_get_batch(None if intent_probs is None else intent_probs.get("lat"), idx, batch_ndim=2)
    intent_lon = safe_get_batch(None if intent_probs is None else intent_probs.get("lon"), idx, batch_ndim=2)
    intent_joint = safe_get_batch(None if intent_probs is None else intent_probs.get("joint"), idx, batch_ndim=2)
    pred_joint_idx = safe_get_batch(None if intent_meta is None else intent_meta.get("pred_joint_idx"), idx, batch_ndim=1)
    gt_joint_idx = safe_get_batch(None if intent_meta is None else intent_meta.get("gt_joint_idx"), idx, batch_ndim=1)
    oracle_joint_idx = safe_get_batch(None if intent_meta is None else intent_meta.get("oracle_joint_idx"), idx, batch_ndim=1)
    routed_sub_idx = safe_get_batch(None if intent_meta is None else intent_meta.get("routed_sub_idx"), idx, batch_ndim=1)
    best_sub_idx = safe_get_batch(None if intent_meta is None else intent_meta.get("best_sub_idx"), idx, batch_ndim=1)

    if pred_modes is not None:
        pred_modes = np.asarray(pred_modes)
        if pred_modes.ndim == 2 and pred_modes.shape[-1] >= 2:
            pred_modes = pred_modes[None, ...]
        if pred_modes.ndim != 3 or pred_modes.shape[-1] < 2:
            pred_modes = None

    if anchor_modes is not None:
        anchor_modes = np.asarray(anchor_modes)
        if anchor_modes.ndim == 2 and anchor_modes.shape[-1] >= 2:
            anchor_modes = anchor_modes[None, ...]
        if anchor_modes.ndim != 3 or anchor_modes.shape[-1] < 2:
            anchor_modes = None

    selected_k = resolve_best_idx(pred_selected_idx, idx, 0 if pred_modes is None else pred_modes.shape[0])
    best_k = resolve_best_idx(pred_best_idx, idx, 0 if pred_modes is None else pred_modes.shape[0])
    if best_k is None and pred_modes is not None:
        best_k = select_best_by_ade(pred_modes, gt_fut, fut_mask_arr)

    pred_best = pred_single
    if pred_best is None and pred_modes is not None and pred_modes.shape[0] > 0:
        pred_best = normalize_traj2d(pred_modes[0 if best_k is None else best_k])

    nbrs_vis = reconstruct_nbrs_from_mask(hist_nbrs, temporal_mask, idx)

    fig, ax = plt.subplots(figsize=(10, 10))
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
        ax.plot(gt_hist[:, 1], gt_hist[:, 0], 'b-o', label='Hist', markersize=4, linewidth=2, alpha=0.8)
    if gt_fut is not None:
        ax.plot(gt_fut[:, 1], gt_fut[:, 0], 'g-o', label='GT Future', markersize=4, linewidth=2, alpha=0.8)
    if pred_modes is not None:
        non_best_labeled = False
        for k in range(pred_modes.shape[0]):
            traj = normalize_traj2d(pred_modes[k])
            if traj is None:
                continue
            is_selected = (selected_k is not None and k == selected_k)
            is_global_best = (best_k is not None and k == best_k)
            label = None
            color = '#5DADE2'
            alpha = 0.65
            linewidth = 1.4
            marker = '+'
            markersize = 6
            markeredgewidth = 1.2
            zorder = 2
            if is_selected and is_global_best:
                label = 'Selected = Global Best'
                color = '#D4AF37'
                alpha = 0.98
                linewidth = 2.2
                marker = 'o'
                markersize = 4
                markeredgewidth = 1.0
                zorder = 4
            elif is_selected:
                label = 'Selected'
                color = '#FF0000'
                alpha = 0.95
                linewidth = 2.0
                marker = 'o'
                markersize = 4
                markeredgewidth = 1.0
                zorder = 4
            elif is_global_best:
                label = 'Global Best'
                color = '#F39C12'
                alpha = 0.95
                linewidth = 2.0
                marker = 'o'
                markersize = 4
                markeredgewidth = 1.0
                zorder = 4
            elif not non_best_labeled:
                label = 'Pred Modes'
                non_best_labeled = True
            ax.plot(
                traj[:, 1], traj[:, 0],
                linestyle='-',
                marker=marker,
                markersize=markersize,
                markeredgewidth=markeredgewidth,
                linewidth=linewidth,
                alpha=alpha,
                color=color,
                label=label,
                zorder=zorder,
            )
    elif pred_best is not None:
        ax.plot(pred_best[:, 1], pred_best[:, 0], 'r-o', label='Pred', markersize=4, linewidth=2, alpha=0.9)

    # Draw best anchor trajectory (粗预测锚点，仅显示 best 模态对应的 anchor)
    if anchor_modes is not None and best_k is not None and anchor_modes.shape[0] > best_k:
        anc = normalize_traj2d(anchor_modes[best_k])
        if anc is not None:
            ax.plot(
                anc[:, 1], anc[:, 0],
                linestyle='--',
                marker='x',
                markersize=5,
                markeredgewidth=1.5,
                linewidth=1.5,
                alpha=0.80,
                color='#9B59B6',
                label='Best Anchor',
                zorder=4,
            )

    if pred_instant_vis is not None:
        ax.plot(
            pred_instant_vis[:, 1], pred_instant_vis[:, 0],
            linestyle='--',
            marker='s',
            markersize=4,
            linewidth=2.0,
            alpha=0.95,
            color='#F39C12',
            label='Pred Instant',
            zorder=4,
        )

    if intent_lat is not None or intent_lon is not None or intent_joint is not None:
        text_lines = []
        if intent_lat is not None:
            text_lines.append(f"lat={_format_prob_list(intent_lat)}")
        if intent_lon is not None:
            text_lines.append(f"lon={_format_prob_list(intent_lon)}")
        if intent_joint is not None:
            text_lines.append(f"joint={_format_prob_list(intent_joint)}")
        if pred_joint_idx is not None:
            text_lines.append(f"pred_joint={_format_joint_label(pred_joint_idx)}")
        if oracle_joint_idx is not None:
            text_lines.append(f"oracle_joint={_format_joint_label(oracle_joint_idx)}")
        if gt_joint_idx is not None:
            text_lines.append(f"gt_joint={_format_joint_label(gt_joint_idx)}")
        routed_sub_text = _format_optional_index("pred_sub", routed_sub_idx, one_based=True)
        best_sub_text = _format_optional_index("best_sub", best_sub_idx, one_based=True)
        if routed_sub_text is not None:
            text_lines.append(routed_sub_text)
        if best_sub_text is not None:
            text_lines.append(best_sub_text)
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=10,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#555555'),
            zorder=5,
        )

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
    ax.set_aspect('auto')
    ax.set_ylim(-18, 18)
    plt.tight_layout()
    plt.show()


def maybe_visualize_future_prediction(hist, hist_nbrs, temporal_mask, future, pred, valid_mask, stage,
                                      enable_train_vis=False, enable_eval_vis=False,
                                      pred_all=None, pred_best_idx=None, pred_selected_idx=None, anchor_all=None,
                                      pred_instant=None, intent_probs=None, intent_meta=None,
                                      meter_per_foot=0.3048, batch_idx=0):
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
    instant_metrics = _compute_vis_metrics(
        pred_traj=None if pred_instant is None else pred_instant[batch_idx],
        target_traj=future[batch_idx],
        valid_mask=valid_mask[batch_idx],
    )
    if instant_metrics is not None:
        metrics["ADE(inst traj)"] = {
            "ft": instant_metrics["ade"],
            "m": instant_metrics["ade"] * meter_per_foot,
        }
        metrics["FDE(inst traj)"] = {
            "ft": instant_metrics["fde"],
            "m": instant_metrics["fde"] * meter_per_foot,
        }
        metrics["RMSE(inst traj)"] = {
            "ft": instant_metrics["rmse"],
            "m": instant_metrics["rmse"] * meter_per_foot,
        }

    visualize_batch_trajectories(
        hist=hist,
        hist_nbrs=hist_nbrs,
        temporal_mask=temporal_mask,
        future=future,
        pred=pred,
        pred_all=pred_all,
        pred_best_idx=pred_best_idx,
        pred_selected_idx=pred_selected_idx,
        anchor_all=anchor_all,
        future_mask=valid_mask,
        pred_instant=pred_instant,
        intent_probs=intent_probs,
        intent_meta=intent_meta,
        batch_idx=batch_idx,
        metrics=metrics,
        input_unit="ft",
    )
