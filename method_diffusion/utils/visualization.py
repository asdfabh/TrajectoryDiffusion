from matplotlib import pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch

def plot_traj(hist, fut=None, fut_pred=None, nbrs=None, fig_num1=3, fig_num2=3, figsize=(15, 6), is_compare=False):
    fig_num1 = len(hist) // fig_num2 + (len(hist) % fig_num2 > 0)
    high = fig_num1 * 5
    wight = fig_num2 * 6
    figsize = (wight, high)
    fig, axes = plt.subplots(fig_num1, fig_num2, figsize=figsize)
    axes_flat = axes.flatten()
    # axes_flat = np.atleast_1d(axes).ravel() 据说可能会鲁棒一点
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    num_plot = len(hist)

    ncols = fig_num2
    axes_flat = np.atleast_1d(axes).ravel()
    row_xlim = {}

    for i in range(num_plot):
        ax = axes_flat[i]
        sample = hist[i]
        if isinstance(sample, torch.Tensor):
            sample = sample.detach().cpu().numpy()
        if type(sample) is not np.ndarray:
            arr = np.asarray(sample)
            print(f"Converted sample to numpy array with shape {arr.shape}.")
        else:
            print("type of sample is numpy array.")
            arr = sample

        ax.plot(arr[:, 1], arr[:, 0], marker='o',linestyle='None', markersize=2, c='blue', label='History', zorder=4)
        # ax.plot(hist[i, :, 1], hist[i, :, 0], marker='o', markersize=2, c='blue', label='History', zorder=2)

        if fut is not None:
            fut_sample = fut[i]
            if isinstance(fut_sample, torch.Tensor):
                fut_sample = fut_sample.detach().cpu().numpy()
            if type(fut_sample) is not np.ndarray:
                fut_arr = np.asarray(fut_sample)
            else:
                fut_arr = fut_sample
            ax.plot(fut_arr[:, 1], fut_arr[:, 0], marker='o', linestyle='None', markersize=2, c='green', label='Future', zorder=4)

        if fut_pred is not None:
            pred_sample = fut_pred[i]
            if isinstance(pred_sample, torch.Tensor):
                pred_sample = pred_sample.detach().cpu().numpy()
            if type(pred_sample) is not np.ndarray:
                pred_arr = np.asarray(pred_sample)
            else:
                pred_arr = pred_sample
            ax.plot(pred_arr[:, 1], pred_arr[:, 0], marker='o', linestyle='None', markersize=2, c='red', label='Prediction', zorder=4)

        # 设置y轴范围
        ax.set_ylim(-20, 20)
        for y in [18, 6, -6, -18]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=2, zorder=2)

        # if is_compare:
        #     # 设置每一行的x轴范围相同
        #     row = i // ncols
        #     if i % ncols == 0:
        #         row_xlim[row] = ax.get_xlim()
        #         ax.set_autoscale_on(False)
        #     else:
        #         if row in row_xlim:
        #             ax.set_xlim(row_xlim[row])
        #             ax.set_autoscale_on(False)

        # 绘制车辆矩形
        x_lim = ax.get_xlim()
        x_range = x_lim[1] - x_lim[0]
        scale = x_range / 1400 # compute the scale
        rect = patches.Rectangle((-40 * scale, -1.5), 80 * scale, 3, linewidth=2, edgecolor='#A12929', facecolor='#A12929', zorder=3)
        ax.add_patch(rect)

        # 绘制周围车辆轨迹与矩形
        if nbrs is None:
            continue

        nbrs_sample = nbrs[i]
        if isinstance(nbrs_sample, torch.Tensor):
            nbrs_sample = nbrs_sample.detach().cpu().numpy()

        if isinstance(nbrs_sample, np.ndarray):
            n_nbrs = nbrs_sample.shape[0]
        else:
            n_nbrs = len(nbrs_sample)

        for j in range(n_nbrs):
            # 逐个取出邻居并转换为 ndarray
            nbr_val = nbrs_sample[j]
            if isinstance(nbr_val, torch.Tensor):
                nbr_val = nbr_val.detach().cpu().numpy()
            nbr = np.asarray(nbr_val)
            if nbr.size == 0 or nbr.shape[0] == 0:
                continue

            ax.plot(nbr[:, 1], nbr[:, 0], marker='o', linestyle='None', markersize=2, c='orange', label='Neighbor', zorder=3)
            last_nbr_point = nbr[-1]  # 格式 [y, x]
            rect_x = last_nbr_point[1] - 40 * scale
            rect_y = last_nbr_point[0] - 1.5
            rect_nbr = patches.Rectangle((rect_x, rect_y), 80 * scale, 3, linewidth=1, edgecolor='#19196B',
                                         facecolor='#19196B', zorder=2)
            ax.add_patch(rect_nbr)

        # if is_compare:
            # 设置每一行的x轴范围相同
            row = i // ncols
            if i % ncols == 0:
                row_xlim[row] = ax.get_xlim()
                # ax.set_autoscale_on(False)
            else:
                if row in row_xlim:
                    ax.set_xlim(row_xlim[row])
                    # ax.set_autoscale_on(False)

    plt.show()


def plot_traj_with_mask(hist_original, hist_masked, hist_pred, nbrs_original=None, nbrs_masked=None, nbrs_pred=None,
                        fig_num1=3, fig_num2=3):
    """
    可视化原始、掩码和预测轨迹

    Args:
        hist_original: 原始历史轨迹 [B, T, 2]
        hist_masked: 掩码后的历史 [B, T, dim] (最后一维是观测标记)
        hist_pred: 预测轨迹 [B, T, 2]
        nbrs_*: 对应的邻车数据
    """
    num_samples = len(hist_original)
    fig_num1 = num_samples // fig_num2 + (num_samples % fig_num2 > 0)
    figsize = (fig_num2 * 6, fig_num1 * 5)

    fig, axes = plt.subplots(fig_num1, fig_num2, figsize=figsize)
    axes_flat = np.atleast_1d(axes).ravel()
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    for i in range(num_samples):
        ax = axes_flat[i]

        # 1. 绘制原始轨迹(蓝色)
        hist_orig = np.asarray(hist_original[i])
        ax.plot(hist_orig[:, 1], hist_orig[:, 0], 'o-', color='blue',
                markersize=5, linewidth=1.5, label='Original', zorder=4)

        # 2. 提取掩码位置并标记(红色x)
        obs_mask = hist_masked[i][:, -1].astype(bool)  # 观测标记
        masked_indices = ~obs_mask  # 被掩码的位置
        if masked_indices.any():
            masked_pos = hist_orig[masked_indices]
            ax.plot(masked_pos[:, 1], masked_pos[:, 0], 'rx',
                    markersize=8, markeredgewidth=2, label='Masked', zorder=5)

        # 3. 绘制预测轨迹(绿色)
        hist_p = np.asarray(hist_pred[i])
        ax.plot(hist_p[:, 1], hist_p[:, 0], 'o-', color='green',
                markersize=3, linewidth=1.5, label='Predicted', zorder=6)

        # 4. 绘制邻车轨迹
        if nbrs_original is not None and i < len(nbrs_original):
            nbrs_orig = nbrs_original[i]
            if isinstance(nbrs_orig, (list, tuple)):
                nbrs_orig = [np.asarray(n) for n in nbrs_orig]
            else:
                nbrs_orig = [np.asarray(nbrs_orig[j]) for j in range(nbrs_orig.shape[0])]

            for j, nbr in enumerate(nbrs_orig):
                if nbr.size == 0:
                    continue

                # 原始邻车(浅蓝)
                ax.plot(nbr[:, 1], nbr[:, 0], 'o-', color='lightblue',
                        markersize=2, linewidth=1, alpha=0.6, zorder=2)

                # 预测邻车(浅绿)
                if nbrs_pred is not None and i < len(nbrs_pred):
                    nbrs_p = nbrs_pred[i]
                    if isinstance(nbrs_p, (list, tuple)):
                        nbr_p = np.asarray(nbrs_p[j]) if j < len(nbrs_p) else None
                    else:
                        nbr_p = np.asarray(nbrs_p[j]) if j < nbrs_p.shape[0] else None

                    if nbr_p is not None and nbr_p.size > 0:
                        ax.plot(nbr_p[:, 1], nbr_p[:, 0], 'o-', color='lightgreen',
                                markersize=2, linewidth=1, alpha=0.6, zorder=2)

        # 5. 绘制车道线
        ax.set_ylim(-20, 20)
        for y in [18, 6, -6, -18]:
            ax.axhline(y=y, color='gray', linestyle='--', linewidth=1, zorder=1)

        # 6. 绘制自车矩形
        x_lim = ax.get_xlim()
        x_range = x_lim[1] - x_lim[0]
        scale = x_range / 1400
        rect = patches.Rectangle((-40 * scale, -1.5), 10 * scale, 2 * scale,
                                 linewidth=2, edgecolor='red', facecolor='red', zorder=5)
        ax.add_patch(rect)

        ax.set_xlabel('Lateral Position (m)')
        ax.set_ylabel('Longitudinal Position (m)')
        ax.set_title(f'Sample {i + 1}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for j in range(num_samples, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_batch_trajectories(hist=None, hist_nbrs=None, temporal_mask=None, future=None, pred=None,
                                 hist_masked=None, future_mask=None, batch_idx=None, save_path=None,
                                 metrics=None, input_unit=None, show_plot=None):
    """
    统一可视化函数（所有参数都允许 None）：
    1) 可视化 Ego 历史、邻车历史、真实 future、预测 future
    2) 输出并标记 FUT/PRED 最后一个点的坐标与 index
    """

    def to_numpy(data):
        if data is None:
            return None
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    def safe_get_batch(data, b_idx):
        arr = to_numpy(data)
        if arr is None:
            return None
        if arr.ndim >= 3 and arr.shape[0] > b_idx:
            return arr[b_idx]
        return arr

    def prepare_nbrs(nbrs_data, tmask_data, b_idx):
        nbrs_arr = to_numpy(nbrs_data)
        if nbrs_arr is None:
            return None

        # Case 1: already [B, T, N, D]
        if nbrs_arr.ndim == 4:
            return nbrs_arr[b_idx] if nbrs_arr.shape[0] > b_idx else None

        # Case 2: compact neighbors [N_total, T, D] + temporal_mask [B, H, W, D]
        if nbrs_arr.ndim == 3 and tmask_data is not None:
            tmask_arr = to_numpy(tmask_data)
            if tmask_arr is not None and tmask_arr.ndim == 4 and tmask_arr.shape[0] > b_idx:
                bmask = tmask_arr[b_idx].reshape(-1, tmask_arr.shape[-1])  # [N_grid, D]
                occ = bmask.any(axis=-1)
                occ_ids = np.where(occ)[0]

                t_len, feat_dim = nbrs_arr.shape[1], nbrs_arr.shape[2]
                n_grid = bmask.shape[0]
                out = np.zeros((t_len, n_grid, feat_dim), dtype=nbrs_arr.dtype)

                take = min(len(occ_ids), nbrs_arr.shape[0])
                if take > 0:
                    out[:, occ_ids[:take], :] = np.transpose(nbrs_arr[:take], (1, 0, 2))
                return out

        # Case 3: maybe single-sample [T, N, D]
        if nbrs_arr.ndim == 3:
            return nbrs_arr

        return None

    idx = 0 if batch_idx is None else int(batch_idx)
    unit = 'ft' if input_unit is None else input_unit
    if show_plot is None:
        show_plot = save_path is None

    recon_hist = safe_get_batch(hist, idx)
    gt_fut = safe_get_batch(future, idx)
    pred_fut = safe_get_batch(pred, idx)
    hist_mask_arr = safe_get_batch(hist_masked, idx)
    fut_mask_arr = safe_get_batch(future_mask, idx)
    nbrs_vis = prepare_nbrs(hist_nbrs, temporal_mask, idx)

    if recon_hist is not None and recon_hist.ndim == 3:
        recon_hist = recon_hist.squeeze(1)
    if gt_fut is not None and gt_fut.ndim == 3:
        gt_fut = gt_fut.squeeze(1)
    if pred_fut is not None and pred_fut.ndim == 3:
        pred_fut = pred_fut.squeeze(1)

    fig, ax = plt.subplots(figsize=(10, 10))
    info = {"fut_last": None, "pred_last": None}

    # A. Neighbors (浅黄色)
    if nbrs_vis is not None:
        if nbrs_vis.ndim == 2:
            nbrs_vis = nbrs_vis[:, None, :]
        for n_idx in range(nbrs_vis.shape[1]):
            traj = nbrs_vis[:, n_idx, :]
            if np.sum(np.abs(traj)) > 1e-2:
                ax.plot(
                    traj[:, 1], traj[:, 0],
                    color='#F9EEB2', alpha=0.95, linewidth=1.2, marker='.', markersize=2,
                    label='Neighbors' if n_idx == 0 else None, zorder=1
                )

    # B. Ego Hist
    gt_ego_traj = recon_hist
    if recon_hist is not None:
        ax.plot(recon_hist[:, 1], recon_hist[:, 0], 'b-o', label='Hist', markersize=4, linewidth=2, alpha=0.8)

    # C. Hist Mask
    if hist_mask_arr is not None and gt_ego_traj is not None:
        m = hist_mask_arr[..., -1] if (hist_mask_arr.ndim > 1 and hist_mask_arr.shape[-1] > 1) else hist_mask_arr.squeeze()
        L = min(len(m), len(gt_ego_traj))
        masked_ids = np.where(m[:L] < 0.5)[0]
        if len(masked_ids) > 0:
            ax.plot(gt_ego_traj[masked_ids, 1], gt_ego_traj[masked_ids, 0], 'rx',
                    markersize=8, markeredgewidth=2, label='Masked Input', zorder=10)

    # D. FUT + last point
    fut_last_idx = None
    if gt_fut is not None and len(gt_fut) > 0:
        ax.plot(gt_fut[:, 1], gt_fut[:, 0], 'g-*', label='GT Future', markersize=4, linewidth=2)
        if gt_ego_traj is not None and len(gt_ego_traj) > 0:
            ax.plot([gt_ego_traj[-1, 1], gt_fut[0, 1]], [gt_ego_traj[-1, 0], gt_fut[0, 0]], 'g--', alpha=0.5)

        if fut_mask_arr is not None:
            fm = np.asarray(fut_mask_arr).squeeze()
            if fm.ndim > 1:
                fm = fm[..., 0]
            valid_ids = np.where(fm[:len(gt_fut)] > 0.5)[0]
            fut_last_idx = int(valid_ids[-1]) if len(valid_ids) > 0 else len(gt_fut) - 1
        else:
            fut_last_idx = len(gt_fut) - 1

        fut_last_pt = gt_fut[fut_last_idx]
        ax.scatter(fut_last_pt[1], fut_last_pt[0], color='darkgreen', s=28, zorder=12)
        ax.text(
            fut_last_pt[1],
            fut_last_pt[0],
            f"({fut_last_pt[1]:.2f}, {fut_last_pt[0]:.2f}) {fut_last_idx}",
            color='darkgreen',
            fontsize=9,
                ha='left', va='bottom')

        info["fut_last"] = {"index": fut_last_idx, "coord": (float(fut_last_pt[1]), float(fut_last_pt[0]))}
        print(f"[Visualization] ({fut_last_pt[1]:.3f}, {fut_last_pt[0]:.3f}) {fut_last_idx}")

    # E. PRED + last point
    if pred_fut is not None and len(pred_fut) > 0:
        ax.plot(pred_fut[:, 1], pred_fut[:, 0], color='deepskyblue', linestyle='--', marker='x',
                label='Pred Future', markersize=4, linewidth=2)

        pred_last_idx = len(pred_fut) - 1 if fut_last_idx is None else min(len(pred_fut) - 1, fut_last_idx)
        pred_last_pt = pred_fut[pred_last_idx]
        ax.scatter(pred_last_pt[1], pred_last_pt[0], color='navy', s=28, zorder=12)
        ax.text(
            pred_last_pt[1],
            pred_last_pt[0],
            f"({pred_last_pt[1]:.2f}, {pred_last_pt[0]:.2f}) {pred_last_idx}",
            color='navy',
            fontsize=9,
                ha='left', va='top')

        info["pred_last"] = {"index": pred_last_idx, "coord": (float(pred_last_pt[1]), float(pred_last_pt[0]))}
        print(f"[Visualization] ({pred_last_pt[1]:.3f}, {pred_last_pt[0]:.3f}) {pred_last_idx}")

    for y in [18, 6, -6, -18]:
        ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

    if unit == 'ft':
        ax.set_xlabel('Longitudinal (ft)')
        ax.set_ylabel('Lateral (ft)')
        ft_to_m = lambda x: x * 0.3048
        m_to_ft = lambda x: x / 0.3048
        secax_x = ax.secondary_xaxis('top', functions=(ft_to_m, m_to_ft))
        secax_y = ax.secondary_yaxis('right', functions=(ft_to_m, m_to_ft))
        secax_x.set_xlabel('Longitudinal (m)')
        secax_y.set_ylabel('Lateral (m)')
    else:
        ax.set_xlabel('Longitudinal (m)')
        ax.set_ylabel('Lateral (m)')
        m_to_ft = lambda x: x / 0.3048
        ft_to_m = lambda x: x * 0.3048
        secax_x = ax.secondary_xaxis('top', functions=(m_to_ft, ft_to_m))
        secax_y = ax.secondary_yaxis('right', functions=(m_to_ft, ft_to_m))
        secax_x.set_xlabel('Longitudinal (ft)')
        secax_y.set_ylabel('Lateral (ft)')

    if metrics:
        metric_lines = []
        for name, values in metrics.items():
            if isinstance(values, dict):
                metric_lines.append(f"{name}: {values.get('m', 0.0):.3f} m | {values.get('ft', 0.0):.3f} ft")
            else:
                metric_lines.append(f"{name}: {values}")
        ax.text(
            0.02, 0.98, "\n".join(metric_lines),
            transform=ax.transAxes, va='top', ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.85, edgecolor='gray')
        )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close(fig)
    return info
