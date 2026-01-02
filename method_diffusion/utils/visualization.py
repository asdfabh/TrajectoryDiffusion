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
        if type(sample) is not np.ndarray:
            arr = np.asarray(sample)
            print(f"Converted sample to numpy array with shape {arr.shape}.")
        else:
            print("type of sample is numpy array.")
            arr = sample

        ax.plot(arr[:, 1], arr[:, 0], marker='o',linestyle='None', markersize=2, c='blue', label='History', zorder=4)
        # ax.plot(hist[i, :, 1], hist[i, :, 0], marker='o', markersize=2, c='blue', label='History', zorder=2)

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
        if isinstance(nbrs_sample, np.ndarray):
            n_nbrs = nbrs_sample.shape[0]
        else:
            n_nbrs = len(nbrs_sample)

        for j in range(n_nbrs):
            # 逐个取出邻居并转换为 ndarray
            nbr = np.asarray(nbrs_sample[j])
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


def plot_traj_with_mask(hist_original, hist_masked=None, hist_pred=None, nbrs_original=None, nbrs_masked=None, nbrs_pred=None,
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
        if hist_masked is not None:
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



def visualize_batch_trajectories(hist=None, hist_nbrs=None, future=None, path1=None, hist_masked=None, batch_idx=0,
                                 save_path=None):
    """
    可视化单个 Batch 的轨迹 (Ego + Neighbors + Future + Mask)
    Args:
        hist: [B, T, 1, 2] or [B, T, 2] (Ego History)
        hist_nbrs: [B, T, N, 2] (Neighbor History)
        future: [B, T_f, 1+N, 2] or [B, T_f, 2] (Future: Ego + Neighbors)
        path1: [B, T, N, 2] (Predicted Path)
        hist_masked: [B, T, N, D] (Masked History, last dim is mask)
        batch_idx: Index of the batch element to visualize (default 0)
        save_path: Path to save the figure (Optional)
    """
    import matplotlib.pyplot as plt

    # Helper to extract data for specific batch index and handle dimensions
    def extract_data(data, idx):
        if data is None:
            return None
        # Ensure CPU and numpy
        if isinstance(data, torch.Tensor):
            d = data[idx].detach().cpu().numpy()
        else:
            d = data[idx]
        return d

    ego_hist = extract_data(hist, batch_idx)
    nbrs_hist = extract_data(hist_nbrs, batch_idx)
    fut_data = extract_data(future, batch_idx)
    path1 = extract_data(path1, batch_idx)
    mask_data = extract_data(hist_masked, batch_idx)

    # Handle dimensions for ego_hist: [T, 1, 2] -> [T, 2]
    if ego_hist is not None:
        if ego_hist.ndim == 3 and ego_hist.shape[1] == 1:
            ego_hist = ego_hist.squeeze(1)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Ego History (Blue) & Neighbors from hist if available
    if ego_hist is not None:
        # ego_hist: [T, N, 2] or [T, 2]
        if ego_hist.ndim == 3:
            num_agents = ego_hist.shape[1]
            # Agent 0 is Ego
            traj = ego_hist[:, 0, :]
            if np.abs(traj).sum() > 1e-3:
                ax.plot(traj[:, 1], traj[:, 0], 'b-o', label='Ego History', markersize=4)

            # Agents 1..N are Neighbors (if any)
            for i in range(1, num_agents):
                traj = ego_hist[:, i, :]
                if np.abs(traj).sum() > 1e-3:
                    ax.plot(traj[:, 1], traj[:, 0], color='orange', marker='.', linestyle='-', markersize=2, alpha=0.6)

        elif ego_hist.ndim == 2:
            # Only Ego
            if np.abs(ego_hist).sum() > 1e-3:
                ax.plot(ego_hist[:, 1], ego_hist[:, 0], 'b-o', label='Ego History', markersize=4)

    # 2. Neighbors History (Yellow/Orange)
    if nbrs_hist is not None:
        # nbrs_hist: [T, N, 2]
        if nbrs_hist.ndim == 3:
            num_nbrs = nbrs_hist.shape[1]
            for i in range(num_nbrs):
                traj = nbrs_hist[:, i, :]
                # Check if neighbor is valid (not all zeros)
                if np.abs(traj).sum() > 1e-3:
                    ax.plot(traj[:, 1], traj[:, 0], color='orange', marker='.', linestyle='-', markersize=2, alpha=0.6)
        elif nbrs_hist.ndim == 2:
            if np.abs(nbrs_hist).sum() > 1e-3:
                ax.plot(nbrs_hist[:, 1], nbrs_hist[:, 0], color='orange', marker='.', linestyle='-', markersize=2,
                        alpha=0.6)

    # 3. Future (Green) - Ego + Neighbors
    if fut_data is not None:
        # fut_data: [T_f, M, 2] or [T_f, 2]
        if fut_data.ndim == 3:
            num_agents = fut_data.shape[1]
            for i in range(num_agents):
                traj = fut_data[:, i, :]
                if np.abs(traj).sum() > 1e-3:
                    # Ego is usually index 0
                    if i == 0:
                        ax.plot(traj[:, 1], traj[:, 0], color='green', marker='*', linestyle='-', markersize=4,
                                label='Future')
                    else:
                        ax.plot(traj[:, 1], traj[:, 0], color='green', marker='.', linestyle='-', markersize=2,
                                alpha=0.6)
        elif fut_data.ndim == 2:
            # Mark start and end
            if np.abs(fut_data).sum() > 1e-3:
                ax.plot(fut_data[:, 1], fut_data[:, 0], color='green', marker='*', linestyle='-', markersize=4,
                        label='Future')

    if path1 is not None:
        # path1: [T, N, 2]
        if path1.ndim == 3:
            num_nbrs = path1.shape[1]
            for i in range(num_nbrs):
                traj = path1[:, i, :]
                # Check if neighbor is valid (not all zeros)
                if np.abs(traj).sum() > 1e-3:
                    ax.plot(traj[:, 1], traj[:, 0], color='cyan', marker='.', linestyle='-', markersize=2,
                            alpha=0.6, label='Pred' if i == 0 else None)
        elif path1.ndim == 2:
            if np.abs(path1).sum() > 1e-3:
                ax.plot(path1[:, 1], path1[:, 0], color='cyan', marker='.', linestyle='-', markersize=2,
                        alpha=0.6, label='Pred')

    # 5. Mask Visualization (Red X)
    if mask_data is not None:
        # mask_data: [T, N, D]
        # Assume Agent 0 is Ego, 1..N are Neighbors
        T_mask, N_mask, D_mask = mask_data.shape

        for i in range(N_mask):
            traj_coords = None

            # Find coordinates for agent i
            if ego_hist is not None and ego_hist.ndim == 3 and ego_hist.shape[1] > i:
                traj_coords = ego_hist[:, i, :]
            elif i == 0 and ego_hist is not None:
                if ego_hist.ndim == 2:
                    traj_coords = ego_hist
                elif ego_hist.ndim == 3:
                    traj_coords = ego_hist[:, 0, :]
            elif i > 0 and nbrs_hist is not None:
                nbr_idx = i - 1
                if nbrs_hist.ndim == 3 and nbrs_hist.shape[1] > nbr_idx:
                    traj_coords = nbrs_hist[:, nbr_idx, :]
                elif nbrs_hist.ndim == 2 and nbr_idx == 0:
                    traj_coords = nbrs_hist

            if traj_coords is not None:
                # Check mask (last dim: 1=observed, 0=masked)
                mask_vec = mask_data[:, i, -1]
                is_masked = (mask_vec < 0.5)

                if np.any(is_masked):
                    min_len = min(len(traj_coords), len(is_masked))
                    masked_pos = traj_coords[:min_len][is_masked[:min_len]]

                    if len(masked_pos) > 0:
                        ax.plot(masked_pos[:, 1], masked_pos[:, 0], 'rx',
                                markersize=4, markeredgewidth=2,
                                label='Masked' if i == 0 else None, zorder=20)

    # 4. Lane Lines
    # Assuming Y-axis is Lateral (based on plot(x=Long, y=Lat))
    # Lane boundaries at 18, 6, -6, -18 (feet approx)
    for y in [18, 6, -6, -18]:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Longitudinal (m)')
    ax.set_ylabel('Lateral (m)')

    # Only add legend if there are labeled artists
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()

    # ax.axis('equal') # Removed for free scaling
    ax.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)
