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


def visualize_batch_trajectories(hist=None, hist_nbrs=None, future=None, pred=None, hist_masked=None, batch_idx=0,
                                 save_path=None, title=None):
    """
    可视化函数：
    - hist (Blue): Hist Model 输出的 Ego 重构轨迹
    - hist_nbrs (Yellow): 真实的完整历史 (Ego + Neighbors)
    - hist_masked (Red X): 在真实的 Ego 历史(hist_nbrs[0])上标记被掩码的点
    - future (Green): 真实未来轨迹
    - pred (Blue/Cyan): 预测未来轨迹
    """

    # --- 数据提取辅助函数 ---
    def get_val(tensor, b_idx):
        if tensor is None: return None
        if isinstance(tensor, torch.Tensor):
            return tensor[b_idx].detach().cpu().numpy()
        return tensor[b_idx]

    # 1. 提取数据
    recon_hist = get_val(hist, batch_idx)  # [T, D] (Hist Model Output)
    gt_context_hist = get_val(hist_nbrs, batch_idx)  # [T, N, D] (GT: Ego + Nbrs)
    gt_fut = get_val(future, batch_idx)  # [T_f, D]
    pred_fut = get_val(pred, batch_idx)  # [T_f, D]
    mask_arr = get_val(hist_masked, batch_idx)  # [T, ...]

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- A. 绘制 Hist Nbrs (所有真实历史，包括 Ego) - 黄色 ---
    # 这是背景信息
    gt_ego_traj = None
    if gt_context_hist is not None:
        # 确保形状是 [T, N, D]
        if gt_context_hist.ndim == 2:
            gt_context_hist = gt_context_hist[:, None, :]  # [T, 1, D]

        num_agents = gt_context_hist.shape[1]

        # 记录 GT Ego 以便后续画 Mask
        gt_ego_traj = gt_context_hist[:, 0, :]

        for i in range(num_agents):
            traj = gt_context_hist[:, i, :]
            # 过滤无效轨迹
            if np.sum(np.abs(traj)) > 1e-2:
                # 统一用黄色/橙色表示真实历史背景
                label = 'GT Hist (Context)' if i == 0 else None
                ax.plot(traj[:, 1], traj[:, 0], color='orange', alpha=0.6, linewidth=1.5, marker='.', markersize=2,
                        label=label)

    # --- B. 绘制 Mask 标记 (红色 X) ---
    # 逻辑：对 hist_nbrs 的第 0 个 (GT Ego) 进行掩码对比
    if mask_arr is not None and gt_ego_traj is not None:
        # 处理 Mask 维度
        if mask_arr.ndim > 1 and mask_arr.shape[-1] > 1:
            m = mask_arr[..., -1]  # 取最后一维作为 mask 标记
        else:
            m = mask_arr.squeeze()

        # 截取长度对齐
        L = min(len(m), len(gt_ego_traj))
        m = m[:L]
        traj_to_mark = gt_ego_traj[:L]

        # 找出被掩码的点 (假设值 < 0.5 为被 Mask)
        masked_indices = np.where(m < 0.5)[0]

        if len(masked_indices) > 0:
            ax.plot(traj_to_mark[masked_indices, 1], traj_to_mark[masked_indices, 0],
                    'rx', markersize=8, markeredgewidth=2, label='Masked Input', zorder=10)

    # --- C. 绘制 Hist Model Output (Ego 重构) - 蓝色 ---
    if recon_hist is not None:
        if recon_hist.ndim == 3: recon_hist = recon_hist.squeeze(1)
        ax.plot(recon_hist[:, 1], recon_hist[:, 0], 'b-o', label='Hist Model Pred', markersize=4, linewidth=2,
                alpha=0.8)

    # --- D. 绘制 Future (真实) - 绿色 ---
    if gt_fut is not None:
        if gt_fut.ndim == 3: gt_fut = gt_fut.squeeze(1)
        ax.plot(gt_fut[:, 1], gt_fut[:, 0], 'g-*', label='GT Future', markersize=4, linewidth=2)

        # 画一条连接线 (从 GT Hist 终点到 GT Fut 起点)
        if gt_ego_traj is not None:
            ax.plot([gt_ego_traj[-1, 1], gt_fut[0, 1]], [gt_ego_traj[-1, 0], gt_fut[0, 0]], 'g--', alpha=0.5)

    # --- E. 绘制 Pred (预测未来) - 蓝色/青色 ---
    if pred_fut is not None:
        if pred_fut.ndim == 3: pred_fut = pred_fut.squeeze(1)
        # 这里用 Cyan (青色) 或者 DodgerBlue 以区别于 Hist Model 的深蓝，保持蓝色系但有区分
        ax.plot(pred_fut[:, 1], pred_fut[:, 0], color='deepskyblue', linestyle='--', marker='x',
                label='Pred Future', markersize=4, linewidth=2)

    # 设置绘图属性
    # 车道线 (假设 Y 轴为横向 Lateral)
    for y in [18, 6, -6, -18]:
        ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

    ax.set_xlabel('Longitudinal (m)')
    ax.set_ylabel('Lateral (m)')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        # print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close(fig)
