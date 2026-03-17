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


def _to_numpy_if_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _normalize_ego_hist(hist_arr):
    if hist_arr is None:
        return None
    arr = np.asarray(hist_arr)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[1] == 1:
            arr = arr[:, 0, :]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    return arr


def _extract_nbrs_for_batch(hist_nbrs, temporal_mask, batch_idx):
    """
    统一邻居输入格式，输出 [N, T, D]。
    支持:
    - None
    - [B, N, T, D]
    - [B, T, N, D]
    - [N_total, T, D] + temporal_mask(用于恢复每个样本的邻居切片)
    """
    if hist_nbrs is None:
        return None

    nbr_data = _to_numpy_if_tensor(hist_nbrs)
    tm_data = _to_numpy_if_tensor(temporal_mask)

    # list/tuple: 视为 batch 维可索引结构
    if isinstance(nbr_data, (list, tuple)):
        if len(nbr_data) == 0:
            return None
        pick_idx = int(max(0, min(batch_idx, len(nbr_data) - 1)))
        sample = _to_numpy_if_tensor(nbr_data[pick_idx])
        sample = np.asarray(sample)
        if sample.ndim == 2:
            return sample[None, ...]
        if sample.ndim == 3:
            return sample
        return None

    nbr_arr = np.asarray(nbr_data)
    if nbr_arr.ndim == 2:
        # 单条邻居轨迹 [T, D]
        return nbr_arr[None, ...]

    if nbr_arr.ndim == 4:
        # [B, N, T, D] 或 [B, T, N, D]
        if batch_idx >= nbr_arr.shape[0]:
            return None
        sample = nbr_arr[batch_idx]
        if sample.ndim != 3:
            return None

        # 优先利用 temporal_mask 计数判断邻居轴。
        if tm_data is not None:
            tm_arr = np.asarray(tm_data)
            if tm_arr.ndim >= 2 and batch_idx < tm_arr.shape[0]:
                if tm_arr.ndim == 4:
                    occ = np.any(tm_arr > 0, axis=-1).reshape(tm_arr.shape[0], -1)
                elif tm_arr.ndim == 3:
                    occ = np.any(tm_arr > 0, axis=-1)
                else:
                    occ = tm_arr > 0
                cur_nbrs = int(occ.sum(axis=1).astype(np.int64)[batch_idx])
                if cur_nbrs >= 0:
                    # [T, N, D] -> [N, T, D]
                    if sample.shape[1] == cur_nbrs and sample.shape[0] != cur_nbrs:
                        return np.transpose(sample, (1, 0, 2))
                    # [N, T, D]
                    if sample.shape[0] == cur_nbrs:
                        return sample

        # 回退启发式：如果第 0 维更像时间长度，按 [T, N, D] 处理
        if sample.shape[0] in (16, 25, 26, 31, 50):
            return np.transpose(sample, (1, 0, 2))
        return sample

    if nbr_arr.ndim == 3:
        # 关键场景: collate 后扁平邻居 [N_total, T, D]
        # 通过 temporal_mask 恢复某个 batch 样本的邻居切片。
        if tm_data is not None:
            tm_arr = np.asarray(tm_data)
            if tm_arr.ndim >= 2 and batch_idx < tm_arr.shape[0]:
                if tm_arr.ndim == 4:
                    occ = np.any(tm_arr > 0, axis=-1).reshape(tm_arr.shape[0], -1)
                elif tm_arr.ndim == 3:
                    occ = np.any(tm_arr > 0, axis=-1)
                else:
                    occ = tm_arr > 0

                occ_counts = occ.sum(axis=1).astype(np.int64)
                start = int(occ_counts[:batch_idx].sum())
                cur = int(occ_counts[batch_idx])
                end = min(start + cur, nbr_arr.shape[0])
                if end > start:
                    return nbr_arr[start:end]
                return np.empty((0, nbr_arr.shape[1], nbr_arr.shape[2]), dtype=nbr_arr.dtype)

        # 无法从 temporal_mask 还原时，尽量按 [N, T, D] 解释
        return nbr_arr

    return None


def visualize_batch_trajectories(
        hist=None,
        hist_nbrs=None,
        future=None,
        pred=None,
        hist_masked=None,
        batch_idx=0,
        save_path=None,
        best_index=None,
        temporal_mask=None,
        pred_all=None,
        pred_best_idx=None,
        future_mask=None,
        metrics=None,
        input_unit='m',
        show_plot=True,
):
    """
    可视化函数 (支持多模态预测):
    - hist (Blue): Hist Model 输出的 Ego 重构轨迹
    - hist_nbrs (Yellow): 真实的完整历史 (Ego + Neighbors)
    - hist_masked (Red X): 在真实的 Ego 历史(hist_nbrs[0])上标记被掩码的点
    - future (Green): 真实未来轨迹
    - pred (Cyan): 预测未来轨迹 (支持多条)
    - best_index: 多模态下指定的最佳轨迹索引，突出显示为红色
    """

    # --- 数据提取辅助函数 ---
    def get_val(tensor, b_idx):
        if tensor is None: return None
        if isinstance(tensor, torch.Tensor):
            return tensor[b_idx].detach().cpu().numpy()
        return tensor[b_idx]

    # 1. 提取数据
    recon_hist = get_val(hist, batch_idx)  # [T, D]
    ego_hist = _normalize_ego_hist(recon_hist)
    nbr_hist = _extract_nbrs_for_batch(hist_nbrs, temporal_mask, batch_idx)  # [N, T, D] or None
    gt_fut = get_val(future, batch_idx)  # [T_f, D]
    pred_src = pred_all if pred_all is not None else pred
    pred_fut = get_val(pred_src, batch_idx)  # [K, T_f, D] / [T_f, K, D] / [T_f, D]
    mask_arr = get_val(hist_masked, batch_idx)  # [T, ...]

    fig, ax = plt.subplots(figsize=(10, 10))

    # 将 best_index 规范为当前 batch 样本的 int，避免张量布尔歧义
    best_index_src = pred_best_idx if pred_best_idx is not None else best_index
    best_idx = None
    if best_index_src is not None:
        if isinstance(best_index_src, torch.Tensor):
            idx_tensor = best_index_src
            if idx_tensor.numel() == 1:
                best_idx = int(idx_tensor.item())
            else:
                best_idx = int(idx_tensor[batch_idx].item())
        elif isinstance(best_index_src, (np.ndarray, list, tuple)):
            if np.asarray(best_index_src).ndim == 0:
                best_idx = int(best_index_src)
            else:
                best_idx = int(best_index_src[batch_idx])
        else:
            best_idx = int(best_index_src)

    # --- A. 绘制 Hist/Nbrs Context ---
    gt_ego_traj = ego_hist
    if gt_ego_traj is not None and np.sum(np.abs(gt_ego_traj[:, :2])) > 1e-2:
        ax.plot(
            gt_ego_traj[:, 1], gt_ego_traj[:, 0],
            color='orange', alpha=0.45, linewidth=1.2, marker='.', markersize=2,
            label='GT Hist (Ego)'
        )

    if nbr_hist is not None:
        nbr_arr = np.asarray(nbr_hist)
        if nbr_arr.ndim == 2:
            nbr_arr = nbr_arr[None, ...]
        if nbr_arr.ndim == 3:
            nbr_label_written = False
            for i in range(nbr_arr.shape[0]):
                traj = nbr_arr[i]
                if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
                    continue
                if np.sum(np.abs(traj[:, :2])) <= 1e-2:
                    continue
                ax.plot(
                    traj[:, 1], traj[:, 0],
                    color='orange', alpha=0.6, linewidth=1.0, marker='.', markersize=2,
                    label='GT Hist (Nbrs)' if not nbr_label_written else None
                )
                nbr_label_written = True

    # --- B. 绘制 Mask 标记 (红色 X) ---
    if mask_arr is not None and gt_ego_traj is not None:
        if mask_arr.ndim > 1 and mask_arr.shape[-1] > 1:
            m = mask_arr[..., -1]
        else:
            m = mask_arr.squeeze()

        L = min(len(m), len(gt_ego_traj))
        m = m[:L]
        traj_to_mark = gt_ego_traj[:L]
        masked_indices = np.where(m < 0.5)[0]

        if len(masked_indices) > 0:
            ax.plot(traj_to_mark[masked_indices, 1], traj_to_mark[masked_indices, 0],
                    'rx', markersize=8, markeredgewidth=2, label='Masked Input', zorder=10)

    # --- C. 绘制 Hist Model Output (Ego 重构) - 蓝色 ---
    if recon_hist is not None:
        if recon_hist.ndim == 3:
            if recon_hist.shape[0] == 1:
                recon_hist = recon_hist[0]
            elif recon_hist.shape[1] == 1:
                recon_hist = recon_hist[:, 0, :]
        if recon_hist.ndim == 2 and recon_hist.shape[1] >= 2:
            ax.plot(recon_hist[:, 1], recon_hist[:, 0], 'b-o', label='Hist Model Pred', markersize=4, linewidth=2,
                    alpha=0.8)

    # --- D. 绘制 Future (真实) - 绿色 ---
    if gt_fut is not None:
        if gt_fut.ndim == 3: gt_fut = gt_fut.squeeze(1)
        ax.plot(gt_fut[:, 1], gt_fut[:, 0], 'g-*', label='GT Future', markersize=5, linewidth=2)

        # 连线
        if gt_ego_traj is not None:
            ax.plot([gt_ego_traj[-1, 1], gt_fut[0, 1]], [gt_ego_traj[-1, 0], gt_fut[0, 0]], 'g--', alpha=0.5)

    # --- E. 绘制 Pred (预测未来) - 深天蓝 ---
    # 兼容三种形状：
    #   1) [T, 2] 单模态
    #   2) [T, K, 2] 多模态（时间优先）
    #   3) [K, T, 2] 多模态（模式优先）
    if pred_fut is not None:
        pred_arr = np.asarray(pred_fut)
        if pred_arr.ndim == 2:
            ax.plot(pred_arr[:, 1], pred_arr[:, 0], color='deepskyblue', linestyle='-', marker='o',
                    label='Pred Future', markersize=4, linewidth=1.5)
        elif pred_arr.ndim == 3:
            fut_len = gt_fut.shape[0] if gt_fut is not None and gt_fut.ndim >= 2 else None
            if pred_arr.shape[0] == 1:
                pred_arr = pred_arr[0]
                ax.plot(pred_arr[:, 1], pred_arr[:, 0], color='deepskyblue', linestyle='-', marker='o',
                        label='Pred Future', markersize=4, linewidth=1.5)
            elif pred_arr.shape[1] == 1:
                pred_arr = pred_arr[:, 0, :]
                ax.plot(pred_arr[:, 1], pred_arr[:, 0], color='deepskyblue', linestyle='-', marker='o',
                        label='Pred Future', markersize=4, linewidth=1.5)
            else:
                # time-first: [T, K, 2]
                if fut_len is not None and pred_arr.shape[0] == fut_len:
                    num_modes = pred_arr.shape[1]
                    mode_getter = lambda k: pred_arr[:, k, :]
                # mode-first: [K, T, 2]
                else:
                    num_modes = pred_arr.shape[0]
                    mode_getter = lambda k: pred_arr[k, :, :]

                for k in range(num_modes):
                    traj_k = mode_getter(k)
                    is_best = (best_idx is not None and k == best_idx)
                    color = 'red' if is_best else 'deepskyblue'
                    alpha = 0.95 if is_best else 0.6
                    marker = 'o' if is_best else 'x'
                    label = None
                    if is_best:
                        label = 'Pred Future (Best)'
                    elif k == 0:
                        label = 'Pred Future (Multi)'
                    ax.plot(traj_k[:, 1], traj_k[:, 0], color=color, linestyle='-', marker=marker,
                            markersize=4, linewidth=1.8 if is_best else 1.2, alpha=alpha, label=label)

    # 设置绘图属性
    for y in [18, 6, -6, -18]:
        ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

    ax.set_xlabel('Longitudinal (m)')
    ax.set_ylabel('Lateral (m)')
    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')

    if isinstance(metrics, dict) and len(metrics) > 0:
        lines = []
        for key, val in metrics.items():
            if isinstance(val, dict):
                if "ft" in val and "m" in val:
                    lines.append(f"{key}: {val['ft']:.3f} ft / {val['m']:.3f} m")
                else:
                    lines.append(f"{key}: {val}")
            else:
                lines.append(f"{key}: {val}")
        if lines:
            ax.text(
                0.02, 0.98, "\n".join(lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    if show_plot and not save_path:
        plt.show()

    plt.close(fig)
    vis_info = {}
    if gt_fut is not None and gt_fut.ndim >= 2 and gt_fut.shape[0] > 0:
        vis_info["fut_last"] = {"coord": (float(gt_fut[-1, 1]), float(gt_fut[-1, 0]))}
    if pred_fut is not None:
        pred_arr = np.asarray(pred_fut)
        pred_last = None
        if pred_arr.ndim == 2 and pred_arr.shape[0] > 0:
            pred_last = pred_arr[-1, :2]
        elif pred_arr.ndim == 3:
            if pred_arr.shape[0] == 1 and pred_arr.shape[1] > 0:
                pred_last = pred_arr[0, -1, :2]
            elif pred_arr.shape[1] == 1 and pred_arr.shape[0] > 0:
                pred_last = pred_arr[-1, 0, :2]
            else:
                idx = int(best_idx) if best_idx is not None else 0
                idx = max(0, min(idx, (pred_arr.shape[1] - 1 if pred_arr.shape[0] != 0 else 0)))
                if gt_fut is not None and pred_arr.shape[0] == gt_fut.shape[0]:
                    pred_last = pred_arr[-1, idx, :2]
                else:
                    idx = max(0, min(int(best_idx) if best_idx is not None else 0, pred_arr.shape[0] - 1))
                    pred_last = pred_arr[idx, -1, :2]
        if pred_last is not None:
            vis_info["pred_last"] = {"coord": (float(pred_last[1]), float(pred_last[0]))}
    return vis_info
