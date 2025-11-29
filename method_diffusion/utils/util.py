import torch
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import random
from scipy.stats import gaussian_kde
import concurrent.futures
import matplotlib.patches as patches

sns.set_style("darkgrid")

# Loss function for training
def detr_loss(pred, gt, mask):
    acc = torch.zeros_like(mask)  # [seq_len, batch_size 2]
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_quries, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]

    muX = pred[:, :, 0]  # [batch_size, seq_len]
    muY = pred[:, :, 1]  # [batch_size, seq_len]
    sigX = pred[:, :, 2]  # [batch_size, seq_len]
    sigY = pred[:, :, 3]  # [batch_size, seq_len]
    rho = pred[:, :, 4]  # [batch_size, seq_len]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)  # [batch_size, seq_len]
    x = gt[:, :, 0]  # [batch_size, seq_len]
    y = gt[:, :, 1]
    out = 0.5 * torch.pow(ohr, 2) * (
            torch.pow(sigX, 2) * torch.pow(x - muX, 2) + torch.pow(sigY, 2) * torch.pow(y - muY, 2)
            - 2 * rho * torch.pow(sigX, 1) * torch.pow(sigY, 1) * (x - muX) * (y - muY)) - \
          torch.log(sigX * sigY * ohr) - 1.8379  # [batch_size, seq_len]
    acc[:, :, 0] = out.permute(1, 0)  # [seq_len, batch_size]
    acc[:, :, 1] = out.permute(1, 0)  # [seq_len, batch_size]
    acc = acc * mask  # [seq_len, batch_size, 2]
    loss_mse = torch.sum(acc) / torch.sum(mask)

    return loss_mse, nearest_mode_ids, nearest_mode_bs_ids

# Outputs a sequence of MSE values, uses mask for variable output lengths, used for evaluation
def detr_test_mse(pred, gt, mask):
    acc = torch.zeros_like(mask)
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_modes, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]
    muX = pred[:, :, 0]
    muY = pred[:, :, 1]
    x = gt[:, :, 0]
    y = gt[:, :, 1]
    out = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)
    acc[:, :, 0] = out.permute(1, 0)
    acc[:, :, 1] = out.permute(1, 0)
    acc = acc * mask
    loss_mse = torch.sum(acc) / torch.sum(mask)
    return loss_mse, nearest_mode_ids, nearest_mode_bs_ids

def detr_val_mse(pred, gt, mask):
    acc = torch.zeros_like(mask)
    gt = gt.permute(1, 0, 2)  # [batch_size, seq_len, 2]
    pred = pred.permute(1, 0, 2, 3)  # [batch_size, num_modes, seq_len, 5]
    batch_size = gt.shape[0]

    distance = (pred[:, :, :, 0:2] - gt[:, None, :, :]).norm(dim=-1)  # [batch_size, num_modes, seq_len]
    distance = distance.sum(dim=-1)  # [batch_size, num_modes]
    nearest_mode_ids = torch.argmin(distance, dim=-1)  # [batch_size]
    nearest_mode_bs_ids = torch.arange(batch_size).type_as(nearest_mode_ids)  # [batch_size]

    pred = pred[nearest_mode_bs_ids, nearest_mode_ids]  # [batch_size, seq_len, 5]
    muX = pred[:, :, 0]
    muY = pred[:, :, 1]
    x = gt[:, :, 0]
    y = gt[:, :, 1]
    out_rmse = torch.pow(x - muX, 2) + torch.pow(y - muY, 2)  # [batch_size, seq_len]
    out_de = torch.sqrt(out_rmse)  # [batch_size, seq_len]
    acc[:, :, 0] = out_rmse.permute(1, 0)
    acc[:, :, 1] = out_de.permute(1, 0)
    acc = acc * mask
    loss_val_rmse = torch.sum(acc[:, :, 0], dim=1)
    loss_val_de = torch.sum(acc[:, :, 1], dim=1)
    counts = torch.sum(mask[:, :, 0], dim=1)
    return loss_val_rmse, loss_val_de, counts, nearest_mode_ids, nearest_mode_bs_ids

# seed everything
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Custom activation for output layer (Graves, 2015)
def out_activation(x):
    muX = x[:, :, :, 0:1]
    muY = x[:, :, :, 1:2]
    sigX = x[:, :, :, 2:3]
    sigY = x[:, :, :, 3:4]
    rho = x[:, :, :, 4:5]
    sigX = torch.exp(sigX)
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho)
    out = torch.cat([muX, muY, sigX, sigY, rho], dim=3)
    return out

# Get the number of neighbors and its trajectory for each batch
def get_nbrs_trajectory(nbrs, nbrs_num_batch, b_index):
    nbr_num = nbrs_num_batch[b_index]
    start_index = int(sum(nbrs_num_batch[:b_index]))
    end_index = start_index + int(nbr_num)
    return nbrs[:, start_index:end_index, :]

# 默认传入hist类型为list， list中每个元素为numpy array
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

# traj size [T, 2]  mask size [T]， 默认数据类型为numpy array
def mask_traj(traj, mask):
    if type(mask) is not np.ndarray:
        # print(f"mask type: {type(mask)}, Changing to numpy array.")
        mask = mask.numpy()
    if type(traj) is not np.ndarray:
        # print(f"traj type: {type(traj)}, Changing to numpy array.")
        traj = traj.numpy()
    T = mask.shape[0]
    if len(traj) != T:
        raise ValueError(f"The length of traj and mask must be the same. traj shape: {traj.shape}, mask shape: {mask.shape}")
    traj = traj[mask]
    return traj

def random_mask_traj(traj, p=0.4):
    T = traj.shape[0]
    mask = np.random.rand(T) < p
    return mask

def random_prefix_keep_traj(traj, p=0.6):
    T = traj.shape[0]
    keep_len = np.random.randint(1, T * p)
    mask = np.zeros(T, dtype=bool)
    mask[T - keep_len:] = 1
    # print(f"keep_len: {keep_len}, mask = {mask}")
    return mask






















