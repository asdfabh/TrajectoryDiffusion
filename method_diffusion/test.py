import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.net import DiffusionPast
from method_diffusion.config import get_args_parser
from method_diffusion.train import prepare_input_data


MAX_NEIGHBORS = 39  # 与 DiffusionPast.preprossess_input 中的稀疏槽数量保持一致


def visualize_batch_with_nbrs(gt_ego, masked_ego, pred_ego,
                              gt_nbrs_list, masked_nbrs_list, pred_nbrs_list,
                              save_path=None, show_nbrs=False):
    """
    可视化 ego 和 nbrs 的预测结果

    Args:
        gt_ego: [B, T, 2]
        masked_ego: [B, T, 2]
        pred_ego: [B, T, 2]
        gt_nbrs_list: list of [count_b, T, 2], 长度为 B
        masked_nbrs_list: list of [count_b, T, 2]
        pred_nbrs_list: list of [count_b, T, 2]
    """
    batch = gt_ego.shape[0]
    cols = min(3, batch)
    rows = math.ceil(batch / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = np.atleast_1d(axes).ravel()

    for i in range(batch):
        ax = axes[i]

        # ========== Ego 可视化 ==========
        # Ground Truth (蓝色)
        ax.scatter(gt_ego[i, :, 0], gt_ego[i, :, 1],
                  color='blue', s=30, label='Ego GT', alpha=0.7, marker='o')

        # Masked Input (金色叉号)
        valid = ~torch.isnan(masked_ego[i, :, 0])
        if valid.any():
            ax.scatter(masked_ego[i, valid, 0], masked_ego[i, valid, 1],
                      color='gold', s=40, marker='x', linewidths=2,
                      label='Ego Masked', alpha=0.8)

        # Prediction (绿色)
        ax.scatter(pred_ego[i, :, 0], pred_ego[i, :, 1],
                  color='green', s=30, marker='^', label='Ego Pred', alpha=0.7)

        # ========== Nbrs 可视化 ==========
        n_nbrs = 0  # 默认无邻车，防止 show_nbrs=False 时未定义
        if show_nbrs:
            gt_nbrs = gt_nbrs_list[i]
            masked_nbrs = masked_nbrs_list[i]
            pred_nbrs = pred_nbrs_list[i]

            n_nbrs = gt_nbrs.shape[0]
            if n_nbrs > 0:
                # Ground Truth (浅蓝色)
                for n in range(n_nbrs):
                    ax.scatter(gt_nbrs[n, :, 0], gt_nbrs[n, :, 1],
                              color='cyan', s=20, alpha=0.5, marker='o')

                # Masked Input (橙色叉号)
                for n in range(n_nbrs):
                    valid = ~torch.isnan(masked_nbrs[n, :, 0])
                    if valid.any():
                        ax.scatter(masked_nbrs[n, valid, 0], masked_nbrs[n, valid, 1],
                                  color='orange', s=25, marker='x', linewidths=1.5, alpha=0.6)

                # Prediction (红色)
                for n in range(n_nbrs):
                    ax.scatter(pred_nbrs[n, :, 0], pred_nbrs[n, :, 1],
                              color='red', s=20, marker='^', alpha=0.6)

        # 图例和格式
        ax.set_title(f'Sample {i} (Ego + {n_nbrs} Nbrs)', fontsize=10)
        ax.axis('equal')
        ax.grid(True, linestyle='--', alpha=0.3)

        # 只在第一个子图显示图例
        if i == 0:
            # 手动创建图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                      markersize=8, label='Ego GT'),
                Line2D([0], [0], marker='x', color='w', markerfacecolor='gold',
                      markersize=8, label='Ego Masked'),
                Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                      markersize=8, label='Ego Pred'),
            ]
            if show_nbrs:
                legend_elements.extend([
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan',
                           markersize=6, label='Nbrs GT', alpha=0.5),
                    Line2D([0], [0], marker='x', color='w', markerfacecolor='orange',
                           markersize=6, label='Nbrs Masked', alpha=0.6),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                           markersize=6, label='Nbrs Pred', alpha=0.6),
                ])
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # 删除多余的子图
    for j in range(batch, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def run_test(batch_size=5, mask_prob=0.2, num_inference_steps=50, save_path=None, show_nbrs=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = get_args_parser().parse_args([])
    args.batch_size = batch_size

    # 加载数据
    data_root = Path(__file__).resolve().parent / 'data/ngsimdata'
    data_path = str(data_root / 'TestSet.mat')
    dataset = NgsimDataset(data_path, t_h=30, t_f=50, d_s=2)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=dataset.collate_fn
    )
    batch = next(iter(loader))

    # 准备输入
    hist_in, nbrs_in, hist_mask, nbrs_mask, nbrs_num = prepare_input_data(
        batch, args.feature_dim, mask_type='random', mask_prob=mask_prob
    )

    hist_mask_cpu = hist_mask.clone()
    nbrs_mask_cpu = nbrs_mask.clone()

    hist_in = hist_in.to(device)
    nbrs_in = nbrs_in.to(device)
    hist_mask = hist_mask.to(device)
    nbrs_mask = nbrs_mask.to(device)
    nbrs_num = nbrs_num.to(device)

    # 模型推理
    model = DiffusionPast(args).to(device)
    model.eval()
    with torch.no_grad():
        loss, pred = model.forward_test(
            hist_in, hist_mask, nbrs_in, nbrs_mask, nbrs_num,
            num_inference_steps=num_inference_steps
        )
    print(f"Test loss (normalized space): {loss.item():.4f}")

    B, T = batch['hist'].shape[:2]
    N_total = nbrs_in.shape[0]

    # ✅ 1. 提取 ego 预测(归一化空间)
    pred_ego_norm = pred[:B, :, :]  # [B, T, 2]

    # ✅ 2. 提取 nbrs 预测(归一化空间)
    if N_total > 0:
        pred_nbrs_norm = pred[B:, :, :]  # [N_total, T, 2]
    else:
        pred_nbrs_norm = torch.empty(0, T, 2, device=device)

    # ========== 反归一化 ==========
    pos_mean = model.pos_mean.view(1, 1, -1)
    pos_std = model.pos_std.view(1, 1, -1)

    # Ego 反归一化
    ego_pred = pred_ego_norm * pos_std + pos_mean  # [B, T, 2]

    # Nbrs 反归一化
    if N_total > 0:
        nbrs_pred = pred_nbrs_norm * pos_std + pos_mean  # [N_total, T, 2]
    else:
        nbrs_pred = torch.empty(0, T, 2)

    # ========== 准备 ground truth 和 masked 数据 ==========
    # Ego
    gt_ego = batch['hist'][:, :, :2]  # [B, T, 2]
    masked_ego = gt_ego.clone()
    mask_bool = hist_mask_cpu.bool().unsqueeze(-1)
    masked_ego = masked_ego.masked_fill(~mask_bool, float('nan'))

    # Nbrs
    if N_total > 0:
        gt_nbrs = batch['nbrs'][:, :, :2]  # [N_total, T, 2]
        masked_nbrs = gt_nbrs.clone()
        nbrs_mask_bool = nbrs_mask_cpu.bool().unsqueeze(-1)
        masked_nbrs = masked_nbrs.masked_fill(~nbrs_mask_bool, float('nan'))
    else:
        gt_nbrs = torch.empty(0, T, 2)
        masked_nbrs = torch.empty(0, T, 2)

    # ========== 按批次重新分组 nbrs ==========
    nbrs_gt_batched = []
    nbrs_masked_batched = []
    nbrs_pred_batched = []

    offset = 0
    for b in range(B):
        count = int(nbrs_num[b])
        if count > 0:
            nbrs_gt_batched.append(gt_nbrs[offset:offset + count])
            nbrs_masked_batched.append(masked_nbrs[offset:offset + count])
            nbrs_pred_batched.append(nbrs_pred[offset:offset + count])
            offset += count
        else:
            nbrs_gt_batched.append(torch.empty(0, T, 2))
            nbrs_masked_batched.append(torch.empty(0, T, 2))
            nbrs_pred_batched.append(torch.empty(0, T, 2))

    # ========== 可视化 ==========
    visualize_batch_with_nbrs(
        gt_ego=gt_ego,
        masked_ego=masked_ego,
        pred_ego=ego_pred.cpu(),
        gt_nbrs_list=nbrs_gt_batched,
        masked_nbrs_list=nbrs_masked_batched,
        pred_nbrs_list=[n.cpu() for n in nbrs_pred_batched],
        save_path=save_path,
        show_nbrs=show_nbrs,
    )




if __name__ == '__main__':
    run_test(batch_size=5, mask_prob=0.4, num_inference_steps=50, show_nbrs=False)
