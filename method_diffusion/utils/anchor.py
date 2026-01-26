import os
import numpy as np
import torch
import matplotlib
from sympy.codegen.ast import none

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path

# 假设 ngsim_dataset 在 method_diffusion.dataset 包下
# 根据你的实际目录结构调整 import
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser


def rotate_traj_to_y_axis(traj, hist):
    """
    将轨迹旋转，使车头朝向 Y 轴正方向
    traj: Future trajectory [T, 2]
    hist: History trajectory [T_h, 2] (用于计算当前朝向)
    """

    if len(hist) < 2:
        return traj  # 历史太短，无法计算朝向，保持原样

    direction_vec = -hist[-2]  # 向量 (dx, dy)

    # 计算当前角度
    yaw = np.arctan2(direction_vec[1], direction_vec[0])
    alpha = np.pi / 2 - yaw

    c = np.cos(alpha)
    s = np.sin(alpha)
    R = np.array([[c, -s], [s, c]])

    traj_rotated = (R @ traj.T).T

    return traj_rotated


def visualize_anchors(anchors, save_path):
    K = anchors.shape[0]
    plt.figure(figsize=(12, 12))
    colors = plt.cm.jet(np.linspace(0, 1, K))

    for i in range(K):
        traj = anchors[i]
        plt.plot(traj[:, 1], traj[:, 0], label=f'Mode {i}',
                 linewidth=2.5, marker='*', markersize=3, color=colors[i], alpha=0.7)
        plt.plot(traj[0, 1], traj[0, 0], 'ks', markersize=6)

    plt.title(f'Generated {K} Anchors (Aligned to Y-Axis)', fontsize=14)
    plt.xlabel(' Y (ft)', fontsize=12)
    plt.ylabel(' X (ft)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.axis('equal')
    plt.xlim(-20, 20)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


def main():
    try:
        args = get_args_parser().parse_args()
    except:
        class Args:
            pass

    root_path = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
    mat_file = str(root_path / 'TrainSet.mat')

    print(f"Loading dataset from {mat_file}...")
    dataset = NgsimDataset(mat_file, t_h=30, t_f=50, d_s=2)

    all_futures = []
    print("Collecting and aligning trajectories...")

    stride = 10

    for i in tqdm(range(0, len(dataset), stride)):
        dsId = int(dataset.D[i, 0])
        vehId = int(dataset.D[i, 1])
        t = dataset.D[i, 2]

        hist = dataset.getHistory(vehId, t, vehId, dsId)
        fut = dataset.getFuture(vehId, t, dsId)

        # 过滤无效数据
        if len(fut) != args.T_f or len(hist) < 2:
            continue

        fut_aligned = rotate_traj_to_y_axis(fut, hist)
        all_futures.append(fut_aligned)

    if not all_futures:
        print("No valid trajectories found.")
        return

    all_futures = np.stack(all_futures).astype(np.float64)  # [N, T, 2]
    print(f"Collected {len(all_futures)} trajectories.")

    LATERAL_SCALE = 15.0

    features = all_futures.copy()
    features[:, :, 0] *= LATERAL_SCALE

    # Flatten [N, T*2]
    features_flat = features.reshape(features.shape[0], -1)

    print(f"Running KMeans (K={args.num_modes}, Scale={LATERAL_SCALE})...")
    kmeans = KMeans(n_clusters=args.num_modes, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(features_flat)

    centers = kmeans.cluster_centers_.reshape(args.num_modes, args.T_f, 2)
    anchors = centers.copy()
    anchors[:, :, 0] /= LATERAL_SCALE  # 缩放回去

    # 强制起点归零 (消除微小误差)
    anchors[:, 0, :] = 0.0

    root_path = Path(__file__).resolve().parent.parent
    save_dir = root_path / 'dataset'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / 'anchors_ngsim.npy')

    np.save(save_path, anchors)
    print(f"Anchors saved to {save_path}. Shape: {anchors.shape}")

    vis_path = str(save_dir / 'anchors_vis_aligned.png')
    visualize_anchors(anchors, vis_path)


if __name__ == "__main__":
    main()

