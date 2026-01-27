import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
from pathlib import Path
from method_diffusion.dataset.ngsim_dataset import NgsimDataset


def visualize_anchors(anchors, save_path):
    K = anchors.shape[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, K))

    for i in range(K):
        traj = anchors[i]  # [T, 2]
        # 画出轨迹，注意确认 0 是横向，1 是纵向
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.8, linewidth=2)
        plt.scatter(traj[-1, 0], traj[-1, 1], color=colors[i], marker='x', s=30)

    plt.title(f'K-Means Anchors in Physical Space (K={K})\nNormalized-Space Clustering')
    plt.xlabel('Lateral Position (ft)')
    plt.ylabel('Longitudinal Position (ft)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')  # 保持比例，能一眼看出变道弧度
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_ngsim_anchors(mat_file, num_clusters=64, t_f=50, d_s=2):
    dataset = NgsimDataset(mat_file, t_f=t_f, d_s=d_s)

    # 与训练逻辑完全一致的归一化参数
    pos_mean = np.array([0.0, 0.0])
    pos_std = np.array([8.0, 120.0])

    print(f"正在从 {mat_file} 提取并归一化轨迹数据...")
    all_futures_norm = []

    # 步长设为 5 (每 0.5s 取一帧)，足以覆盖全量特征且速度极快
    for i in tqdm(range(0, len(dataset), 5)):
        dsId, vehId, t = int(dataset.D[i, 0]), int(dataset.D[i, 1]), dataset.D[i, 2]
        fut = dataset.getFuture(vehId, t, dsId)

        if len(fut) == (t_f // d_s):
            # 归一化并 Flatten
            fut_norm = (fut - pos_mean) / pos_std
            fut_norm = np.clip(fut_norm, -5.0, 5.0)
            all_futures_norm.append(fut_norm.flatten())

    all_futures_norm = np.array(all_futures_norm)

    # 这里的索引 -2 对应最后一个点的 X (横向)
    lat_displacements = np.abs(all_futures_norm[:, -2])
    maneuver_idx = np.where(lat_displacements > 0.1)[0]
    straight_idx = np.where(lat_displacements <= 0.1)[0]

    # 平衡变道与直行样本
    np.random.shuffle(straight_idx)
    num_samples = len(maneuver_idx)
    selected_idx = np.concatenate([maneuver_idx, straight_idx[:num_samples]])
    final_train_data = all_futures_norm[selected_idx]

    print(f"聚类平衡完成: 变道 {len(maneuver_idx)} 条, 直行抽样 {num_samples} 条")

    print(f"正在归一化空间执行 K-Means (K={num_clusters})...")
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(final_train_data)

    # 将聚类中心还原回物理空间
    anchors_norm = kmeans.cluster_centers_.reshape(num_clusters, -1, 2)
    anchors_phys = (anchors_norm * pos_std) + pos_mean

    return anchors_phys


if __name__ == "__main__":
    # 路径确保存在
    root_path = Path(__file__).resolve().parent.parent / 'dataset'
    root_path.mkdir(parents=True, exist_ok=True)

    mat_path = '/mnt/datasets/ngsimdata/TrainSet.mat'
    K = 10  # 建议 64 以覆盖更多变道细节

    anchors = generate_ngsim_anchors(mat_path, num_clusters=K)

    # 保存结果
    save_npy_path = root_path / f'anchors_ngsim.npy'
    np.save(save_npy_path, anchors)
    print(f"成功！Anchor 保存至: {save_npy_path}")

    # 可视化
    vis_path = root_path / f'anchors_vis_k{K}.png'
    visualize_anchors(anchors, vis_path)