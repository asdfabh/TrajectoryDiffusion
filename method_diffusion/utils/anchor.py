import os
import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from tqdm import tqdm


def visualize_anchors(anchors, save_path):
    # anchors: [K, F*2]
    K = anchors.shape[0]
    anchors = anchors.reshape(K, -1, 2)  # [K, F, 2]

    plt.figure(figsize=(10, 10))
    for i in range(K):
        traj = anchors[i]
        plt.plot(traj[:, 0], traj[:, 1], label=f'Anchor {i}', linewidth=2, marker='o', markersize=3)
        # Plot start point
        plt.plot(traj[0, 0], traj[0, 1], 'ks', markersize=5)

    plt.title(f'Generated {K} Anchors')
    plt.xlabel('X (ft)')
    plt.ylabel('Y (ft)')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.xlim(-10, 10)
    plt.savefig(save_path)
    print(f"Anchors visualization saved to {save_path}")
    plt.close()


def generate_anchors():
    args = get_args_parser().parse_args()

    method_dir = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.dirname(method_dir)

    if args.dataset == 'ngsim':
        trSet_path = os.path.join(root_path, '/mnt/datasets/ngsimdata/TrainSet.mat')
    elif args.dataset == 'highd':
        trSet_path = os.path.join(root_path, 'data/highDdata/TrainSet.mat')
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    print(f"Loading dataset from {trSet_path}...")

    d_s = 2
    t_h = (args.T - 1) * d_s
    t_f = args.T_f * d_s

    dataset = NgsimDataset(trSet_path, t_h=t_h, t_f=t_f, d_s=d_s)

    # Collect all future trajectories
    all_futures = []
    print("Collecting trajectories...")

    for i in tqdm(range(len(dataset))):
        dsId = int(dataset.D[i, 0])
        vehId = int(dataset.D[i, 1])
        t = dataset.D[i, 2]
        fut = dataset.getFuture(vehId, t, dsId)

        #  Only use trajectories with full prediction length for clustering
        if len(fut) == args.T_f:
            all_futures.append(fut.reshape(-1))

    if len(all_futures) == 0:
        print("No valid trajectories found!")
        return

    all_futures = np.stack(all_futures).astype(np.float64)
    print(f"Collected {len(all_futures)} valid trajectories. Shape: {all_futures.shape}")

    # KMeans Clustering
    k = args.num_modes
    print(f"Running KMeans with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(all_futures)

    anchors = kmeans.cluster_centers_

    # Save anchors
    save_path = os.path.join(os.path.dirname(__file__), './method_diffusion/dataset/anchors_ngsim.npy')
    torch.save({
        'means': torch.from_numpy(anchors).float()
    }, save_path)
    print(f"Anchors (KMeans) saved to {save_path}")

    # Visualize
    vis_path = os.path.join(os.path.dirname(__file__), './method_diffusion/dataset/anchors_vis.png')
    visualize_anchors(anchors, vis_path)


if __name__ == "__main__":
    generate_anchors()
