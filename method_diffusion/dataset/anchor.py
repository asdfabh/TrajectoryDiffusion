import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm
from method_diffusion.dataset.ngsim_dataset import NgsimDataset


def get_args():
    parser = argparse.ArgumentParser("Anchor tools")
    parser.add_argument("--dataset", default="ngsim", type=str, choices=["ngsim", "highd"])
    parser.add_argument("--data_root_ngsim", default="/mnt/datasets/ngsimdata", type=str)
    parser.add_argument("--data_root_highd", default="/mnt/datasets/highDdata", type=str)
    parser.add_argument("--hist_length", default=16, type=int)
    parser.add_argument("--pred_length", default=25, type=int)
    parser.add_argument("--anchor_k", default=12, type=int, help="anchor 个数，即 KMeans 聚类数")
    parser.add_argument("--anchor_kmeans_random_state", default=42, type=int, help="KMeans 随机种子")
    parser.add_argument("--anchor_kmeans_n_init", default=10, type=int, help="KMeans 重启次数")
    return parser.parse_args()


def build_anchor_dataset(args):
    if args.dataset == "ngsim":
        train_path = Path(args.data_root_ngsim) / "TrainSet.mat"
    else:
        train_path = Path(args.data_root_highd) / "TrainSet.mat"

    print(f"Loading dataset from {train_path}...")
    return NgsimDataset(
        str(train_path),
        t_h=(args.hist_length - 1) * 2,
        t_f=args.pred_length * 2,
        d_s=2,
    )


def get_save_paths(args):
    save_dir = Path(__file__).resolve().parent / "anchor"
    save_dir.mkdir(exist_ok=True)
    name = f"{args.dataset}_k{args.anchor_k}"
    anchor_path = save_dir / f"{name}.pt"
    vis_path = save_dir / f"{name}_vis.png"
    return anchor_path, vis_path


def visualize_anchors(anchor, vis_path):
    if isinstance(anchor, torch.Tensor):
        anchor = anchor.detach().cpu().numpy()

    anchor = np.asarray(anchor, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, traj in enumerate(anchor):
        ax.plot(traj[:, 1], traj[:, 0], label=f"Anchor {i}", linewidth=2, marker="o", markersize=3)
        ax.plot(traj[0, 1], traj[0, 0], "ks", markersize=5)

    ax.set_title(f"Generated {anchor.shape[0]} Anchors")
    ax.set_xlabel("Y (ft)")
    ax.set_ylabel("X (ft)")
    ax.grid(True)
    ax.set_aspect("auto")
    ax.set_ylim(-6, 6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(vis_path)
    print(f"Anchors visualization saved to {vis_path}")
    plt.close(fig)


def generate_anchors(args):
    dataset = build_anchor_dataset(args)
    all_futures = []

    print("Collecting trajectories...")
    for i in tqdm(range(len(dataset))):
        ds_id = int(dataset.D[i, 0])
        veh_id = int(dataset.D[i, 1])
        t = dataset.D[i, 2]
        fut = dataset.getFuture(veh_id, t, ds_id)
        if len(fut) == args.pred_length:
            all_futures.append(fut.reshape(-1))

    all_futures = np.stack(all_futures).astype(np.float64)

    print(f"Running KMeans with k={args.anchor_k}...")
    kmeans = KMeans(
        n_clusters=args.anchor_k,
        random_state=args.anchor_kmeans_random_state,
        n_init=args.anchor_kmeans_n_init,
    )
    kmeans.fit(all_futures)

    anchor = torch.from_numpy(kmeans.cluster_centers_).float().reshape(args.anchor_k, args.pred_length, 2)
    anchor_path, _ = get_save_paths(args)
    torch.save(anchor, anchor_path)
    print(f"Anchors saved to {anchor_path}")
    return anchor


def evaluate_anchors(args, anchor):
    if isinstance(anchor, torch.Tensor):
        anchor = anchor.detach().cpu().numpy()

    dataset = build_anchor_dataset(args)
    anchor = np.asarray(anchor, dtype=np.float32)

    total_de = np.zeros(args.pred_length, dtype=np.float64)
    total_coord_se = np.zeros(args.pred_length, dtype=np.float64)
    total_counts = np.zeros(args.pred_length, dtype=np.float64)

    print("Evaluating anchors...")
    for i in tqdm(range(len(dataset))):
        ds_id = int(dataset.D[i, 0])
        veh_id = int(dataset.D[i, 1])
        t = dataset.D[i, 2]
        fut = dataset.getFuture(veh_id, t, ds_id)
        if len(fut) != args.pred_length:
            continue

        fut = fut.astype(np.float32)
        diff = anchor - fut[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        dist = np.sqrt(dist_sq)
        ade_all = dist.mean(axis=1)
        best_idx = int(np.argmin(ade_all))
        best_dist = dist[best_idx]
        best_diff = diff[best_idx]
        best_dist_sq = np.sum(best_diff * best_diff, axis=-1)

        total_de += best_dist
        total_coord_se += best_dist_sq
        total_counts += 1.0

    counts = np.clip(total_counts, a_min=1.0, a_max=None)
    mean_ade = total_de.sum() / counts.sum()
    mean_fde = total_de[-1] / counts[-1]
    print(
        f"Anchor Eval | ADE: {mean_ade * 0.3048:.4f} m | "
        f"FDE: {mean_fde * 0.3048:.4f} m"
    )

    rmse_per_step_m = np.sqrt(total_coord_se / counts) * 0.3048
    time_pairs = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]
    rmse_items = []
    for label, idx in time_pairs:
        if idx < len(rmse_per_step_m):
            rmse_items.append(f"{label}: {rmse_per_step_m[idx]:.4f} m")
    print("Anchor Eval | RMSE | " + ", ".join(rmse_items))


if __name__ == "__main__":
    args = get_args()
    anchor = generate_anchors(args)
    _, vis_path = get_save_paths(args)
    visualize_anchors(anchor, vis_path)
    # anchor_path, _ = get_save_paths(args)
    # anchor = torch.load(anchor_path, map_location="cpu")
    evaluate_anchors(args, anchor)
