import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.visualization import visualize_batch_trajectories

class OptimizedDTW(nn.Module):
    def __init__(self, gamma=0.01):
        super(OptimizedDTW, self).__init__()
        self.gamma = gamma

    def forward(self, anchors, ground_truth):
        """
        anchors: [K, T, 2]
        ground_truth: [B, T, 2]
        """
        B, T, D = ground_truth.shape
        K = anchors.shape[0]

        dist_mat = torch.cdist(
            ground_truth.unsqueeze(1).repeat(1, K, 1, 1).view(-1, T, D),
            anchors.unsqueeze(0).repeat(B, 1, 1, 1).view(-1, T, D),
            p=2
        ) ** 2  # [B*K, T, T]

        BK = B * K
        r = torch.full((BK, T + 1), 1e8, device=ground_truth.device)
        r[:, 0] = 0

        dist_mat = dist_mat.permute(1, 2, 0)  # [T_gt, T_anchor, BK]

        R = torch.full((T + 1, T + 1, BK), 1e8, device=ground_truth.device)
        R[0, 0] = 0

        for i in range(1, T + 1):
            for j in range(1, T + 1):
                # 获取三个相邻状态
                prec = torch.stack([R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]], dim=0)  # [3, BK]
                # Soft-min 操作
                softmin = -self.gamma * torch.logsumexp(-prec / self.gamma, dim=0)
                R[i, j] = dist_mat[i - 1, j - 1] + softmin

        return R[T, T].view(B, K)

def generate_ngsim_anchors(mat_file, num_clusters=64, t_f=50, d_s=2):
    dataset = NgsimDataset(mat_file, t_f=t_f, d_s=d_s)
    pos_mean, pos_std = np.array([0.0, 0.0]), np.array([8.0, 120.0])

    all_futures_norm = []
    print("正在提取轨迹数据用于聚类...")
    for i in tqdm(range(0, len(dataset), 5)):
        dsId, vehId, t = int(dataset.D[i, 0]), int(dataset.D[i, 1]), dataset.D[i, 2]
        fut = dataset.getFuture(vehId, t, dsId)
        if len(fut) == (t_f // d_s):
            fut_norm = (fut - pos_mean) / pos_std
            all_futures_norm.append(np.clip(fut_norm, -5.0, 5.0).flatten())

    all_futures_norm = np.array(all_futures_norm)
    lat_dis = np.abs(all_futures_norm[:, -2])
    man_idx = np.where(lat_dis > 0.1)[0]
    str_idx = np.where(lat_dis <= 0.1)[0]
    np.random.shuffle(str_idx)
    selected_idx = np.concatenate([man_idx, str_idx[:len(man_idx)]])

    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(all_futures_norm[selected_idx])

    anchors_norm = kmeans.cluster_centers_.reshape(num_clusters, -1, 2)
    return (anchors_norm * pos_std) + pos_mean


def compute_and_save_best_indices(mat_file, anchor_file, save_path, device='cuda'):
    """
    1. 使用全量数组存储，彻底解决内存崩溃问题
    2. 增加 Batch 显存管理，防止 GPU 崩溃
    """
    dataset = NgsimDataset(mat_file)
    anchors_phys = np.load(anchor_file)
    anchors = torch.from_numpy(anchors_phys).float().to(device)

    # 归一化参数
    pos_mean = torch.tensor([0.0, 0.0], device=device)
    pos_std = torch.tensor([8.0, 120.0], device=device)
    anchors_norm = (anchors - pos_mean) / pos_std

    dtw_tool = OptimizedDTW(gamma=0.01).to(device)

    # 使用 NumPy 数组预分配内存，不使用字典！
    total_samples = len(dataset)
    best_indices = np.zeros(total_samples, dtype=np.int32)

    print(f"开始离线计算 DTW 最佳索引 (样本数: {total_samples})...")

    batch_size = 256  # 适当调大，如果显存小则调为 128
    for i in tqdm(range(0, total_samples, batch_size)):
        end_i = min(i + batch_size, total_samples)

        current_batch_fut = []
        valid_mask = []

        for j in range(i, end_i):
            dsId, vehId, t = int(dataset.D[j, 0]), int(dataset.D[j, 1]), dataset.D[j, 2]
            fut = dataset.getFuture(vehId, t, dsId)

            if len(fut) == anchors.shape[1]:
                fut_norm = (torch.from_numpy(fut).float() - pos_mean.cpu()) / pos_std.cpu()
                current_batch_fut.append(fut_norm)
                valid_mask.append(True)
            else:
                # 异常处理：轨迹长度不足
                current_batch_fut.append(torch.zeros((anchors.shape[1], 2)))
                valid_mask.append(False)

        if not current_batch_fut: continue

        futs_tensor = torch.stack(current_batch_fut).to(device)

        with torch.no_grad():
            # 计算相似度
            scores = dtw_tool(anchors_norm, futs_tensor)  # [B, K]
            batch_indices = torch.argmin(scores, dim=1).cpu().numpy()

            # 处理无效轨迹（如果有）
            batch_indices[~np.array(valid_mask)] = 0

            # 存入顺序数组
            best_indices[i:end_i] = batch_indices

        # 关键：定期清理显存碎片，防止崩溃
        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    # 直接保存为数组，训练时直接用 index 读取
    np.save(save_path, best_indices)
    print(f"成功！索引数组已保存至: {save_path}")


def visualize_anchors(anchors, save_path):
    K = anchors.shape[0]
    plt.figure(figsize=(10, 8))
    colors = plt.cm.jet(np.linspace(0, 1, K))
    for i in range(K):
        traj = anchors[i]
        plt.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.7)
    plt.title(f'Anchors (K={K})')
    plt.savefig(save_path)
    plt.close()


def test_dtw_index_accuracy(mat_path, anchor_path, index_path, index_path2=None, num_samples=5):
    """
    单元测试：直接可视化检查离线计算的索引是否准确
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载数据
    # 确保加载时传入 index_file
    dataset = NgsimDataset(mat_path, index_file=index_path)
    if index_path2 is None:
        pass
    else:
        dataset2 = NgsimDataset(mat_path, index_file=index_path2)
    anchors_phys = torch.from_numpy(np.load(anchor_path)).float().to(device)  # [K, T, 2]

    # 2. 随机抽取样本进行测试
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"正在验证样本索引: {indices}")

    for i in indices:
        # 获取数据 (target_idx 是通过 idx 直接从 npy 读取的)
        hist, fut, nbrs, lat, lon, va, nva, lane, nlane, ref, ndis, ccl, ncl, nnum, target_idx = dataset[i]
        _, _, _, _, _, _, _, _, _, _, _, _, _, _, target_idx2 = dataset2[i] if index_path2 else (None,)*15

        # 转换为 Tensor 满足可视化函数要求
        # visualize_batch_trajectories 期望的 pred 维度通常是 [B, K, T, 2]
        hist_tensor = torch.from_numpy(hist).unsqueeze(0).to(device)
        fut_tensor = torch.from_numpy(fut).unsqueeze(0).to(device)
        # 将所有 Anchor 作为预测候选，显示 target_idx 选中的那条
        anchors_expanded = anchors_phys.unsqueeze(0).permute(0, 2, 1, 3)  # [1, T, K, 2]

        print(f"样本 {i}: 预计算得到的 Best Anchor Index 为: {target_idx}")

        # 3. 调用可视化 (查看选中的 Anchor 是否真的贴合 GT Future)
        # 注意：这里传入 pred 是为了让函数画出选中的那条 index
        visualize_batch_trajectories(
            hist=hist_tensor,
            future=fut_tensor,
            pred=anchors_expanded,  # 传入所有 anchors
            best_index=torch.tensor([target_idx]).to(device),
        )
        if index_path2 is None:
            continue
        else:
            visualize_batch_trajectories(
                hist=hist_tensor,
                future=fut_tensor,
                pred=anchors_expanded,  # 传入所有 anchors
                best_index=torch.tensor([target_idx2]).to(device),
            )


def compute_and_save_best_indices_mse(mat_file, anchor_file, save_path, device='cuda'):
    """
    使用归一化空间下的 MSE (Mean Squared Error) 计算最佳索引
    用于作为 DTW 的对比基准，排查计算准确性问题
    """
    dataset = NgsimDataset(mat_file)
    anchors_phys = np.load(anchor_file)
    anchors = torch.from_numpy(anchors_phys).float().to(device)  # [K, T, 2]

    # 归一化参数 (必须与模型完全一致)
    pos_mean = torch.tensor([0.0, 0.0], device=device)
    pos_std = torch.tensor([8.0, 120.0], device=device)
    anchors_norm = (anchors - pos_mean) / pos_std  # [K, T, 2]

    total_samples = len(dataset)
    best_indices = np.zeros(total_samples, dtype=np.int32)

    print(f"开始离线计算 MSE 最佳索引 (样本数: {total_samples})...")

    batch_size = 512  # MSE 计算极快，可以使用大 Batch
    for i in tqdm(range(0, total_samples, batch_size)):
        end_i = min(i + batch_size, total_samples)
        current_batch_fut = []
        valid_mask = []

        for j in range(i, end_i):
            dsId, vehId, t = int(dataset.D[j, 0]), int(dataset.D[j, 1]), dataset.D[j, 2]
            fut = dataset.getFuture(vehId, t, dsId)

            if len(fut) == anchors.shape[1]:
                fut_norm = (torch.from_numpy(fut).float() - pos_mean.cpu()) / pos_std.cpu()
                current_batch_fut.append(fut_norm)
                valid_mask.append(True)
            else:
                current_batch_fut.append(torch.zeros((anchors.shape[1], 2)))
                valid_mask.append(False)

        if not current_batch_fut: continue
        futs_tensor = torch.stack(current_batch_fut).to(device)  # [B, T, 2]

        with torch.no_grad():
            # 计算 MSE: (GT - Anchor)^2 的均值
            # [B, 1, T, 2] - [1, K, T, 2] -> [B, K, T, 2]
            diff = futs_tensor.unsqueeze(1) - anchors_norm.unsqueeze(0)
            mse = (diff ** 2).sum(dim=-1).mean(dim=-1)  # [B, K]

            # 选取最小 MSE 的索引
            batch_indices = torch.argmin(mse, dim=1).cpu().numpy()

            # 处理无效轨迹
            batch_indices[~np.array(valid_mask)] = 0
            best_indices[i:end_i] = batch_indices

    np.save(save_path, best_indices)
    print(f"MSE 索引计算完成！保存至: {save_path}")

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    root_path = Path(__file__).resolve().parent.parent / 'dataset'
    root_path.mkdir(parents=True, exist_ok=True)

    mat_path = '/mnt/datasets/ngsimdata/TrainSet.mat'
    anchor_path = root_path / 'anchors_ngsim.npy'
    index_path_dtw = root_path / 'best_anchor_indices_ngsim_dtw.npy'
    index_path_mse = root_path / 'best_anchor_indices_ngsim_mse.npy'

    # 步骤 1: 生成锚点 (如果已存在可跳过)
    if not anchor_path.exists():
        anchors = generate_ngsim_anchors(mat_path, num_clusters=args.num_modes)
        np.save(anchor_path, anchors)
        visualize_anchors(anchors, root_path / 'anchors_vis.png')

    # 步骤 2: 计算索引 (通过 OptimizedDTW)
    # compute_and_save_best_indices(mat_path, anchor_path, index_path_dtw)
    # compute_and_save_best_indices_mse(mat_path, anchor_path, index_path_mse)
    test_dtw_index_accuracy(mat_path, anchor_path, index_path_dtw, index_path_mse, num_samples=20)