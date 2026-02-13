import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from method_diffusion.dataset.ngsim_dataset import NgsimDataset


def calculate_fast_stats(mat_file_path, num_workers=8, batch_size=1024):
    print(f"Loading dataset from: {mat_file_path}")
    print(f"Using {num_workers} workers for parallel processing...")

    dataset = NgsimDataset(mat_file_path, t_h=30, t_f=50, d_s=2)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True  # 加速数据从 CPU 到 Tensor 的传输
    )

    # Position (X, Y)
    pos_sum = torch.zeros(2, dtype=torch.float64)
    pos_sq_sum = torch.zeros(2, dtype=torch.float64)
    pos_count = 0

    # VA (Velocity, Acceleration)
    va_sum = torch.zeros(2, dtype=torch.float64)
    va_sq_sum = torch.zeros(2, dtype=torch.float64)
    va_count = 0

    # Lane and Class
    lane_sum = torch.zeros(1, dtype=torch.float64)
    lane_sq_sum = torch.zeros(1, dtype=torch.float64)
    lane_count = 0
    class_sum = torch.zeros(1, dtype=torch.float64)
    class_sq_sum = torch.zeros(1, dtype=torch.float64)
    class_count = 0

    print("Iterating through batches...")

    # 使用 tqdm 显示进度
    for batch in tqdm(loader):
        # 从 batch 中提取数据 (这些已经是 Tensor 了)
        # hist: [B, T, 2], fut: [B, T, 2]
        # nbrs: [Total_Nbrs_In_Batch, T, 2] <- collate_fn 已经剔除了无效邻居，只保留了存在的
        hist = batch['hist']
        fut = batch['fut']
        nbrs = batch['nbrs']

        va = batch['va']
        nbrs_va = batch['nbrs_va']
        lane = batch['lane']
        nbrs_lane = batch['nbrs_lane']
        cclass = batch['cclass']
        nbrs_class = batch['nbrs_class']

        current_pos = torch.cat([
            hist.reshape(-1, 2),
            fut.reshape(-1, 2),
            nbrs.reshape(-1, 2)
        ], dim=0).double()  # 转为 double 精度计算

        pos_sum += current_pos.sum(dim=0)
        pos_sq_sum += (current_pos ** 2).sum(dim=0)
        pos_count += current_pos.shape[0]

        current_va = torch.cat([
            va.reshape(-1, 2),
            nbrs_va.reshape(-1, 2)
        ], dim=0).double()

        va_sum += current_va.sum(dim=0)
        va_sq_sum += (current_va ** 2).sum(dim=0)
        va_count += current_va.shape[0]

        current_lane = torch.cat([
            lane.reshape(-1, 1),
            nbrs_lane.reshape(-1, 1)
        ], dim=0).double()
        lane_sum += current_lane.sum(dim=0)
        lane_sq_sum += (current_lane ** 2).sum(dim=0)
        lane_count += current_lane.shape[0]

        current_class = torch.cat([
            cclass.reshape(-1, 1),
            nbrs_class.reshape(-1, 1)
        ], dim=0).double()
        class_sum += current_class.sum(dim=0)
        class_sq_sum += (current_class ** 2).sum(dim=0)
        class_count += current_class.shape[0]

    # E[X]
    pos_mean = pos_sum / pos_count
    va_mean = va_sum / va_count

    # Std = sqrt( E[X^2] - (E[X])^2 )
    pos_std = torch.sqrt(pos_sq_sum / pos_count - pos_mean ** 2)
    va_std = torch.sqrt(va_sq_sum / va_count - va_mean ** 2)
    lane_mean = lane_sum / lane_count
    lane_std = torch.sqrt(lane_sq_sum / lane_count - lane_mean ** 2)
    class_mean = class_sum / class_count
    class_std = torch.sqrt(class_sq_sum / class_count - class_mean ** 2)

    # 5. 打印结果
    print("\n" + "=" * 50)
    print("   FAST CALCULATED STATISTICS   ")
    print("=" * 50)
    print(f"Total Points Processed: {pos_count:,}")
    print("-" * 30)
    print(f"Position Mean: {pos_mean.tolist()}")
    print(f"Position Std:  {pos_std.tolist()}")
    print("-" * 30)
    print(f"VA Mean:       {va_mean.tolist()}")
    print(f"VA Std:        {va_std.tolist()}")
    print("-" * 30)
    print(f"Lane Mean:     {lane_mean.tolist()}")
    print(f"Lane Std:      {lane_std.tolist()}")
    print(f"Class Mean:    {class_mean.tolist()}")
    print(f"Class Std:     {class_std.tolist()}")
    print("=" * 50)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ngsim_stats.npz")
    np.savez(
        save_path,
        pos_mean=pos_mean.cpu().numpy().astype(np.float32),
        pos_std=pos_std.cpu().numpy().astype(np.float32),
        va_mean=va_mean.cpu().numpy().astype(np.float32),
        va_std=va_std.cpu().numpy().astype(np.float32),
        lane_mean=lane_mean.cpu().numpy().astype(np.float32),
        lane_std=lane_std.cpu().numpy().astype(np.float32),
        class_mean=class_mean.cpu().numpy().astype(np.float32),
        class_std=class_std.cpu().numpy().astype(np.float32),
    )
    print(f"\nSaved stats to: {save_path}")


if __name__ == "__main__":
    workers = max(1, os.cpu_count() - 2)

    data_root = '/mnt/datasets/ngsimdata'
    mat_file = os.path.join(data_root, 'TrainSet.mat')

    if os.path.exists(mat_file):
        calculate_fast_stats(mat_file, num_workers=workers)
    else:
        print(f"Error: File not found at {mat_file}")
