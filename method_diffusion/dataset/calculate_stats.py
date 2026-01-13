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

    # E[X]
    pos_mean = pos_sum / pos_count
    va_mean = va_sum / va_count

    # Std = sqrt( E[X^2] - (E[X])^2 )
    pos_std = torch.sqrt(pos_sq_sum / pos_count - pos_mean ** 2)
    va_std = torch.sqrt(va_sq_sum / va_count - va_mean ** 2)

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
    print("=" * 50)

    print("\n[Copy & Paste] Replace these lines in `fut_model.py` -> `__init__`:")
    print(
        f"self.register_buffer('pos_mean', torch.tensor([{pos_mean[0]:.4f}, {pos_mean[1]:.4f}]).float(), persistent=False)")
    print(
        f"self.register_buffer('pos_std', torch.tensor([{pos_std[0]:.4f}, {pos_std[1]:.4f}]).float(), persistent=False)")
    print(
        f"self.register_buffer('va_mean', torch.tensor([{va_mean[0]:.4f}, {va_mean[1]:.4f}]).float(), persistent=False)")
    print(f"self.register_buffer('va_std', torch.tensor([{va_std[0]:.4f}, {va_std[1]:.4f}]).float(), persistent=False)")


if __name__ == "__main__":
    workers = max(1, os.cpu_count() - 2)

    data_root = '/mnt/datasets/ngsimdata'
    mat_file = os.path.join(data_root, 'TrainSet.mat')

    if os.path.exists(mat_file):
        calculate_fast_stats(mat_file, num_workers=workers)
    else:
        print(f"Error: File not found at {mat_file}")