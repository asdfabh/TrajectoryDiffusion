import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset


class RunningStats:
    """流式累计均值与标准差统计量。"""

    def __init__(self, dim):
        self.value_sum = torch.zeros(dim, dtype=torch.float64)
        self.value_sq_sum = torch.zeros(dim, dtype=torch.float64)
        self.count = 0

    def update(self, values):
        if values is None or values.numel() == 0:
            return
        values = values.reshape(-1, values.shape[-1]).double()
        self.value_sum += values.sum(dim=0)
        self.value_sq_sum += (values ** 2).sum(dim=0)
        self.count += int(values.shape[0])

    def summary(self):
        if self.count == 0:
            dim = self.value_sum.shape[0]
            zeros = torch.zeros(dim, dtype=torch.float64)
            return zeros, zeros
        mean = self.value_sum / self.count
        var = (self.value_sq_sum / self.count) - mean ** 2
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var)
        return mean, std


def parse_args():
    parser = argparse.ArgumentParser("Calculate normalization stats for NGSIM hist branch")
    parser.add_argument("--data_root", default="/mnt/datasets/ngsimdata", type=str)
    parser.add_argument("--split", default="TrainSet.mat", type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=max(1, (os.cpu_count() or 4) - 2), type=int)
    parser.add_argument("--sample_ratio", default=1.0, type=float)
    parser.add_argument("--log_every_batches", default=1000, type=int)
    parser.add_argument("--t_h", default=30, type=int)
    parser.add_argument("--d_s", default=2, type=int)
    return parser.parse_args()


def build_hist_loader(args):
    mat_path = Path(args.data_root) / args.split
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    dataset = NgsimHistDataset(str(mat_path), t_h=args.t_h, d_s=args.d_s)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )
    return dataset, loader, mat_path


def collect_valid_hist_rows(batch_tensor, sample_valid):
    """按有效样本掩码提取整条 hist 序列，避免误删合法的零值相对坐标。"""
    if batch_tensor.numel() == 0 or sample_valid.numel() == 0:
        return batch_tensor.reshape(0, batch_tensor.shape[-1])
    valid_tensor = batch_tensor[sample_valid]
    return valid_tensor.reshape(-1, valid_tensor.shape[-1])


def print_running_stats(batch_idx, total_batches, pos_stats, va_stats):
    pos_mean, pos_std = pos_stats.summary()
    va_mean, va_std = va_stats.summary()
    print("\n" + "=" * 60)
    print(f"[HistStats] Batch {batch_idx}/{total_batches}")
    print(f"[HistStats] Position Count: {pos_stats.count:,}")
    print(f"[HistStats] VA Count:       {va_stats.count:,}")
    print(f"[HistStats] Position Mean: {pos_mean.tolist()}")
    print(f"[HistStats] Position Std:  {pos_std.tolist()}")
    print(f"[HistStats] VA Mean:       {va_mean.tolist()}")
    print(f"[HistStats] VA Std:        {va_std.tolist()}")
    print("=" * 60)


def print_final_stats(pos_stats, va_stats):
    pos_mean, pos_std = pos_stats.summary()
    va_mean, va_std = va_stats.summary()

    print("\n" + "=" * 60)
    print("Hist-only normalization statistics")
    print("=" * 60)
    print(f"Position Mean: {pos_mean.tolist()}")
    print(f"Position Std:  {pos_std.tolist()}")
    print(f"VA Mean:       {va_mean.tolist()}")
    print(f"VA Std:        {va_std.tolist()}")
    print("-" * 60)
    print("[Copy & Paste] Replace these lines in `hist_model.py` -> `__init__`:")
    print(f"self.register_buffer(\"pos_mean\", torch.tensor([{pos_mean[0]:.4f}, {pos_mean[1]:.4f}]).float(), persistent=False)")
    print(f"self.register_buffer(\"pos_std\", torch.tensor([{pos_std[0]:.4f}, {pos_std[1]:.4f}]).float(), persistent=False)")
    print(f"self.register_buffer(\"va_mean\", torch.tensor([{va_mean[0]:.4f}, {va_mean[1]:.4f}]).float(), persistent=False)")
    print(f"self.register_buffer(\"va_std\", torch.tensor([{va_std[0]:.4f}, {va_std[1]:.4f}]).float(), persistent=False)")
    print("=" * 60)


def calculate_hist_stats(args):
    dataset, loader, mat_path = build_hist_loader(args)
    total_batches = len(loader)
    ratio = max(0.0, min(1.0, float(args.sample_ratio)))
    target_batches = total_batches if ratio <= 0.0 or ratio >= 1.0 else max(1, int(math.ceil(total_batches * ratio)))

    print(f"[HistStats] Dataset: {mat_path}")
    print(f"[HistStats] Samples: {len(dataset):,}")
    print(f"[HistStats] Batch size: {args.batch_size}")
    print(f"[HistStats] Num workers: {args.num_workers}")
    print(f"[HistStats] Sample ratio: {ratio:.4f}")
    print(f"[HistStats] Target batches: {target_batches}/{total_batches}")

    pos_stats = RunningStats(dim=2)
    va_stats = RunningStats(dim=2)

    pbar = tqdm(enumerate(loader, start=1), total=target_batches, desc="Hist Stats", ncols=120)
    for batch_idx, batch in pbar:
        if batch_idx > target_batches:
            break

        sample_valid = batch["sample_valid"].bool()
        hist = collect_valid_hist_rows(batch["hist"], sample_valid)
        va = collect_valid_hist_rows(batch["va"], sample_valid)

        pos_stats.update(hist)
        va_stats.update(va)

        pos_mean, pos_std = pos_stats.summary()
        va_mean, va_std = va_stats.summary()
        pbar.set_postfix({
            "pos_mean_x": f"{pos_mean[0].item():.4f}",
            "pos_std_y": f"{pos_std[1].item():.4f}",
            "va_mean_v": f"{va_mean[0].item():.4f}",
            "va_std_a": f"{va_std[1].item():.4f}",
        })

        if batch_idx % max(1, int(args.log_every_batches)) == 0:
            print_running_stats(batch_idx, target_batches, pos_stats, va_stats)

    print_running_stats(min(target_batches, total_batches), target_batches, pos_stats, va_stats)
    print_final_stats(pos_stats, va_stats)


def main():
    args = parse_args()
    calculate_hist_stats(args)


if __name__ == "__main__":
    main()
