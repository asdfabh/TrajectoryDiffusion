import argparse
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.dataset.ngsim_dataset import NgsimDataset
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
    parser = argparse.ArgumentParser("Calculate normalization stats for hist/future branches")
    parser.add_argument("--mode", default="hist_nbrs", type=str, choices=["hist", "hist_nbrs", "future"])
    parser.add_argument("--data_root", default="/mnt/datasets/highDdata", type=str, choices=["ngsimdata", "highDdata"])
    parser.add_argument("--split", default="TrainSet.mat", type=str)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=max(1, (os.cpu_count() or 4) - 2), type=int)
    parser.add_argument("--sample_ratio", default=1.0, type=float)
    parser.add_argument("--log_every_batches", default=1000, type=int)
    parser.add_argument("--t_h", default=30, type=int)
    parser.add_argument("--t_f", default=50, type=int)
    parser.add_argument("--d_s", default=2, type=int)
    return parser.parse_args()


def resolve_mat_path(args):
    mat_path = Path(args.data_root) / args.split
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")
    return mat_path


def build_hist_loader(args):
    mat_path = resolve_mat_path(args)
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


def build_full_loader(args):
    mat_path = resolve_mat_path(args)
    dataset = NgsimDataset(str(mat_path), t_h=args.t_h, t_f=args.t_f, d_s=args.d_s)
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


def collect_hist_valid_rows(batch_tensor, sample_valid):
    """按有效 ego 样本掩码提取整条 hist 序列。"""
    if batch_tensor.numel() == 0 or sample_valid.numel() == 0:
        return batch_tensor.reshape(0, batch_tensor.shape[-1])
    valid_tensor = batch_tensor[sample_valid]
    return valid_tensor.reshape(-1, valid_tensor.shape[-1])


def collect_future_relative_rows(hist, fut, op_mask):
    """按 fut 训练分支逻辑提取 Ego future 的有效帧间相对位移。"""
    if hist.numel() == 0 or fut.numel() == 0 or op_mask.numel() == 0:
        return fut.reshape(0, fut.shape[-1])
    anchor_phys = hist[:, -1:, :2]
    future_phys = fut[..., :2]
    shifted_future_phys = torch.cat([anchor_phys, future_phys[:, :-1, :]], dim=1)
    future_rel = future_phys - shifted_future_phys
    valid_mask = op_mask[..., 0] > 0.5
    return future_rel[valid_mask]


def collect_full_loader_valid_hist_rows(batch_tensor, hist, va, lane, cclass):
    """过滤 full loader 中被全零 padding 的无效 ego 样本。"""
    if batch_tensor.numel() == 0:
        return batch_tensor.reshape(0, batch_tensor.shape[-1])
    sample_valid = (
        hist.abs().sum(dim=(1, 2))
        + va.abs().sum(dim=(1, 2))
        + lane.abs().sum(dim=(1, 2))
        + cclass.abs().sum(dim=(1, 2))
    ) > 0
    if sample_valid.numel() == 0:
        return batch_tensor.reshape(0, batch_tensor.shape[-1])
    return batch_tensor[sample_valid].reshape(-1, batch_tensor.shape[-1])


def print_running_stats(batch_idx, total_batches, title, stats_items):
    print("\n" + "=" * 60)
    print(f"[{title}] Batch {batch_idx}/{total_batches}")
    for name, stats in stats_items:
        mean, std = stats.summary()
        print(f"[{title}] {name} Count: {stats.count:,}")
        print(f"[{title}] {name} Mean:  {mean.tolist()}")
        print(f"[{title}] {name} Std:   {std.tolist()}")
    print("=" * 60)


def print_final_stats(title, stats_items):
    print("\n" + "=" * 60)
    print(f"{title} normalization statistics")
    print("=" * 60)
    for name, stats in stats_items:
        mean, std = stats.summary()
        print(f"{name} Mean: {mean.tolist()}")
        print(f"{name} Std:  {std.tolist()}")
    print("-" * 60)
    print("[Copy & Paste]")
    for name, stats in stats_items:
        mean, std = stats.summary()
        key = name.lower()
        print(f"{key}_mean = {mean.tolist()}")
        print(f"{key}_std = {std.tolist()}")
    print("=" * 60)


def print_run_header(title, dataset, loader, mat_path, args):
    total_batches = len(loader)
    ratio = max(0.0, min(1.0, float(args.sample_ratio)))
    target_batches = total_batches if ratio <= 0.0 or ratio >= 1.0 else max(1, int(math.ceil(total_batches * ratio)))
    print(f"[{title}] Dataset Path: {mat_path}")
    print(f"[{title}] Mode: {args.mode}")
    print(f"[{title}] Samples: {len(dataset):,}")
    print(f"[{title}] Batch size: {args.batch_size}")
    print(f"[{title}] Num workers: {args.num_workers}")
    print(f"[{title}] Sample ratio: {ratio:.4f}")
    print(f"[{title}] Target batches: {target_batches}/{total_batches}")
    return target_batches, total_batches


def calculate_hist_stats(args):
    dataset, loader, mat_path = build_hist_loader(args)
    title = "HistStats"
    target_batches, total_batches = print_run_header(title, dataset, loader, mat_path, args)

    pos_stats = RunningStats(dim=2)
    va_stats = RunningStats(dim=2)
    lane_stats = RunningStats(dim=1)
    class_stats = RunningStats(dim=1)
    stats_items = [("POS", pos_stats), ("VA", va_stats), ("LANE", lane_stats), ("CLASS", class_stats)]

    pbar = tqdm(enumerate(loader, start=1), total=target_batches, desc=title, ncols=120)
    for batch_idx, batch in pbar:
        if batch_idx > target_batches:
            break

        sample_valid = batch["sample_valid"].bool()
        pos_stats.update(collect_hist_valid_rows(batch["hist"], sample_valid))
        va_stats.update(collect_hist_valid_rows(batch["va"], sample_valid))
        lane_stats.update(collect_hist_valid_rows(batch["lane"], sample_valid))
        class_stats.update(collect_hist_valid_rows(batch["cclass"], sample_valid))

        pos_mean, pos_std = pos_stats.summary()
        va_mean, va_std = va_stats.summary()
        pbar.set_postfix({
            "pos_mean_x": f"{pos_mean[0].item():.4f}",
            "pos_std_y": f"{pos_std[1].item():.4f}",
            "va_mean_v": f"{va_mean[0].item():.4f}",
            "va_std_a": f"{va_std[1].item():.4f}",
        })

        if batch_idx % max(1, int(args.log_every_batches)) == 0:
            print_running_stats(batch_idx, target_batches, title, stats_items)

    print_running_stats(min(target_batches, total_batches), target_batches, title, stats_items)
    print_final_stats("Hist Ego", stats_items)


def calculate_hist_nbrs_stats(args):
    dataset, loader, mat_path = build_full_loader(args)
    title = "HistNbrsStats"
    target_batches, total_batches = print_run_header(title, dataset, loader, mat_path, args)

    pos_stats = RunningStats(dim=2)
    va_stats = RunningStats(dim=2)
    lane_stats = RunningStats(dim=1)
    class_stats = RunningStats(dim=1)
    stats_items = [("POS", pos_stats), ("VA", va_stats), ("LANE", lane_stats), ("CLASS", class_stats)]

    pbar = tqdm(enumerate(loader, start=1), total=target_batches, desc=title, ncols=120)
    for batch_idx, batch in pbar:
        if batch_idx > target_batches:
            break

        ego_hist = collect_full_loader_valid_hist_rows(
            batch["hist"], batch["hist"], batch["va"], batch["lane"], batch["cclass"]
        )
        ego_va = collect_full_loader_valid_hist_rows(
            batch["va"], batch["hist"], batch["va"], batch["lane"], batch["cclass"]
        )
        ego_lane = collect_full_loader_valid_hist_rows(
            batch["lane"], batch["hist"], batch["va"], batch["lane"], batch["cclass"]
        )
        ego_class = collect_full_loader_valid_hist_rows(
            batch["cclass"], batch["hist"], batch["va"], batch["lane"], batch["cclass"]
        )

        hist_all = torch.cat([ego_hist, batch["nbrs"].reshape(-1, 2)], dim=0)
        va_all = torch.cat([ego_va, batch["nbrs_va"].reshape(-1, 2)], dim=0)
        lane_all = torch.cat([ego_lane, batch["nbrs_lane"].reshape(-1, 1)], dim=0)
        class_all = torch.cat([ego_class, batch["nbrs_class"].reshape(-1, 1)], dim=0)

        pos_stats.update(hist_all)
        va_stats.update(va_all)
        lane_stats.update(lane_all)
        class_stats.update(class_all)

        pos_mean, pos_std = pos_stats.summary()
        va_mean, va_std = va_stats.summary()
        pbar.set_postfix({
            "pos_mean_x": f"{pos_mean[0].item():.4f}",
            "pos_std_y": f"{pos_std[1].item():.4f}",
            "va_mean_v": f"{va_mean[0].item():.4f}",
            "va_std_a": f"{va_std[1].item():.4f}",
        })

        if batch_idx % max(1, int(args.log_every_batches)) == 0:
            print_running_stats(batch_idx, target_batches, title, stats_items)

    print_running_stats(min(target_batches, total_batches), target_batches, title, stats_items)
    print_final_stats("Hist Plus Nbrs", stats_items)


def calculate_future_stats(args):
    dataset, loader, mat_path = build_full_loader(args)
    title = "FutureStats"
    target_batches, total_batches = print_run_header(title, dataset, loader, mat_path, args)

    vel_stats = RunningStats(dim=2)
    stats_items = [("VEL", vel_stats)]

    pbar = tqdm(enumerate(loader, start=1), total=target_batches, desc=title, ncols=120)
    for batch_idx, batch in pbar:
        if batch_idx > target_batches:
            break

        fut_rel = collect_future_relative_rows(batch["hist"], batch["fut"], batch["op_mask"])
        vel_stats.update(fut_rel)

        vel_mean, vel_std = vel_stats.summary()
        pbar.set_postfix({
            "vel_mean_x": f"{vel_mean[0].item():.4f}",
            "vel_std_y": f"{vel_std[1].item():.4f}",
        })

        if batch_idx % max(1, int(args.log_every_batches)) == 0:
            print_running_stats(batch_idx, target_batches, title, stats_items)

    print_running_stats(min(target_batches, total_batches), target_batches, title, stats_items)
    print_final_stats("Future Ego Relative", stats_items)


def main():
    args = parse_args()
    if args.mode == "hist":
        calculate_hist_stats(args)
        return
    if args.mode == "hist_nbrs":
        calculate_hist_nbrs_stats(args)
        return
    if args.mode == "future":
        calculate_future_stats(args)
        return
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
