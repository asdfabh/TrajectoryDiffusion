import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import scipy.io as scp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


LATERAL_LABELS = {
    1: "lat_1",
    2: "lat_2",
    3: "lat_3",
}

LONGITUDINAL_LABELS = {
    1: "lon_1",
    2: "lon_2",
    3: "lon_3",
}


def parse_args():
    parser = argparse.ArgumentParser("统计 NGSIM / highD 数据集意图分布")
    parser.add_argument("--data_root_ngsim", default="/mnt/datasets/ngsimdata", type=str)
    parser.add_argument("--data_root_highd", default="/mnt/datasets/highDdata", type=str)
    parser.add_argument("--datasets", nargs="+", default=["ngsim", "highd"], choices=["ngsim", "highd"])
    parser.add_argument("--show_per_file", default=1, type=int)
    return parser.parse_args()


def resolve_dataset_root(dataset_name, args):
    if dataset_name == "ngsim":
        return Path(args.data_root_ngsim)
    if dataset_name == "highd":
        return Path(args.data_root_highd)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def list_mat_files(data_root):
    if not data_root.exists():
        return []
    return sorted([path for path in data_root.glob("*.mat") if path.is_file()])


def safe_ratio(count, total):
    return 0.0 if total <= 0 else float(count) / float(total)


def init_stats():
    return {
        "total": 0,
        "invalid": 0,
        "lateral": Counter(),
        "longitudinal": Counter(),
        "joint": Counter(),
    }


def merge_stats(target, source):
    target["total"] += source["total"]
    target["invalid"] += source["invalid"]
    target["lateral"].update(source["lateral"])
    target["longitudinal"].update(source["longitudinal"])
    target["joint"].update(source["joint"])


def collect_file_stats(mat_path):
    mat_data = scp.loadmat(str(mat_path))
    if "traj" not in mat_data:
        return None, f"'traj' not found"

    traj = mat_data["traj"]
    if traj.ndim != 2 or traj.shape[1] <= 10:
        return None, f"unexpected traj shape: {traj.shape}"

    stats = init_stats()
    lateral_values = traj[:, 9]
    longitudinal_values = traj[:, 10]

    for lat_raw, lon_raw in zip(lateral_values, longitudinal_values):
        lat = int(lat_raw)
        lon = int(lon_raw)
        if lat not in LATERAL_LABELS or lon not in LONGITUDINAL_LABELS:
            stats["invalid"] += 1
            continue

        stats["total"] += 1
        stats["lateral"][lat] += 1
        stats["longitudinal"][lon] += 1
        stats["joint"][(lat, lon)] += 1

    return stats, None


def print_header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_basic_info(name, stats, num_files):
    print(f"[{name}] 文件数: {num_files}")
    print(f"[{name}] 有效样本数: {stats['total']}")
    print(f"[{name}] 无效标签数: {stats['invalid']}")


def print_lateral_distribution(stats):
    total = stats["total"]
    print("-" * 100)
    print("横向意图分布")
    print("-" * 100)
    print(f"{'Label':<12} | {'Count':<12} | {'Frequency':<12}")
    print("-" * 100)
    for label_id in sorted(LATERAL_LABELS):
        count = stats["lateral"][label_id]
        print(f"{LATERAL_LABELS[label_id]:<12} | {count:<12d} | {safe_ratio(count, total):<12.6f}")


def print_longitudinal_distribution(stats):
    total = stats["total"]
    print("-" * 100)
    print("纵向意图分布")
    print("-" * 100)
    print(f"{'Label':<12} | {'Count':<12} | {'Frequency':<12}")
    print("-" * 100)
    for label_id in sorted(LONGITUDINAL_LABELS):
        count = stats["longitudinal"][label_id]
        print(f"{LONGITUDINAL_LABELS[label_id]:<12} | {count:<12d} | {safe_ratio(count, total):<12.6f}")


def print_joint_distribution(stats):
    total = stats["total"]
    print("-" * 100)
    print("横向 + 纵向联合意图分布 (共 9 类)")
    print("-" * 100)
    print(f"{'Joint Label':<24} | {'Count':<12} | {'Frequency':<12}")
    print("-" * 100)
    for lat_id in sorted(LATERAL_LABELS):
        for lon_id in sorted(LONGITUDINAL_LABELS):
            count = stats["joint"][(lat_id, lon_id)]
            joint_name = f"{LATERAL_LABELS[lat_id]} + {LONGITUDINAL_LABELS[lon_id]}"
            print(f"{joint_name:<24} | {count:<12d} | {safe_ratio(count, total):<12.6f}")


def print_dataset_report(name, stats, num_files):
    print_header(f"{name} 意图分布统计")
    print_basic_info(name, stats, num_files)
    print_lateral_distribution(stats)
    print_longitudinal_distribution(stats)
    print_joint_distribution(stats)


def main():
    args = parse_args()
    show_per_file = int(args.show_per_file) > 0

    print("说明: 本脚本按原始标签 1/2/3 统计横向与纵向意图分布，不对标签语义做额外假设。")
    print(f"横向标签映射: {LATERAL_LABELS}")
    print(f"纵向标签映射: {LONGITUDINAL_LABELS}")

    for dataset_name in args.datasets:
        data_root = resolve_dataset_root(dataset_name, args)
        mat_files = list_mat_files(data_root)
        if not mat_files:
            print_header(f"{dataset_name} 意图分布统计")
            print(f"[{dataset_name}] 未找到可用的 .mat 文件: {data_root}")
            continue

        dataset_stats = init_stats()
        skipped_files = []

        for mat_path in mat_files:
            file_stats, skip_reason = collect_file_stats(mat_path)
            if file_stats is None:
                skipped_files.append((mat_path.name, skip_reason))
                print(f"[{dataset_name}] 跳过文件: {mat_path.name} | 原因: {skip_reason}")
                continue
            merge_stats(dataset_stats, file_stats)

            if show_per_file:
                print_dataset_report(f"{dataset_name} / {mat_path.name}", file_stats, 1)

        print_dataset_report(dataset_name, dataset_stats, len(mat_files))
        if skipped_files:
            print("-" * 100)
            print(f"[{dataset_name}] 跳过文件汇总 ({len(skipped_files)} 个)")
            for file_name, reason in skipped_files:
                print(f"- {file_name}: {reason}")


if __name__ == "__main__":
    main()
