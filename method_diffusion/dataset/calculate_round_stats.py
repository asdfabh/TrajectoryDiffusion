"""
RounD 数据集归一化参数统计脚本。

仿照 calculate_stats.py 的轻量模式：用最小 Dataset 仅提取 future/hist/va，
不做邻居搜索、网格编码等重计算。RounD .mat 为 HDF5 v7.3 格式，需 h5py 加载。

用法:
    conda run -n tame python method_diffusion/dataset/calculate_round_stats.py \
        --mat_file /mnt/datasets/round/TrainSet.mat \
        --batch_size 1024
"""

import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from method_diffusion.dataset.build import get_raw_dt, get_time_params
from method_diffusion.dataset.future_features import build_xy_theta_v_from_positions
from method_diffusion.dataset.round_dataset import RoundHistDataset


# ---------------------------------------------------------------------------
# 轻量 Dataset：仅提取 future [x,y,θ,v]、hist [x,y]、va [lon_vel,lon_acc]
# 不做邻居搜索、不做网格编码、不做 lat/lon encoding
# ---------------------------------------------------------------------------

class RounDStatsDataset(Dataset):
    """
    极简 RounD 数据访问器，对齐 NgsimFutureDataset 的轻量模式。

    tracks 列: Frame(0), X(1), Y(2), Heading(3), LonVel(4), LatVel(5), LonAcc(6), ...
    输出 (fut, hist, va)，其中 fut 为 ego-frame [x, y, θ, v]。
    """

    def __init__(self, mat_file, t_h=50, t_f=100, d_s=5):
        self.t_h = int(t_h)
        self.t_f = int(t_f)
        self.d_s = int(d_s)
        self.raw_dt = get_raw_dt("round")
        self.max_hist = t_h // d_s + 1  # 11
        self.max_fut = t_f // d_s       # 20

        self.IDX_X = 1
        self.IDX_Y = 2
        self.IDX_HEADING = 3
        self.IDX_LON_VEL = 4
        self.IDX_LON_ACC = 6

        self.D, self.T = self._load(mat_file)
        self._track_cache = {}

    # ---- 加载 ----

    def _load(self, mat_file):
        try:
            import scipy.io as scp
            mat = scp.loadmat(mat_file)
            return mat["traj"], mat["tracks"]
        except (NotImplementedError, TypeError):
            return self._load_hdf5(mat_file)

    def _load_hdf5(self, mat_file):
        with h5py.File(mat_file, "r") as f:
            traj = f["traj"][:].transpose()
            track_refs = f["tracks"][:].transpose()
            tracks = np.empty(track_refs.shape, dtype=object)
            for ds_idx in range(track_refs.shape[0]):
                for veh_idx in range(track_refs.shape[1]):
                    try:
                        tdata = f[track_refs[ds_idx, veh_idx]][:]
                    except (KeyError, TypeError, ValueError):
                        tracks[ds_idx, veh_idx] = np.empty((0, 0), dtype=np.float32)
                        continue
                    if tdata.ndim < 2:
                        tracks[ds_idx, veh_idx] = np.empty((0, 0), dtype=np.float32)
                    else:
                        tracks[ds_idx, veh_idx] = tdata.transpose().astype(np.float32, copy=False)
        return traj, tracks

    # ---- 基础访问 ----

    def __len__(self):
        return len(self.D)

    def _get_track(self, ds_id, veh_id):
        key = (int(ds_id), int(veh_id))
        if key in self._track_cache:
            return self._track_cache[key]
        if veh_id <= 0 or self.T.shape[1] <= veh_id - 1:
            self._track_cache[key] = None
            return None
        track = self.T[ds_id - 1][veh_id - 1]
        if track.size == 0:
            self._track_cache[key] = None
            return None
        track = track.transpose()
        self._track_cache[key] = track
        return track

    def _find_frame(self, track, t):
        idx = np.where(track[:, 0] == t)[0]
        return int(idx.item()) if idx.size > 0 else None

    # ---- ego-frame 旋转 ----

    @staticmethod
    def _rotate_to_ego(positions, ref_pos, heading):
        if len(positions) == 0:
            return positions.astype(np.float32, copy=False)
        dx = positions[:, 0] - ref_pos[0]
        dy = positions[:, 1] - ref_pos[1]
        c, s = np.cos(heading), np.sin(heading)
        local_x = -dx * s + dy * c
        local_y = dx * c + dy * s
        return np.column_stack((local_x, local_y)).astype(np.float32, copy=False)

    # ---- 核心提取 ----

    def _extract(self, ds_id, veh_id, t):
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return None
        frame_idx = self._find_frame(track, t)
        if frame_idx is None:
            return None

        ego_pos = track[frame_idx, self.IDX_X:self.IDX_Y + 1].astype(np.float32)
        ego_heading = float(track[frame_idx, self.IDX_HEADING])

        # --- future: ego-frame [x, y, θ, v] ---
        fut_start = frame_idx + self.d_s
        fut_end = min(len(track), frame_idx + self.t_f + 1)
        fut_global = track[fut_start:fut_end:self.d_s, self.IDX_X:self.IDX_Y + 1].astype(np.float32)
        if len(fut_global) < self.max_fut:
            return None

        fut_xy = self._rotate_to_ego(fut_global, ego_pos, ego_heading)
        fut = build_xy_theta_v_from_positions(fut_xy, np.zeros(2, dtype=np.float32), self.d_s, track_dt=self.raw_dt)

        # --- hist: ego-frame [x, y] ---
        hist_start = max(0, frame_idx - self.t_h)
        hist_global = track[hist_start:frame_idx + 1:self.d_s, self.IDX_X:self.IDX_Y + 1].astype(np.float32)
        if len(hist_global) < self.max_hist:
            return None
        hist = self._rotate_to_ego(hist_global, ego_pos, ego_heading)

        # --- va: [lon_vel, lon_acc] ---
        hist_track = track[hist_start:frame_idx + 1:self.d_s]
        va = np.column_stack((hist_track[:, self.IDX_LON_VEL], hist_track[:, self.IDX_LON_ACC])).astype(np.float32)
        if len(va) < self.max_hist:
            return None

        return fut, hist, va

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]
        result = self._extract(ds_id, veh_id, t)
        if result is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
            )
        return result

    def collate_fn(self, samples):
        batch_size = len(samples)
        fut_batch = torch.zeros(batch_size, self.max_fut, 4, dtype=torch.float32)
        fut_mask = torch.zeros(batch_size, self.max_fut, dtype=torch.bool)
        hist_batch = torch.zeros(batch_size, self.max_hist, 2, dtype=torch.float32)
        va_batch = torch.zeros(batch_size, self.max_hist, 2, dtype=torch.float32)

        for i, (fut, hist, va) in enumerate(samples):
            if len(fut) == 0:
                continue
            fl = min(len(fut), self.max_fut)
            hl = min(len(hist), self.max_hist)
            fut_batch[i, :fl] = torch.from_numpy(fut[:fl])
            fut_mask[i, :fl] = True
            hist_batch[i, :hl] = torch.from_numpy(hist[:hl])
            va_batch[i, :hl] = torch.from_numpy(va[:hl])

        return fut_batch, fut_mask, hist_batch, va_batch


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("Compute RounD normalization stats")
    parser.add_argument(
        "--mat_file",
        default="/mnt/datasets/round/TrainSet.mat",
        type=str,
        help="Path to RounD TrainSet.mat",
    )
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 历史统计：只统计目标车辆自身历史轨迹，不统计邻车
# ---------------------------------------------------------------------------

def collect_round_hist_stats(mat_path, t_h, d_s, batch_size, num_workers):
    dataset = RoundHistDataset(str(mat_path), t_h=t_h, t_f=0, d_s=d_s)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    hist_values = []
    va_values = []
    valid_samples = 0

    for batch in tqdm(loader, desc="RoundHistStats", unit="batch", ncols=100):
        valid = batch["sample_valid"].bool()
        if valid.numel() == 0 or not bool(valid.any()):
            continue

        valid_samples += int(valid.sum().item())
        hist_values.append(batch["hist"][valid].reshape(-1, 2).cpu().numpy())
        va_values.append(batch["va"][valid].reshape(-1, 2).cpu().numpy())

    hist_all = (
        np.concatenate(hist_values, axis=0).astype(np.float64, copy=False)
        if hist_values
        else np.empty((0, 2), dtype=np.float64)
    )
    va_all = (
        np.concatenate(va_values, axis=0).astype(np.float64, copy=False)
        if va_values
        else np.empty((0, 2), dtype=np.float64)
    )

    pos_mean = hist_all.mean(axis=0) if hist_all.shape[0] > 0 else np.zeros(2)
    pos_std = hist_all.std(axis=0) if hist_all.shape[0] > 0 else np.zeros(2)
    va_mean = va_all.mean(axis=0) if va_all.shape[0] > 0 else np.zeros(2)
    va_std = va_all.std(axis=0) if va_all.shape[0] > 0 else np.zeros(2)

    return {
        "valid_samples": valid_samples,
        "hist_points": hist_all.shape[0],
        "va_points": va_all.shape[0],
        "pos_mean": pos_mean,
        "pos_std": pos_std,
        "va_mean": va_mean,
        "va_std": va_std,
    }


# ---------------------------------------------------------------------------
# 主流程：accumulate → concatenate → mean/std（对齐 calculate_stats.py）
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    mat_path = Path(args.mat_file)
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    print(f"[RoundStats] mat_file   = {mat_path}")
    print(f"[RoundStats] batch_size = {args.batch_size}")

    t_h, t_f, d_s, _ = get_time_params("round")
    print(f"[RoundStats] t_h={t_h}, t_f={t_f}, d_s={d_s}")

    dataset = RounDStatsDataset(str(mat_path), t_h=t_h, t_f=t_f, d_s=d_s)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    fut_list = []
    for fut_batch, fut_mask, _, _ in tqdm(loader, desc="RoundFutureStats", unit="batch", ncols=100):
        valid = fut_batch[fut_mask]
        if valid.numel() > 0:
            fut_list.append(valid.cpu().numpy())

    # ---- 合并 & 计算 ----
    fut_all = np.concatenate(fut_list, axis=0).astype(np.float64, copy=False) if fut_list else np.empty((0, 4))

    fut_mean = fut_all.mean(axis=0) if fut_all.shape[0] > 0 else np.zeros(4)
    fut_std = fut_all.std(axis=0) if fut_all.shape[0] > 0 else np.zeros(4)
    hist_stats = collect_round_hist_stats(
        mat_path=mat_path,
        t_h=t_h,
        d_s=d_s,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    pos_mean = hist_stats["pos_mean"]
    pos_std = hist_stats["pos_std"]
    va_mean = hist_stats["va_mean"]
    va_std = hist_stats["va_std"]

    # ---- 输出 ----
    print(f"\n[RoundStats] fut points:  {fut_all.shape[0]:,}")
    print(f"[RoundStats] hist valid samples: {hist_stats['valid_samples']:,}")
    print(f"[RoundStats] hist points: {hist_stats['hist_points']:,}")
    print(f"[RoundStats] va points:   {hist_stats['va_points']:,}")
    print(f"[RoundStats] fut_mean = {fut_mean.tolist()}")
    print(f"[RoundStats] fut_std  = {fut_std.tolist()}")
    print(f"[RoundStats] fut_var  = {(fut_std ** 2).tolist()}")
    print(f"[RoundStats] pos_mean = {pos_mean.tolist()}")
    print(f"[RoundStats] pos_std  = {pos_std.tolist()}")
    print(f"[RoundStats] va_mean  = {va_mean.tolist()}")
    print(f"[RoundStats] va_std   = {va_std.tolist()}")

    # ---- 可粘贴代码 ----
    print("\n# ===== fut_model.py (DiffusionFut.__init__) =====")
    print(
        'self.register_buffer("fut_mean", '
        f'torch.tensor({fut_mean.tolist()}, dtype=torch.float32), persistent=False)'
    )
    print(
        'self.register_buffer("fut_std", '
        f'torch.tensor({fut_std.tolist()}, dtype=torch.float32), persistent=False)'
    )

    print("\n# ===== hist_model.py (DiffusionPast.__init__) =====")
    print(f"pos_mean = {pos_mean.tolist()}")
    print(f"pos_std  = {pos_std.tolist()}")
    print(f"va_mean  = {va_mean.tolist()}")
    print(f"va_std   = {va_std.tolist()}")

    print("\n# lane/class 固定值 (RounD 无变化)")
    print("lane_center = [59.0]")
    print("lane_scale  = [59.0]")
    print("class_center = [1.0]")
    print("class_scale  = [1.0]")
    print("lane_min  = [0.0]")
    print("lane_max  = [118.0]")
    print("class_min = [1.0]")
    print("class_max = [1.0]")


if __name__ == "__main__":
    main()
