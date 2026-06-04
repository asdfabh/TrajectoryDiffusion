import argparse
from pathlib import Path

import numpy as np
import scipy.io as scp
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from method_diffusion.dataset.future_features import build_future_xy_theta_v
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset


RAW_TO_METER = 0.3048


class NgsimFutureDataset(Dataset):
    """
    仅构造 Ego future 轨迹。
    输出为相对当前时刻参考点的 future [x, y, theta, v]。
    """

    def __init__(self, mat_file, t_f=50, d_s=2):
        mat_data = scp.loadmat(mat_file)
        self.D = mat_data["traj"]
        self.T = mat_data["tracks"]
        self.t_f = int(t_f)
        self.d_s = int(d_s)
        self.maxlen = self.t_f // self.d_s
        self._track_cache = {}

    def __len__(self):
        return len(self.D)

    def _get_track(self, ds_id, veh_id):
        key = (int(ds_id), int(veh_id))
        if key in self._track_cache:
            return self._track_cache[key]

        if veh_id <= 0 or self.T.shape[1] <= veh_id - 1:
            self._track_cache[key] = None
            return None

        track = self.T[ds_id - 1][veh_id - 1].transpose()
        if track.size == 0:
            self._track_cache[key] = None
            return None

        self._track_cache[key] = track
        return track

    def _extract_future(self, ds_id, veh_id, t):
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return None

        frame_idx = np.where(track[:, 0] == t)[0]
        if frame_idx.size == 0:
            return None
        frame_idx = int(frame_idx.item())
        fut = build_future_xy_theta_v(track, frame_idx, self.t_f, self.d_s)
        if len(fut) == 0:
            return None
        fut = fut.astype(np.float32, copy=True)
        fut[:, 0:2] *= RAW_TO_METER
        fut[:, 3:4] *= RAW_TO_METER
        return fut

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]

        fut = self._extract_future(ds_id, veh_id, t)
        if fut is None:
            return np.empty((0, 4), dtype=np.float32)
        return fut

    def collate_fn(self, samples):
        batch_size = len(samples)
        fut_batch = torch.zeros(batch_size, self.maxlen, 4, dtype=torch.float32)
        fut_mask_batch = torch.zeros(batch_size, self.maxlen, dtype=torch.bool)

        for sample_id, fut in enumerate(samples):
            if len(fut) == 0:
                continue
            cur_len = min(len(fut), self.maxlen)
            fut_batch[sample_id, :cur_len] = torch.from_numpy(fut[:cur_len])
            fut_mask_batch[sample_id, :cur_len] = True

        return fut_batch, fut_mask_batch

def parse_args():
    parser = argparse.ArgumentParser("Calculate future normalization stats")
    parser.add_argument("--mat_file", default="/mnt/datasets/highDdata/TrainSet.mat", type=str, help="highDdata or ngsimdata")
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--t_h", default=30, type=int)
    parser.add_argument("--t_f", default=50, type=int)
    parser.add_argument("--d_s", default=2, type=int)
    return parser.parse_args()


def collect_hist_stats(mat_file, t_h, d_s, batch_size, num_workers):
    dataset = NgsimHistDataset(str(mat_file), t_h=t_h, d_s=d_s)
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
    for batch in tqdm(loader, desc="HistStats", ncols=100):
        valid = batch["sample_valid"]
        if valid.numel() == 0 or not bool(valid.any()):
            continue
        hist_values.append(batch["hist"][valid].reshape(-1, 2).cpu().numpy())
        va_values.append(batch["va"][valid].reshape(-1, 2).cpu().numpy())

    if len(hist_values) == 0:
        pos_mean = np.zeros(2, dtype=np.float64)
        pos_std = np.zeros(2, dtype=np.float64)
        va_mean = np.zeros(2, dtype=np.float64)
        va_std = np.zeros(2, dtype=np.float64)
    else:
        hist_values = np.concatenate(hist_values, axis=0).astype(np.float64, copy=False)
        va_values = np.concatenate(va_values, axis=0).astype(np.float64, copy=False)
        pos_mean = np.mean(hist_values, axis=0)
        pos_std = np.std(hist_values, axis=0)
        va_mean = np.mean(va_values, axis=0)
        va_std = np.std(va_values, axis=0)

    return pos_mean, pos_std, va_mean, va_std


def main():
    args = parse_args()
    mat_path = Path(args.mat_file)
    if not mat_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {mat_path}")

    dataset = NgsimFutureDataset(str(mat_path), t_f=args.t_f, d_s=args.d_s)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        drop_last=False,
    )

    fut_values = []
    for fut_batch, fut_mask_batch in tqdm(loader, desc="FutureStats", ncols=100):
        valid_fut = fut_batch[fut_mask_batch]
        if valid_fut.numel() == 0:
            continue
        fut_values.append(valid_fut.cpu().numpy())

    if len(fut_values) == 0:
        fut_mean = np.zeros(4, dtype=np.float64)
        fut_std = np.zeros(4, dtype=np.float64)
    else:
        fut_values = np.concatenate(fut_values, axis=0).astype(np.float64, copy=False)
        fut_mean = np.mean(fut_values, axis=0)
        fut_std = np.std(fut_values, axis=0)

    fut_var = fut_std ** 2
    print(f"fut_mean = {fut_mean.tolist()}")
    print(f"fut_std = {fut_std.tolist()}")
    print(f"fut_var = {fut_var.tolist()}")

    pos_mean, pos_std, va_mean, va_std = collect_hist_stats(
        mat_file=mat_path,
        t_h=args.t_h,
        d_s=args.d_s,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"pos_mean = {pos_mean.tolist()}")
    print(f"pos_std = {pos_std.tolist()}")
    print(f"va_mean = {va_mean.tolist()}")
    print(f"va_std = {va_std.tolist()}")


if __name__ == "__main__":
    main()
