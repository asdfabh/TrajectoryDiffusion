import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scp


class NgsimHistDataset(Dataset):
    """
    Hist 训练/评估专用轻量数据集：
    仅构造 ego 历史相关特征，避免邻居网格与未来轨迹的额外开销。
    """

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, **kwargs):
        mat_data = scp.loadmat(mat_file)
        self.D = mat_data["traj"]
        self.T = mat_data["tracks"]
        self.t_h = int(t_h)
        self.d_s = int(d_s)
        self.maxlen = self.t_h // self.d_s + 1

        # 轨迹缓存：避免重复 transpose
        self._track_cache = {}

    def __len__(self):
        return len(self.D)

    def _get_track(self, ds_id, veh_id):
        key = (int(ds_id), int(veh_id))
        if key in self._track_cache:
            return self._track_cache[key]

        if veh_id <= 0:
            self._track_cache[key] = None
            return None
        if self.T.shape[1] <= veh_id - 1:
            self._track_cache[key] = None
            return None

        track = self.T[ds_id - 1][veh_id - 1].transpose()
        if track.size == 0:
            self._track_cache[key] = None
            return None

        self._track_cache[key] = track
        return track

    def _extract_hist_bundle(self, ds_id, veh_id, t):
        """
        返回：
        hist:  [L, 2]  以当前时刻 ego 位置为原点的历史位置
        va:    [L, 2]  速度/加速度
        lane:  [L, 1]  车道编号
        cclass:[L, 1]  车辆类别
        """
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return None

        frame_idx = np.where(track[:, 0] == t)[0]
        if frame_idx.size == 0:
            return None
        frame_idx = int(frame_idx.item())

        stpt = max(0, frame_idx - self.t_h)
        enpt = frame_idx + 1
        hist_raw = track[stpt:enpt:self.d_s, :]

        if len(hist_raw) < self.maxlen:
            return None

        ref_pos = track[frame_idx, 1:3]
        hist = (hist_raw[:, 1:3] - ref_pos).astype(np.float32, copy=False)
        va = hist_raw[:, 3:5].astype(np.float32, copy=False)
        lane = hist_raw[:, 5:6].astype(np.float32, copy=False)
        cclass = hist_raw[:, 6:7].astype(np.float32, copy=False)
        return hist, va, lane, cclass

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]

        out = self._extract_hist_bundle(ds_id, veh_id, t)
        if out is None:
            # 与原始数据集行为对齐：无效样本返回空数组，由 collate 处理
            empty_hist = np.empty((0, 2), dtype=np.float32)
            empty_lane = np.empty((0, 1), dtype=np.float32)
            return empty_hist, empty_hist, empty_lane, empty_lane

        return out

    def collate_fn(self, samples):
        batch_size = len(samples)
        hist_batch = torch.zeros(batch_size, self.maxlen, 2, dtype=torch.float32)
        va_batch = torch.zeros(batch_size, self.maxlen, 2, dtype=torch.float32)
        lane_batch = torch.zeros(batch_size, self.maxlen, 1, dtype=torch.float32)
        class_batch = torch.zeros(batch_size, self.maxlen, 1, dtype=torch.float32)

        for sample_id, (hist, va, lane, cclass) in enumerate(samples):
            if len(hist) == 0:
                continue

            cur_len = min(len(hist), self.maxlen)
            hist_batch[sample_id, :cur_len, :] = torch.from_numpy(hist[:cur_len])
            va_batch[sample_id, :cur_len, :] = torch.from_numpy(va[:cur_len])
            lane_batch[sample_id, :cur_len, :] = torch.from_numpy(lane[:cur_len])
            class_batch[sample_id, :cur_len, :] = torch.from_numpy(cclass[:cur_len])

        return {
            "hist": hist_batch,
            "va": va_batch,
            "lane": lane_batch,
            "cclass": class_batch,
        }
