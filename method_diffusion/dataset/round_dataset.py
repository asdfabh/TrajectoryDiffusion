import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scp
import h5py

from method_diffusion.dataset.future_features import derive_theta_from_positions


class RoundDataset(Dataset):
    """
    rounD 数据集适配器。

    输入 .mat 使用 AnchorFlow 的 rounD 预处理格式，输出 batch 字典对齐 NgsimDataset：
    history / neighbor 使用 ego 局部坐标，future 为 [x, y, theta, v]。
    """

    def __init__(self, mat_file, t_h=20, t_f=40, d_s=2, enc_size=64, grid_size=(13, 3), feature_dim=4):
        self.D, self.T = self._load_mat(mat_file)
        self.t_h = int(t_h)
        self.t_f = int(t_f)
        self.d_s = int(d_s)
        self.enc_size = int(enc_size)
        self.grid_size = grid_size
        self.feature_dim = int(feature_dim)
        self.max_hist_len = self.t_h // self.d_s + 1
        self.max_fut_len = self.t_f // self.d_s if self.t_f > 0 else 0

        # tracks 列：Frame, X, Y, Heading, LonVel, LatVel, LonAcc, LatAcc, Lane
        self.idx_x = 1
        self.idx_y = 2
        self.idx_heading = 3
        self.idx_v = 4
        self.idx_lane = 8
        self.valid_indices = self._build_valid_indices()

    def _load_mat(self, mat_file):
        try:
            mat = scp.loadmat(mat_file)
            return mat["traj"], mat["tracks"]
        except NotImplementedError:
            return self._load_hdf5_mat(mat_file)

    def _load_hdf5_mat(self, mat_file):
        with h5py.File(mat_file, "r") as f:
            traj = f["traj"][:].transpose()
            track_refs = f["tracks"][:].transpose()
            tracks = np.empty(track_refs.shape, dtype=object)
            for ds_idx in range(track_refs.shape[0]):
                for veh_idx in range(track_refs.shape[1]):
                    try:
                        track = f[track_refs[ds_idx, veh_idx]][:]
                    except (KeyError, TypeError, ValueError):
                        tracks[ds_idx, veh_idx] = np.empty((0, 0), dtype=np.float32)
                        continue

                    if track.ndim < 2:
                        tracks[ds_idx, veh_idx] = np.empty((0, 0), dtype=np.float32)
                    else:
                        tracks[ds_idx, veh_idx] = track.transpose().astype(np.float32, copy=False)
        return traj, tracks

    def __len__(self):
        return len(self.valid_indices)

    def _build_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.D)):
            if self._is_valid_index(idx):
                valid_indices.append(idx)
        return np.asarray(valid_indices, dtype=np.int64)

    def _is_valid_index(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]

        track = self._get_track(ds_id, veh_id)
        if track is None:
            return False
        frame_idx = self._find_frame_idx(track, t)
        if frame_idx is None:
            return False

        hist_start = max(0, frame_idx - self.t_h)
        hist_len = len(track[hist_start:frame_idx + 1:self.d_s])
        if hist_len < self.max_hist_len:
            return False

        if self.max_fut_len == 0:
            return True
        fut_start = frame_idx + self.d_s
        fut_end = min(len(track), frame_idx + self.t_f + 1)
        fut_len = len(track[fut_start:fut_end:self.d_s])
        return fut_len >= self.max_fut_len

    def _get_track(self, ds_id, veh_id):
        if veh_id <= 0 or self.T.shape[1] <= veh_id - 1:
            return None
        track = self.T[ds_id - 1][veh_id - 1]
        if track.size == 0:
            return None
        return track.transpose()

    def _find_frame_idx(self, track, t):
        frame_idx = np.where(track[:, 0] == t)[0]
        if frame_idx.size == 0:
            return None
        return int(frame_idx.item())

    def rotate_to_ego_frame(self, positions, ref_pos, ego_heading):
        if len(positions) == 0:
            return positions.astype(np.float32, copy=False)

        delta_x = positions[:, 0] - ref_pos[0]
        delta_y = positions[:, 1] - ref_pos[1]
        cos_h = np.cos(ego_heading)
        sin_h = np.sin(ego_heading)

        local_x = -delta_x * sin_h + delta_y * cos_h
        local_y = delta_x * cos_h + delta_y * sin_h
        return np.column_stack((local_x, local_y)).astype(np.float32, copy=False)

    def _ego_pose(self, ds_id, veh_id, t):
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return None
        frame_idx = self._find_frame_idx(track, t)
        if frame_idx is None:
            return None
        ego_pos = track[frame_idx, self.idx_x:self.idx_y + 1]
        ego_heading = float(track[frame_idx, self.idx_heading])
        return track, frame_idx, ego_pos, ego_heading

    def getHistory(self, veh_id, t, ref_veh_id, ds_id, ego_pos=None, ego_heading=None):
        if ego_pos is None or ego_heading is None:
            ego = self._ego_pose(ds_id, ref_veh_id, t)
            if ego is None:
                return np.empty((0, 2), dtype=np.float32)
            _, _, ego_pos, ego_heading = ego

        track = self._get_track(ds_id, veh_id)
        if track is None:
            return np.empty((0, 2), dtype=np.float32)
        frame_idx = self._find_frame_idx(track, t)
        if frame_idx is None:
            return np.empty((0, 2), dtype=np.float32)

        stpt = max(0, frame_idx - self.t_h)
        hist_global = track[stpt:frame_idx + 1:self.d_s, self.idx_x:self.idx_y + 1]
        if len(hist_global) < self.max_hist_len:
            return np.empty((0, 2), dtype=np.float32)
        return self.rotate_to_ego_frame(hist_global, ego_pos, ego_heading)

    def getFuture(self, veh_id, t, ds_id, ego_pos=None, ego_heading=None):
        if ego_pos is None or ego_heading is None:
            ego = self._ego_pose(ds_id, veh_id, t)
            if ego is None:
                return np.empty((0, 4), dtype=np.float32)
            track, frame_idx, ego_pos, ego_heading = ego
        else:
            track = self._get_track(ds_id, veh_id)
            if track is None:
                return np.empty((0, 4), dtype=np.float32)
            frame_idx = self._find_frame_idx(track, t)
            if frame_idx is None:
                return np.empty((0, 4), dtype=np.float32)

        stpt = frame_idx + self.d_s
        enpt = min(len(track), frame_idx + self.t_f + 1)
        fut_global = track[stpt:enpt:self.d_s, self.idx_x:self.idx_y + 1]
        if len(fut_global) == 0:
            return np.empty((0, 4), dtype=np.float32)

        fut_xy = self.rotate_to_ego_frame(fut_global, ego_pos, ego_heading)
        theta = derive_theta_from_positions(fut_xy, np.zeros(2, dtype=np.float32))
        speed = track[stpt:enpt:self.d_s, self.idx_v:self.idx_v + 1].astype(np.float32, copy=False)
        return np.concatenate([fut_xy, theta, speed], axis=1).astype(np.float32, copy=False)

    def getVA(self, veh_id, t, ds_id):
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return np.empty((0, 2), dtype=np.float32)
        frame_idx = self._find_frame_idx(track, t)
        if frame_idx is None:
            return np.empty((0, 2), dtype=np.float32)

        stpt = max(0, frame_idx - self.t_h)
        va = track[stpt:frame_idx + 1:self.d_s, self.idx_v:self.idx_v + 2]
        if len(va) < self.max_hist_len:
            return np.empty((0, 2), dtype=np.float32)
        return va.astype(np.float32, copy=False)

    def getLane(self, veh_id, t, ds_id):
        track = self._get_track(ds_id, veh_id)
        if track is None:
            return np.empty((0, 1), dtype=np.float32)
        frame_idx = self._find_frame_idx(track, t)
        if frame_idx is None:
            return np.empty((0, 1), dtype=np.float32)

        stpt = max(0, frame_idx - self.t_h)
        lane = track[stpt:frame_idx + 1:self.d_s, self.idx_lane:self.idx_lane + 1]
        if len(lane) < self.max_hist_len:
            return np.empty((0, 1), dtype=np.float32)
        return lane.astype(np.float32, copy=False)

    def _build_sample(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]

        ego = self._ego_pose(ds_id, veh_id, t)
        if ego is None:
            return None
        _, _, ego_pos, ego_heading = ego

        hist = self.getHistory(veh_id, t, veh_id, ds_id, ego_pos, ego_heading)
        fut = self.getFuture(veh_id, t, ds_id, ego_pos, ego_heading)
        va = self.getVA(veh_id, t, ds_id)
        lane = self.getLane(veh_id, t, ds_id)
        if (
            hist.shape[0] < self.max_hist_len
            or fut.shape[0] < self.max_fut_len
            or va.shape[0] < self.max_hist_len
            or lane.shape[0] < self.max_hist_len
        ):
            return None

        refdistance = np.zeros((self.max_hist_len, 1), dtype=np.float32)
        cclass = np.ones_like(lane, dtype=np.float32)

        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []
        grid_positions = []

        raw_neighbors = np.unique(self.D[idx, 15:])
        raw_neighbors = raw_neighbors[raw_neighbors != 0]
        for nbr_id in raw_neighbors:
            nbr_hist = self.getHistory(int(nbr_id), t, veh_id, ds_id, ego_pos, ego_heading)
            nbr_va = self.getVA(int(nbr_id), t, ds_id)
            nbr_lane = self.getLane(int(nbr_id), t, ds_id)
            if len(nbr_hist) == 0 or len(nbr_va) == 0 or len(nbr_lane) == 0:
                continue

            nbr_local_x, nbr_local_y = nbr_hist[-1, 0], nbr_hist[-1, 1]
            if nbr_local_x < -2.0:
                col = 0
            elif nbr_local_x > 2.0:
                col = 2
            else:
                col = 1

            row = int(np.round(nbr_local_y / 4.5)) + 6
            if row < 0 or row > 12:
                continue

            distance = np.linalg.norm(hist - nbr_hist, axis=1, keepdims=True).astype(np.float32, copy=False)
            neighbors.append(nbr_hist)
            neighborsva.append(nbr_va)
            neighborslane.append(nbr_lane)
            neighborsclass.append(np.ones_like(distance, dtype=np.float32))
            neighborsdistance.append(distance)
            grid_positions.append((row, col))

        lat_enc = np.zeros(3, dtype=np.float32)
        lat_class_raw = int(self.D[idx, 12])
        if lat_class_raw in (1, 4, 8):
            lat_enc[0] = 1.0
        elif lat_class_raw in (2, 5, 7):
            lat_enc[2] = 1.0
        else:
            lat_enc[1] = 1.0

        lon_enc = np.zeros(3, dtype=np.float32)
        lon_class_raw = int(self.D[idx, 13])
        if lon_class_raw in (1, 2, 3):
            lon_enc[lon_class_raw - 1] = 1.0
        else:
            lon_enc[1] = 1.0

        nbrs_num = np.array([[len(neighbors)]], dtype=np.float32)
        return (
            hist,
            fut,
            neighbors,
            lat_enc,
            lon_enc,
            va,
            neighborsva,
            lane,
            neighborslane,
            refdistance,
            neighborsdistance,
            cclass,
            neighborsclass,
            nbrs_num,
            grid_positions,
        )

    def __getitem__(self, idx):
        sample = self._build_sample(int(self.valid_indices[idx]))
        if sample is None:
            raise IndexError(f"Invalid rounD sample after filtering: idx={idx}")
        return sample

    def collate_fn(self, samples):
        nbr_batch_size = sum(len(sample[2]) for sample in samples)
        maxlen = self.max_hist_len

        nbrs_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrsva_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrslane_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsclass_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsdis_batch = torch.zeros(nbr_batch_size, maxlen, 1)

        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size, dtype=torch.bool)
        temporal_mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.feature_dim, dtype=torch.bool)
        map_position = torch.zeros(0, 2)

        hist_batch = torch.zeros(len(samples), maxlen, 2)
        distance_batch = torch.zeros(len(samples), maxlen, 1)
        fut_batch = torch.zeros(len(samples), self.max_fut_len, 4)
        op_mask_batch = torch.zeros(len(samples), self.max_fut_len, 1)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 3)
        va_batch = torch.zeros(len(samples), maxlen, 2)
        lane_batch = torch.zeros(len(samples), maxlen, 1)
        class_batch = torch.zeros(len(samples), maxlen, 1)
        nbrs_num_batch = torch.zeros(len(samples), 1)

        count = 0
        for sample_id, data in enumerate(samples):
            (
                hist,
                fut,
                nbrs,
                lat_enc,
                lon_enc,
                va,
                nbrs_va,
                lane,
                nbrs_lane,
                refdistance,
                nbrs_dis,
                cclass,
                nbrs_class,
                nbrs_num,
                grid_pos,
            ) = data

            hist_batch[sample_id, :len(hist), :] = torch.from_numpy(hist)
            distance_batch[sample_id, :len(refdistance), :] = torch.from_numpy(refdistance)
            fut_batch[sample_id, :len(fut), :] = torch.from_numpy(fut)
            op_mask_batch[sample_id, :len(fut), 0] = 1
            lat_enc_batch[sample_id, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sample_id, :] = torch.from_numpy(lon_enc)
            va_batch[sample_id, :len(va), :] = torch.from_numpy(va)
            lane_batch[sample_id, :len(lane), :] = torch.from_numpy(lane)
            class_batch[sample_id, :len(cclass), :] = torch.from_numpy(cclass)
            nbrs_num_batch[sample_id, :] = torch.from_numpy(nbrs_num.reshape(1))

            for nbr_id, nbr in enumerate(nbrs):
                nbrs_batch[count, :len(nbr), :] = torch.from_numpy(nbr)
                nbrsva_batch[count, :len(nbrs_va[nbr_id]), :] = torch.from_numpy(nbrs_va[nbr_id])
                nbrslane_batch[count, :len(nbrs_lane[nbr_id]), :] = torch.from_numpy(nbrs_lane[nbr_id])
                nbrsdis_batch[count, :len(nbrs_dis[nbr_id]), :] = torch.from_numpy(nbrs_dis[nbr_id])
                nbrsclass_batch[count, :len(nbrs_class[nbr_id]), :] = torch.from_numpy(nbrs_class[nbr_id])

                row, col = grid_pos[nbr_id]
                mask_batch[sample_id, col, row, :] = True
                temporal_mask_batch[sample_id, col, row, :] = True
                map_position = torch.cat((map_position, torch.tensor([[col, row]], dtype=torch.float32)), 0)
                count += 1

        return {
            "hist": hist_batch,
            "nbrs": nbrs_batch,
            "mask": mask_batch,
            "lat_enc": lat_enc_batch,
            "lon_enc": lon_enc_batch,
            "fut": fut_batch,
            "op_mask": op_mask_batch,
            "va": va_batch,
            "nbrs_va": nbrsva_batch,
            "lane": lane_batch,
            "nbrs_lane": nbrslane_batch,
            "distance": distance_batch,
            "nbrs_distance": nbrsdis_batch,
            "cclass": class_batch,
            "nbrs_class": nbrsclass_batch,
            "map_position": map_position,
            "nbrs_num": nbrs_num_batch,
            "temporal_mask": temporal_mask_batch,
        }


class RoundHistDataset(RoundDataset):
    """rounD history-only 数据集，输出对齐 NgsimHistDataset。"""

    def __init__(self, mat_file, t_h=20, t_f=0, d_s=2, **kwargs):
        super().__init__(mat_file, t_h=t_h, t_f=t_f, d_s=d_s, **kwargs)

    def __getitem__(self, idx):
        ds_id = int(self.D[idx, 0])
        veh_id = int(self.D[idx, 1])
        t = self.D[idx, 2]
        ego = self._ego_pose(ds_id, veh_id, t)
        if ego is None:
            empty_hist = np.empty((0, 2), dtype=np.float32)
            empty_lane = np.empty((0, 1), dtype=np.float32)
            return empty_hist, empty_hist, empty_lane, empty_lane

        _, _, ego_pos, ego_heading = ego
        hist = self.getHistory(veh_id, t, veh_id, ds_id, ego_pos, ego_heading)
        va = self.getVA(veh_id, t, ds_id)
        lane = self.getLane(veh_id, t, ds_id)
        if len(hist) == 0 or len(va) == 0 or len(lane) == 0:
            empty_hist = np.empty((0, 2), dtype=np.float32)
            empty_lane = np.empty((0, 1), dtype=np.float32)
            return empty_hist, empty_hist, empty_lane, empty_lane
        return hist, va, lane, np.ones_like(lane, dtype=np.float32)

    def collate_fn(self, samples):
        batch_size = len(samples)
        hist_batch = torch.zeros(batch_size, self.max_hist_len, 2, dtype=torch.float32)
        va_batch = torch.zeros(batch_size, self.max_hist_len, 2, dtype=torch.float32)
        lane_batch = torch.zeros(batch_size, self.max_hist_len, 1, dtype=torch.float32)
        class_batch = torch.zeros(batch_size, self.max_hist_len, 1, dtype=torch.float32)
        sample_valid_batch = torch.zeros(batch_size, dtype=torch.bool)

        for sample_id, (hist, va, lane, cclass) in enumerate(samples):
            if len(hist) == 0:
                continue
            cur_len = min(len(hist), self.max_hist_len)
            hist_batch[sample_id, :cur_len, :] = torch.from_numpy(hist[:cur_len])
            va_batch[sample_id, :cur_len, :] = torch.from_numpy(va[:cur_len])
            lane_batch[sample_id, :cur_len, :] = torch.from_numpy(lane[:cur_len])
            class_batch[sample_id, :cur_len, :] = torch.from_numpy(cclass[:cur_len])
            sample_valid_batch[sample_id] = True

        return {
            "hist": hist_batch,
            "va": va_batch,
            "lane": lane_batch,
            "cclass": class_batch,
            "sample_valid": sample_valid_batch,
        }
