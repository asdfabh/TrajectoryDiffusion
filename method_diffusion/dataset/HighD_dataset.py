import os

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError:  # pragma: no cover - highD 运行环境缺少 h5py 时给出明确报错
    h5py = None


class _HighDTrackStore:
    def __init__(self, mat_file):
        if h5py is None:
            raise ModuleNotFoundError(
                f"h5py is required to read MATLAB v7.3/HDF5 highD dataset: {mat_file}"
            )

        self.mat_file = mat_file
        self.traj = self._load_traj()
        self._pid = None
        self._file = None
        self._refs = None
        self._tracks = None

    def _load_traj(self):
        with h5py.File(self.mat_file, "r") as f:
            traj = np.asarray(f["traj"], dtype=np.float32)
        if traj.ndim != 2:
            raise ValueError(f"Unexpected highD traj ndim={traj.ndim}, expected 2.")
        if traj.shape[1] == 50:
            return traj
        if traj.shape[0] == 50:
            return traj.transpose()
        raise ValueError(f"Unexpected highD traj shape {traj.shape}, expected [N, 50] or [50, N].")

    def _ensure_open(self):
        pid = os.getpid()
        if self._file is not None and self._pid == pid:
            return

        self.close()
        self._file = h5py.File(self.mat_file, "r")
        self._refs = self._file["#refs#"]
        self._tracks = self._file["tracks"]
        self._pid = pid

    def get_track(self, ds_id, veh_id):
        if veh_id <= 0:
            return None

        self._ensure_open()
        # highD 的 HDF5 tracks 原始布局是 [veh, ds]
        if veh_id > self._tracks.shape[0]:
            return None
        if ds_id <= 0 or ds_id > self._tracks.shape[1]:
            return None

        ref = self._tracks[veh_id - 1, ds_id - 1]
        if not ref:
            return None

        track = np.asarray(self._refs[ref][()], dtype=np.float32)
        if track.ndim != 2:
            return None
        if track.shape[1] == 9:
            return track
        if track.shape[0] == 9:
            return track.transpose()
        return None

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None
                self._refs = None
                self._tracks = None
                self._pid = None


class HighDDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3), feature_dim=4):
        self.track_store = _HighDTrackStore(mat_file)
        self.D = self.track_store.traj
        self.t_h = t_h
        self.t_f = t_f
        self.d_s = d_s
        self.enc_size = enc_size
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.alltime = 0
        self._track_cache = {}

    def __len__(self):
        return len(self.D)

    def __del__(self):
        if hasattr(self, "track_store"):
            self.track_store.close()

    def _get_track(self, ds_id, veh_id):
        key = (int(ds_id), int(veh_id))
        if key in self._track_cache:
            return self._track_cache[key]

        track = self.track_store.get_track(int(ds_id), int(veh_id))
        self._track_cache[key] = track
        return track

    def __getitem__(self, idx):
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        grid = self.D[idx, 11:]
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        hist = self.getHistory(vehId, t, vehId, dsId)
        refdistance = np.zeros_like(hist[:, 0]).reshape(len(hist), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1]).reshape(len(uu), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)

        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1
        nbrs_num = np.array(sum(1 for arr in neighbors if arr.size != 0))

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
        )

    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        refTrack = self._get_track(dsId, refVehId)
        vehTrack = self._get_track(dsId, vehId)
        if refTrack is None or vehTrack is None:
            return np.empty([0, 1])

        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 1])

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 5]
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 1])
        return hist

    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        refTrack = self._get_track(dsId, refVehId)
        vehTrack = self._get_track(dsId, vehId)
        if refTrack is None or vehTrack is None:
            return np.empty([0, 1])

        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 1])

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 6]
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 1])
        return hist

    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        refTrack = self._get_track(dsId, refVehId)
        vehTrack = self._get_track(dsId, vehId)
        if refTrack is None or vehTrack is None:
            return np.empty([0, 2])

        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 2])

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 3:5]
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 2])
        return hist

    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        refTrack = self._get_track(dsId, refVehId)
        vehTrack = self._get_track(dsId, vehId)
        if refTrack is None or vehTrack is None:
            return np.empty([0, 2])

        x = np.where(refTrack[:, 0] == t)
        if x[0].size == 0:
            return np.empty([0, 2])
        refPos = refTrack[x][0, 1:3]

        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 2])

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 2])
        return hist

    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        refTrack = self._get_track(dsId, refVehId)
        vehTrack = self._get_track(dsId, vehId)
        if refTrack is None or vehTrack is None:
            return np.empty([0, 1])

        x = np.where(refTrack[:, 0] == t)
        if x[0].size == 0:
            return np.empty([0, 1])
        refPos = refTrack[x][0, 1:3]

        if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 1])

        stpt = np.maximum(0, np.argwhere(vehTrack[:, 0] == t).item() - self.t_h)
        enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
        hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
        uu = np.power(hist - hist_ref, 2)
        distance = np.sqrt(uu[:, 0] + uu[:, 1]).reshape(len(hist), 1)
        if len(hist) < self.t_h // self.d_s + 1:
            return np.empty([0, 1])
        return distance

    def getFuture(self, vehId, t, dsId):
        vehTrack = self._get_track(dsId, vehId)
        if vehTrack is None or vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
            return np.empty([0, 2])

        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    def collate_fn(self, samples):
        nbr_batch_size = 0
        for _, _, nbrs, _, _, _, _, _, _, _, _, _, _, _ in samples:
            nbr_batch_size += sum(len(nbrs[i]) != 0 for i in range(len(nbrs)))

        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrsva_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrslane_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsclass_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsdis_batch = torch.zeros(nbr_batch_size, maxlen, 1)

        pos = [0, 0]
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size).bool()
        temporal_mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.feature_dim).bool()
        map_position = torch.zeros(0, 2)

        hist_batch = torch.zeros(len(samples), maxlen, 2)
        distance_batch = torch.zeros(len(samples), maxlen, 1)
        fut_batch = torch.zeros(len(samples), self.t_f // self.d_s, 2)
        op_mask_batch = torch.zeros(len(samples), self.t_f // self.d_s, 2)
        lat_enc_batch = torch.zeros(len(samples), 3)
        lon_enc_batch = torch.zeros(len(samples), 3)
        va_batch = torch.zeros(len(samples), maxlen, 2)
        lane_batch = torch.zeros(len(samples), maxlen, 1)
        class_batch = torch.zeros(len(samples), maxlen, 1)
        nbrs_num_batch = torch.zeros(len(samples), 1)
        count = count1 = count2 = count3 = count4 = 0

        for sampleId, (hist, fut, nbrs, lat_enc, lon_enc, va, neighborsva, lane, neighborslane, refdistance,
                       neighborsdistance, cclass, neighborsclass, nbrs_num) in enumerate(samples):
            hist_batch[sampleId, 0:len(hist), 0] = torch.from_numpy(hist[:, 0])
            hist_batch[sampleId, 0:len(hist), 1] = torch.from_numpy(hist[:, 1])
            distance_batch[sampleId, 0:len(hist), :] = torch.from_numpy(refdistance)
            fut_batch[sampleId, 0:len(fut), 0] = torch.from_numpy(fut[:, 0])
            fut_batch[sampleId, 0:len(fut), 1] = torch.from_numpy(fut[:, 1])
            op_mask_batch[sampleId, 0:len(fut), :] = 1
            lat_enc_batch[sampleId, :] = torch.from_numpy(lat_enc)
            lon_enc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            va_batch[sampleId, 0:len(va), 0] = torch.from_numpy(va[:, 0])
            va_batch[sampleId, 0:len(va), 1] = torch.from_numpy(va[:, 1])
            lane_batch[sampleId, 0:len(lane), 0] = torch.from_numpy(lane)
            class_batch[sampleId, 0:len(cclass), 0] = torch.from_numpy(cclass)
            nbrs_num_batch[sampleId, :] = torch.from_numpy(nbrs_num)

            for idx, nbr in enumerate(nbrs):
                if len(nbr) != 0:
                    nbrs_batch[count, 0:len(nbr), 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[count, 0:len(nbr), 1] = torch.from_numpy(nbr[:, 1])
                    pos[0] = idx % self.grid_size[0]
                    pos[1] = idx // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    temporal_mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.feature_dim).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1

            for _, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrsva_batch[count1, 0:len(nbrva), 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[count1, 0:len(nbrva), 1] = torch.from_numpy(nbrva[:, 1])
                    count1 += 1

            for _, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[count2, 0:len(nbrlane), :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for _, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[count3, 0:len(nbrdis), :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for _, nbrclass in enumerate(neighborsclass):
                if len(nbrclass) != 0:
                    nbrsclass_batch[count4, 0:len(nbrclass), :] = torch.from_numpy(nbrclass)
                    count4 += 1

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


class HighDHistDataset(Dataset):
    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, **kwargs):
        self.track_store = _HighDTrackStore(mat_file)
        self.D = self.track_store.traj
        self.t_h = int(t_h)
        self.d_s = int(d_s)
        self.maxlen = self.t_h // self.d_s + 1
        self._track_cache = {}

    def __len__(self):
        return len(self.D)

    def __del__(self):
        if hasattr(self, "track_store"):
            self.track_store.close()

    def _get_track(self, ds_id, veh_id):
        key = (int(ds_id), int(veh_id))
        if key in self._track_cache:
            return self._track_cache[key]
        track = self.track_store.get_track(int(ds_id), int(veh_id))
        self._track_cache[key] = track
        return track

    def _extract_hist_bundle(self, ds_id, veh_id, t):
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
        sample_valid_batch = torch.zeros(batch_size, dtype=torch.bool)

        for sample_id, (hist, va, lane, cclass) in enumerate(samples):
            if len(hist) == 0:
                continue
            cur_len = min(len(hist), self.maxlen)
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
