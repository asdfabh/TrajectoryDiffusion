import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as scp
from pathlib import Path

"""
1: Dataset Id
2: Vehicle Id
3: Frame Number
4: Local X
5: Local Y
6: velocity
7: acceleration
8: Lane Id
9: class
10: Lateral maneuver
11: Longitudinal maneuver
12-50: Neighbor Car Ids at grid location
"""


# Dataset class for the dataset
class NgsimDataset(Dataset):

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3)):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  #
        self.t_f = t_f  #
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of the grid cell
        self.grid_size = grid_size  # size of social context grid
        self.alltime = 0

        self.normalize = False
        self.norm_method = 'zscore'
        self.fixed_scale = 100.0


        self.pos_mean = np.array([0.0, 0.0])
        self.pos_std = np.array([1.0, 1.0])
        self.va_mean = np.array([0.0, 0.0])
        self.va_std = np.array([1.0, 1.0])
        self.lane_mean = np.array([0.0])
        self.lane_std = np.array([1.0])
        self.class_mean = np.array([0.0])
        self.class_std = np.array([1.0])
        self.stats_file = str(Path(__file__).resolve().parent / 'ngsim_stats.npz')

    def __len__(self):
        return len(self.D)

    """
    - 功能概述  
      - `__getitem__(self, idx)` 从数据表 `self.D` 取出第 `idx` 条样本，基于该样本的 dataset id / 车辆 id / 帧号，从轨迹数据 `self.T` 查询该车辆及其网格内邻居的历史、速度/加速度、车道、类别和未来轨迹，并构建邻居相关的列表与掩码信息（在本函数只是返回原始数据，掩码在 `collate_fn` 中构建）。

    - 输入  
      - `idx`：整数，样本在 `self.D` 中的行索引。  
      - 在 `self.D[idx, :]` 中使用的字段：  
        - `self.D[idx, 0]` -> `dsId`（数据集 id）  
        - `self.D[idx, 1]` -> `vehId`（目标车辆 id）  
        - `self.D[idx, 2]` -> `t`（参考帧号）  
        - `self.D[idx, 11:]` -> `grid`（网格中每个格子的车辆 id 列表）  
        - `self.D[idx, 9]`, `self.D[idx, 10]` -> 横/纵向动作（用于 one-hot 编码）

    - 查询依据（如何查）  
      - 使用 `dsId` 和 `vehId` 去索引 `self.T[dsId-1][vehId-1]`（每个 cell 存储该车的轨迹矩阵，第一列为帧号）。  
      - 在每个 helper 函数（`getHistory` / `getFuture` / `getVA` / `getLane` / `getClass`）中，通过匹配轨迹矩阵第一列等于 `t` 来定位参考行，然后按照 `t_h`, `t_f`, `d_s` 提取时间窗口的数据并以参考车辆在帧 `t` 的位置做相对化。  
      - 网格邻居循环用 `grid` 中的车辆 id 调用同样的 helper，从而得到每格的历史/VA/车道/类别；若格子为 0 或数据不足则返回空数组。

    - 输出（返回值，按顺序）  
      - `hist`：目标车辆相对历史轨迹，numpy ndarray，形状一般为 (t_h//d_s + 1, 2) 或空 `np.empty([0,2])`。  
      - `fut`：目标车辆相对未来轨迹，numpy ndarray，长度 ≤ t_f//d_s。  
      - `neighbors`：列表，每个元素为对应格子的邻居历史 ndarray（或空 ndarray）。  
      - `lat_enc`、`lon_enc`：长度 3 的 one-hot numpy 向量（横/纵动作）。  
      - `va`：目标车辆历史的速度/加速度 ndarray，形状与 `hist` 时间维一致或空。  
      - `neighborsva`：列表，对应格子中邻居的 VA ndarray 或空。  
      - `lane`：目标车辆历史车道 ndarray（每步一个值）或空。  
      - `neighborslane`：列表，邻居车道 ndarray （或空），在这里函数返回时已做 `.reshape(-1,1)`。  
      - `refdistance`：参考距离列，为与 `hist` 时间长度一致的零数组（用于后续填充/对齐）。  
      - `neighborsdistance`：列表，每个非空邻居与目标历史在每时间步的欧氏距离，形状 (T,1)，空则 (0,1)。  
      - `cclass`：目标车辆历史类别 ndarray 或空。  
      - `neighborsclass`：列表，邻居类别 ndarray 或空。  
      - `nbrs_num`：标量 numpy，网格中非空邻居的数量。

    - 其他要点  
      - 若目标或邻居在查询帧 `t` 找不到对应记录或历史长度不足，helper 会返回空数组；此函数会把空数组放入对应列表并在后续 `collate_fn` 中据此跳过或掩码处理。  
      - 邻居到目标的距离通过对历史位置逐步求差后计算欧氏距离得到。
    """

    def __getitem__(self, idx):

        dsId = self.D[idx, 0].astype(int)  # dataset id
        vehId = self.D[idx, 1].astype(int)  # agent id
        t = self.D[idx, 2]  # frame
        grid = self.D[idx, 11:]  # grid id
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []

        # Get track history 'hist' = ndarray, and future track 'fut' = ndarray
        hist = self.getHistory(vehId, t, vehId, dsId)  # 获取vehId 在vehID2下的相对历史轨迹，形状为(T,2)
        refdistance = np.zeros_like(hist[:, 0])
        refdistance = refdistance.reshape(len(refdistance), 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        # Get track histories of all neighbours 'neighbors' = [ndarray,[],ndarray,ndarray]
        for i in grid:
            nbrsdis = self.getHistory(i.astype(int), t, vehId, dsId)
            if nbrsdis.shape != (0, 2):
                uu = np.power(hist - nbrsdis, 2)
                distancexxx = np.sqrt(uu[:, 0] + uu[:, 1])
                distancexxx = distancexxx.reshape(len(distancexxx), 1)
            else:
                distancexxx = np.empty([0, 1])
            neighbors.append(nbrsdis)
            neighborsva.append(self.getVA(i.astype(int), t, vehId, dsId))
            neighborslane.append(self.getLane(
                i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(
                i.astype(int), t, vehId, dsId).reshape(-1, 1))
            neighborsdistance.append(distancexxx)
        # 横纵向动作独热编码，int(self.D[idx, 10] - 1 ) 读取的当作对的数字表示，然后转化为独热编码
        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1
        nbrs_num = np.array(sum(1 for arr in neighbors if arr.size != 0))

        sample = {
            "hist": hist,
            "fut": fut,
            "neighbors": neighbors,
            "lat_enc": lat_enc,
            "lon_enc": lon_enc,
            "va": va,
            "neighborsva": neighborsva,
            "lane": lane,
            "neighborslane": neighborslane,
            "refdistance": refdistance,
            "neighborsdistance": neighborsdistance,
            "cclass": cclass,
            "neighborsclass": neighborsclass,
            "nbrs_num": nbrs_num,
        }
        return sample

    # Get the lane of the vehicle
    def getLane(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    # Get the class of the vehicle
    def getClass(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 6]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 6]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return hist

    # Get the velocity and acceleration of the vehicle
    def getVA(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 3:5]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 3:5]

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    """
    该函数 getHistory(self, vehId, t, refVehId, dsId) 用于从 tracks（按数据集和车辆索引的 cell 数组）
    中抽取指定车辆在参考时刻 t 之前的一段历史轨迹，并以参考车辆在时刻 t 的位置为原点做相对坐标化。返回形状为 
    (T, 2) 的 numpy 数组，表示相对位置序列；在任一条件不满足时返回空数组 np.empty([0, 2])。
    """

    # Helper function to get track history
    def getHistory(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 2])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 2])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            x = np.where(refTrack[:, 0] == t)
            refPos = refTrack[x][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 2])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 2])
            return hist

    # Helper function to get track distance
    def getdistance(self, vehId, t, refVehId, dsId):
        if vehId == 0:
            return np.empty([0, 1])
        else:
            if self.T.shape[1] <= vehId - 1:
                return np.empty([0, 1])
            refTrack = self.T[dsId - 1][refVehId - 1].transpose()
            vehTrack = self.T[dsId - 1][vehId - 1].transpose()
            refPos = refTrack[np.where(refTrack[:, 0] == t)][0, 1:3]

            if vehTrack.size == 0 or np.argwhere(vehTrack[:, 0] == t).size == 0:
                return np.empty([0, 1])
            else:
                stpt = np.maximum(0, np.argwhere(
                    vehTrack[:, 0] == t).item() - self.t_h)
                enpt = np.argwhere(vehTrack[:, 0] == t).item() + 1
                hist = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
                hist_ref = refTrack[stpt:enpt:self.d_s, 1:3] - refPos
                uu = np.power(hist - hist_ref, 2)
                distance = np.sqrt(uu[:, 0] + uu[:, 1])
                distance = distance.reshape(len(distance), 1)

            if len(hist) < self.t_h // self.d_s + 1:
                return np.empty([0, 1])
            return distance

    # Helper function to get track future
    def getFuture(self, vehId, t, dsId):
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()
        refPos = vehTrack[np.where(vehTrack[:, 0] == t)][0, 1:3]
        stpt = np.argwhere(vehTrack[:, 0] == t).item() + self.d_s
        enpt = np.minimum(len(vehTrack), np.argwhere(
            vehTrack[:, 0] == t).item() + self.t_f + 1)
        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    # Collate function for dataloader
    def collate_fn(self, samples):
        # ttt = time.time()
        # Initialize neighbors and neighbors length batches:
        nbr_batch_size = 0
        for sample in samples:
            nbrs = sample["neighbors"]
            nbr_batch_size += sum(len(nbr) != 0 for nbr in nbrs)

        # Initialize neighbor batches:
        maxlen = self.t_h // self.d_s + 1
        nbrs_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrsva_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrslane_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsclass_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsdis_batch = torch.zeros(nbr_batch_size, maxlen, 1)

        hist_valid_mask = torch.zeros(len(samples), maxlen, 1, dtype=torch.bool)
        fut_valid_mask = torch.zeros(len(samples), self.t_f // self.d_s, 1, dtype=torch.bool)
        va_valid_mask = torch.zeros(len(samples), maxlen, 1, dtype=torch.bool)
        nbrs_valid_mask = torch.zeros(nbr_batch_size, maxlen, 1, dtype=torch.bool)
        nbrsva_valid_mask = torch.zeros(nbr_batch_size, maxlen, 1, dtype=torch.bool)

        # Initialize social mask batch:
        pos = [0, 0]
        map_position = torch.zeros(0, 2)
        mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0]).bool()
        # mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], self.enc_size)  # (batch,3,13,h)
        # temporal_mask_batch = torch.zeros(len(samples), self.grid_size[1], self.grid_size[0], 6)  # (batch,3,13,h)
        # mask_batch = mask_batch.bool()
        # temporal_mask_batch = temporal_mask_batch.bool()

        # Initialize history, history lengths, future, output mask, lateral maneuver and longitudinal maneuver batches:
        hist_batch = torch.zeros(len(samples), maxlen, 2)  # (len1,batch,2)
        distance_batch = torch.zeros(len(samples), maxlen, 1)
        fut_batch = torch.zeros(len(samples), self.t_f // self.d_s, 2)  # (len2,batch,2)
        op_mask_batch = torch.zeros(len(samples), self.t_f // self.d_s, 2)  # (len2,batch,2)
        lat_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        lon_enc_batch = torch.zeros(len(samples), 3)  # (batch,3)
        va_batch = torch.zeros(len(samples), maxlen, 2)
        lane_batch = torch.zeros(len(samples), maxlen, 1)
        class_batch = torch.zeros(len(samples), maxlen, 1)
        nbrs_num_batch = torch.zeros(len(samples), 1)
        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for sampleId, sample in enumerate(samples):
            hist = sample["hist"]
            fut = sample["fut"]
            neighbors = sample["neighbors"]
            lat_enc = sample["lat_enc"]
            lon_enc = sample["lon_enc"]
            va = sample["va"]
            neighborsva = sample["neighborsva"]
            lane = sample["lane"]
            neighborslane = sample["neighborslane"]
            refdistance = sample["refdistance"]
            neighborsdistance = sample["neighborsdistance"]
            cclass = sample["cclass"]
            neighborsclass = sample["neighborsclass"]
            nbrs_num = sample["nbrs_num"]

            # Set up history, future, lateral maneuver and longitudinal maneuver batches:
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

            hist_len = len(hist)
            fut_len = len(fut)
            va_len = len(va)
            if hist_len:
                hist_valid_mask[sampleId, :hist_len, 0] = True
            if fut_len:
                fut_valid_mask[sampleId, :fut_len, 0] = True
            if va_len:
                va_valid_mask[sampleId, :va_len, 0] = True

            # Set up neighbor, neighbor sequence length, and mask batches:
            for id, nbr in enumerate(neighbors):
                if len(nbr) != 0:
                    nbr_len = len(nbr)
                    nbrs_batch[count, 0:nbr_len, 0] = torch.from_numpy(nbr[:, 0])
                    nbrs_batch[count, 0:nbr_len, 1] = torch.from_numpy(nbr[:, 1])
                    nbrs_valid_mask[count, :nbr_len, 0] = True
                    pos[0] = id % self.grid_size[0]
                    pos[1] = id // self.grid_size[0]
                    mask_batch[sampleId, pos[1], pos[0]] = True
                    # mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(self.enc_size).byte()
                    # temporal_mask_batch[sampleId, pos[1], pos[0], :] = torch.ones(6).byte()
                    map_position = torch.cat((map_position, torch.tensor([[pos[1], pos[0]]])), 0)
                    count += 1
            for id, nbrva in enumerate(neighborsva):
                if len(nbrva) != 0:
                    nbrva_len = len(nbrva)
                    nbrsva_batch[count1, 0:nbrva_len, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[count1, 0:nbrva_len, 1] = torch.from_numpy(nbrva[:, 1])
                    nbrsva_valid_mask[count1, :nbrva_len, 0] = True
                    count1 += 1

            # for id, nbrlane in enumerate(neighborslane):
            #     if len(nbrlane) != 0:
            #         for nbrslanet in range(len(nbrlane)):
            #             nbrslane_batch[nbrslanet, count2, int(nbrlane[nbrslanet] - 1)] = 1
            #         count2 += 1

            for id, nbrlane in enumerate(neighborslane):
                if len(nbrlane) != 0:
                    nbrslane_batch[count2, 0:len(nbrlane), :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for id, nbrdis in enumerate(neighborsdistance):
                if len(nbrdis) != 0:
                    nbrsdis_batch[count3, 0:len(nbrdis), :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for id, nbrclass in enumerate(neighborsclass):
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
            # "temporal_mask": temporal_mask_batch,
        }

    def compute_stats(self, save_path=None, sample_limit=None):
        print("computing dataset statistics...")

        pos_sum = np.zeros(2, dtype=np.float64)
        pos_sq_sum = np.zeros(2, dtype=np.float64)
        va_sum = np.zeros(2, dtype=np.float64)
        va_sq_sum = np.zeros(2, dtype=np.float64)
        lane_sum = np.zeros(1, dtype=np.float64)
        lane_sq_sum = np.zeros(1, dtype=np.float64)
        class_sum = np.zeros(1, dtype=np.float64)
        class_sq_sum = np.zeros(1, dtype=np.float64)
        pos_count = va_count = lane_count = class_count = 0

        def ensure_col(arr):
            if arr.size == 0:
                return np.empty((0, 1))
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        n = len(self.D) if sample_limit is None else min(len(self.D), sample_limit)

        for idx in range(n):

            dsId = self.D[idx, 0].astype(int)
            vehId = self.D[idx, 1].astype(int)
            t = self.D[idx, 2]
            grid = self.D[idx, 11:]

            # ego hist / fut / va
            try:
                hist = self.getHistory(vehId, t, vehId, dsId)
            except Exception:
                hist = np.empty((0, 2))
            try:
                fut = self.getFuture(vehId, t, dsId)
            except Exception:
                fut = np.empty((0, 2))
            try:
                va = self.getVA(vehId, t, vehId, dsId)
            except Exception:
                va = np.empty((0, 2))
            try:
                lane = ensure_col(self.getLane(vehId, t, vehId, dsId))
            except Exception:
                lane = np.empty((0, 1))
            try:
                cclass = ensure_col(self.getClass(vehId, t, vehId, dsId))
            except Exception:
                cclass = np.empty((0, 1))

            for arr in (hist, fut):
                if arr.size != 0:
                    pos_sum += arr.sum(axis=0)
                    pos_sq_sum += (arr ** 2).sum(axis=0)
                    pos_count += arr.shape[0]
            if va.size != 0:
                va_sum += va.sum(axis=0)
                va_sq_sum += (va ** 2).sum(axis=0)
                va_count += va.shape[0]
            if lane.size != 0:
                lane_sum += lane.sum(axis=0)
                lane_sq_sum += (lane ** 2).sum(axis=0)
                lane_count += lane.shape[0]
            if cclass.size != 0:
                class_sum += cclass.sum(axis=0)
                class_sq_sum += (cclass ** 2).sum(axis=0)
                class_count += cclass.shape[0]

            for v_id in grid:
                vid = int(v_id)
                if vid == 0:
                    continue
                try:
                    n_hist = self.getHistory(vid, t, vehId, dsId)
                except Exception:
                    n_hist = np.empty((0, 2))
                try:
                    n_va = self.getVA(vid, t, vehId, dsId)
                except Exception:
                    n_va = np.empty((0, 2))
                try:
                    n_lane = ensure_col(self.getLane(vid, t, vehId, dsId))
                except Exception:
                    n_lane = np.empty((0, 1))
                try:
                    n_class = ensure_col(self.getClass(vid, t, vehId, dsId))
                except Exception:
                    n_class = np.empty((0, 1))

                if n_hist.size != 0:
                    pos_sum += n_hist.sum(axis=0)
                    pos_sq_sum += (n_hist ** 2).sum(axis=0)
                    pos_count += n_hist.shape[0]
                if n_va.size != 0:
                    va_sum += n_va.sum(axis=0)
                    va_sq_sum += (n_va ** 2).sum(axis=0)
                    va_count += n_va.shape[0]
                if n_lane.size != 0:
                    lane_sum += n_lane.sum(axis=0)
                    lane_sq_sum += (n_lane ** 2).sum(axis=0)
                    lane_count += n_lane.shape[0]
                if n_class.size != 0:
                    class_sum += n_class.sum(axis=0)
                    class_sq_sum += (n_class ** 2).sum(axis=0)
                    class_count += n_class.shape[0]

        if pos_count > 0:
            self.pos_mean = pos_sum / pos_count
            pos_var = (pos_sq_sum / pos_count) - (self.pos_mean ** 2)
            self.pos_std = np.sqrt(np.maximum(pos_var, 1e-6))
        if va_count > 0:
            self.va_mean = va_sum / va_count
            va_var = (va_sq_sum / va_count) - (self.va_mean ** 2)
            self.va_std = np.sqrt(np.maximum(va_var, 1e-6))
        if lane_count > 0:
            self.lane_mean = lane_sum / lane_count
            lane_var = (lane_sq_sum / lane_count) - (self.lane_mean ** 2)
            self.lane_std = np.sqrt(np.maximum(lane_var, 1e-6))
        if class_count > 0:
            self.class_mean = class_sum / class_count
            class_var = (class_sq_sum / class_count) - (self.class_mean ** 2)
            self.class_std = np.sqrt(np.maximum(class_var, 1e-6))

        save_path = save_path if save_path is not None else self.stats_file
        np.savez(
            save_path,
            pos_mean=self.pos_mean, pos_std=self.pos_std,
            va_mean=self.va_mean, va_std=self.va_std,
            lane_mean=self.lane_mean, lane_std=self.lane_std,
            class_mean=self.class_mean, class_std=self.class_std,
        )
        print("Computed stats saved to", save_path)

    def get_stats(self):
        return {
            "pos_mean": self.pos_mean,
            "pos_std": self.pos_std,
            "va_mean": self.va_mean,
            "va_std": self.va_std,
            "lane_mean": self.lane_mean,
            "lane_std": self.lane_std,
            "class_mean": self.class_mean,
            "class_std": self.class_std,
        }
