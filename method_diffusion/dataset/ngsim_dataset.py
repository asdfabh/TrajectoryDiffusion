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

    def __init__(self, mat_file, t_h=30, t_f=50, d_s=2, enc_size=64, grid_size=(13, 3),
                 max_neighbors=10, neighbor_radius=100.0):
        self.D = scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = t_h  #
        self.t_f = t_f  #
        self.d_s = d_s  # skip
        self.enc_size = enc_size  # size of the grid cell
        self.grid_size = grid_size  # size of social context grid

        self.max_neighbors = max_neighbors  # [修改] 最大邻居数量
        self.neighbor_radius = neighbor_radius

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
        dsId = self.D[idx, 0].astype(int)
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]

        # [修改] 获取所有候选邻居 ID (从原有的 grid 列中获取，去重)
        grid_candidates = self.D[idx, 11:].astype(int)
        unique_candidates = np.unique(grid_candidates)
        unique_candidates = unique_candidates[unique_candidates != 0]  # 去除 0

        # 1. 筛选和计算距离
        valid_neighbors = []
        hist = self.getHistory(vehId, t, vehId, dsId)  # 自车历史

        for nbr_id in unique_candidates:
            # 获取邻居相对于自车的历史轨迹
            nbr_hist = self.getHistory(nbr_id, t, vehId, dsId)

            if nbr_hist.shape[0] > 0:
                # 计算当前时刻 t (轨迹的最后一点) 的距离
                current_pos = nbr_hist[-1]
                dist = np.linalg.norm(current_pos)

                if dist <= self.neighbor_radius:
                    valid_neighbors.append({
                        'id': nbr_id,
                        'dist': dist,
                        'hist': nbr_hist
                    })

        # 2. 按距离排序并截取前 K 个
        valid_neighbors.sort(key=lambda x: x['dist'])
        selected_neighbors = valid_neighbors[:self.max_neighbors]

        # 3. 构建输出数据
        neighbors = []
        neighborsva = []
        neighborslane = []
        neighborsclass = []
        neighborsdistance = []
        neighborsfut = []

        refdistance = np.zeros_like(hist[:, 0]).reshape(-1, 1)
        fut = self.getFuture(vehId, t, dsId)
        va = self.getVA(vehId, t, vehId, dsId)
        lane = self.getLane(vehId, t, vehId, dsId)
        cclass = self.getClass(vehId, t, vehId, dsId)

        for item in selected_neighbors:
            nbr_id = item['id']
            nbr_hist = item['hist']

            # 计算历史距离序列 (用于特征)
            uu = np.power(hist - nbr_hist, 2)
            distancexxx = np.sqrt(uu[:, 0] + uu[:, 1]).reshape(-1, 1)

            neighbors.append(nbr_hist)
            neighborsdistance.append(distancexxx)
            neighborsva.append(self.getVA(nbr_id, t, vehId, dsId))
            neighborslane.append(self.getLane(nbr_id, t, vehId, dsId).reshape(-1, 1))
            neighborsclass.append(self.getClass(nbr_id, t, vehId, dsId).reshape(-1, 1))
            neighborsfut.append(self.getFuture(nbr_id, t, dsId, vehId))

        # 独热编码部分保持不变
        lon_enc = np.zeros([3])
        lon_enc[int(self.D[idx, 10] - 1)] = 1
        lat_enc = np.zeros([3])
        lat_enc[int(self.D[idx, 9] - 1)] = 1
        nbrs_num = np.array(len(neighbors))

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
            "neighborsfut": neighborsfut,
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
    def getFuture(self, vehId, t, dsId, refVehId=None):
        if refVehId is None:
            refVehId = vehId

        if vehId == 0:
            return np.empty([0, 2])

        if self.T.shape[1] <= vehId - 1:
            return np.empty([0, 2])

        refTrack = self.T[dsId - 1][refVehId - 1].transpose()
        vehTrack = self.T[dsId - 1][vehId - 1].transpose()

        ref_idx = np.where(refTrack[:, 0] == t)
        if ref_idx[0].size == 0:
            return np.empty([0, 2])
        refPos = refTrack[ref_idx][0, 1:3]

        veh_t_idx = np.argwhere(vehTrack[:, 0] == t)
        if vehTrack.size == 0 or veh_t_idx.size == 0:
            return np.empty([0, 2])

        stpt = veh_t_idx.item() + self.d_s
        enpt = np.minimum(len(vehTrack), veh_t_idx.item() + self.t_f + 1)

        if stpt >= len(vehTrack):
            return np.empty([0, 2])

        fut = vehTrack[stpt:enpt:self.d_s, 1:3] - refPos
        return fut

    # Collate function for dataloader
    def collate_fn(self, samples):
        nbr_batch_size = 0
        for sample in samples:
            nbrs = sample["neighbors"]
            nbr_batch_size += len(nbrs)  # 直接统计有效邻居总数

        maxlen = self.t_h // self.d_s + 1

        # 初始化 Batch Tensor
        nbrs_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrsva_batch = torch.zeros(nbr_batch_size, maxlen, 2)
        nbrslane_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsclass_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsdis_batch = torch.zeros(nbr_batch_size, maxlen, 1)
        nbrsfut_batch = torch.zeros(nbr_batch_size, self.t_f // self.d_s, 2)

        # Mask 初始化
        hist_valid_mask = torch.zeros(len(samples), maxlen, 1, dtype=torch.bool)
        fut_valid_mask = torch.zeros(len(samples), self.t_f // self.d_s, 1, dtype=torch.bool)
        va_valid_mask = torch.zeros(len(samples), maxlen, 1, dtype=torch.bool)
        nbrs_valid_mask = torch.zeros(nbr_batch_size, maxlen, 1, dtype=torch.bool)
        nbrsva_valid_mask = torch.zeros(nbr_batch_size, maxlen, 1, dtype=torch.bool)

        # [修改] Mask Batch 不再是 Grid 形状，而是 [Batch, Max_Neighbors]
        # 这里的 mask 表示第 i 个邻居槽位是否有车
        mask_batch = torch.zeros(len(samples), self.max_neighbors).bool()
        map_position = torch.zeros(0, 2)  # 仅用于兼容，实际意义变弱

        # Ego 相关 Batch 初始化
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

        count = 0
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0

        for sampleId, sample in enumerate(samples):
            hist = sample["hist"]
            fut = sample["fut"]
            lat_enc = sample["lat_enc"]
            lon_enc = sample["lon_enc"]
            va = sample["va"]
            lane = sample["lane"]
            refdistance = sample["refdistance"]
            cclass = sample["cclass"]
            nbrs_num = sample["nbrs_num"]

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

            neighbors = sample["neighbors"]
            neighborsva = sample["neighborsva"]
            neighborslane = sample["neighborslane"]
            neighborsdistance = sample["neighborsdistance"]
            neighborsclass = sample["neighborsclass"]
            neighborsfut = sample["neighborsfut"]

            # [修改] 邻居填充逻辑：按顺序填充到 0 ~ K-1 的槽位中
            for i, nbr in enumerate(neighbors):
                if i >= self.max_neighbors: break  # 理论上 getitem 已截断，双重保险

                # 标记该样本的第 i 个邻居槽位有效
                mask_batch[sampleId, i] = True

                # 记录位置映射 (Rank, 0) - 仅作兼容
                map_position = torch.cat((map_position, torch.tensor([[i, 0]])), 0)

                # 填充具体的轨迹数据到扁平化的 batch 中
                nbr_len = len(nbr)
                nbrs_batch[count, 0:nbr_len, 0] = torch.from_numpy(nbr[:, 0])
                nbrs_batch[count, 0:nbr_len, 1] = torch.from_numpy(nbr[:, 1])
                nbrs_valid_mask[count, :nbr_len, 0] = True

                nbr_fut = neighborsfut[i]
                if len(nbr_fut) != 0:
                    nbr_fut_len = len(nbr_fut)
                    nbrsfut_batch[count, 0:nbr_fut_len, 0] = torch.from_numpy(nbr_fut[:, 0])
                    nbrsfut_batch[count, 0:nbr_fut_len, 1] = torch.from_numpy(nbr_fut[:, 1])

                count += 1

            # 填充其他属性 (VA, Lane, Distance, Class)
            # 逻辑同上，只是遍历对应的列表
            for nbrva in neighborsva:
                if len(nbrva) != 0:
                    nbrva_len = len(nbrva)
                    nbrsva_batch[count1, 0:nbrva_len, 0] = torch.from_numpy(nbrva[:, 0])
                    nbrsva_batch[count1, 0:nbrva_len, 1] = torch.from_numpy(nbrva[:, 1])
                    nbrsva_valid_mask[count1, :nbrva_len, 0] = True
                    count1 += 1

            for nbrlane in neighborslane:
                if len(nbrlane) != 0:
                    nbrslane_batch[count2, 0:len(nbrlane), :] = torch.from_numpy(nbrlane)
                    count2 += 1

            for nbrdis in neighborsdistance:
                if len(nbrdis) != 0:
                    nbrsdis_batch[count3, 0:len(nbrdis), :] = torch.from_numpy(nbrdis)
                    count3 += 1

            for nbrclass in neighborsclass:
                if len(nbrclass) != 0:
                    nbrsclass_batch[count4, 0:len(nbrclass), :] = torch.from_numpy(nbrclass)
                    count4 += 1

        return {
            "hist": hist_batch,
            "nbrs": nbrs_batch,
            "mask": mask_batch,  # [B, max_neighbors]
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
            "nbrs_fut": nbrsfut_batch,
        }

    def compute_stats(self, save_path=None, sample_limit=None):
        """
        计算数据集的归一化参数 (Mean, Std)。
        关键修正：统计范围包含了 Ego 和 Neighbors 的【未来轨迹】，确保预测目标的分布也被正确归一化。
        """
        print("Computing dataset statistics...")

        pos_sum = np.zeros(2, dtype=np.float64)
        pos_sq_sum = np.zeros(2, dtype=np.float64)
        va_sum = np.zeros(2, dtype=np.float64)
        va_sq_sum = np.zeros(2, dtype=np.float64)

        # 计数器
        pos_count = 0
        va_count = 0

        # 辅助函数：确保是列向量
        def ensure_col(arr):
            if arr.size == 0:
                return np.empty((0, 1))
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        # 限制采样数用于快速调试，正式训练设为 None
        n = len(self.D) if sample_limit is None else min(len(self.D), sample_limit)

        for idx in range(n):
            dsId = self.D[idx, 0].astype(int)
            vehId = self.D[idx, 1].astype(int)
            t = self.D[idx, 2]
            grid = self.D[idx, 11:]

            # 1. 获取 Ego 数据
            # 注意：getHistory 和 getFuture 必须传入 refVehId=vehId 以获取相对坐标
            try:
                hist = self.getHistory(vehId, t, vehId, dsId)
            except:
                hist = np.empty((0, 2))

            try:
                fut = self.getFuture(vehId, t, dsId, refVehId=vehId)
            except:
                fut = np.empty((0, 2))

            try:
                va = self.getVA(vehId, t, vehId, dsId)
            except:
                va = np.empty((0, 2))

            # 累加 Ego 的位置 (History + Future)
            for arr in (hist, fut):
                if arr.size > 0:
                    pos_sum += arr.sum(axis=0)
                    pos_sq_sum += (arr ** 2).sum(axis=0)
                    pos_count += arr.shape[0]

            # 累加 Ego 的速度/加速度
            if va.size > 0:
                va_sum += va.sum(axis=0)
                va_sq_sum += (va ** 2).sum(axis=0)
                va_count += va.shape[0]

            # 2. 获取 Neighbors 数据
            for v_id in grid:
                vid = int(v_id)
                if vid == 0:
                    continue

                try:
                    n_hist = self.getHistory(vid, t, vehId, dsId)
                except:
                    n_hist = np.empty((0, 2))

                try:
                    # [重要] 获取 Neighbor 未来轨迹用于统计
                    n_fut = self.getFuture(vid, t, dsId, refVehId=vehId)
                except:
                    n_fut = np.empty((0, 2))

                try:
                    n_va = self.getVA(vid, t, vehId, dsId)
                except:
                    n_va = np.empty((0, 2))

                # 累加 Neighbor 的位置 (History + Future)
                for arr in (n_hist, n_fut):
                    if arr.size > 0:
                        pos_sum += arr.sum(axis=0)
                        pos_sq_sum += (arr ** 2).sum(axis=0)
                        pos_count += arr.shape[0]

                if n_va.size > 0:
                    va_sum += n_va.sum(axis=0)
                    va_sq_sum += (n_va ** 2).sum(axis=0)
                    va_count += n_va.shape[0]

        # 计算 Mean 和 Std
        # Std = sqrt( E[x^2] - (E[x])^2 )
        if pos_count > 0:
            self.pos_mean = pos_sum / pos_count
            pos_var = (pos_sq_sum / pos_count) - (self.pos_mean ** 2)
            self.pos_std = np.sqrt(np.maximum(pos_var, 1e-6))  # 避免除零

        if va_count > 0:
            self.va_mean = va_sum / va_count
            va_var = (va_sq_sum / va_count) - (self.va_mean ** 2)
            self.va_std = np.sqrt(np.maximum(va_var, 1e-6))

        # 保存
        save_path = save_path if save_path is not None else self.stats_file
        np.savez(
            save_path,
            pos_mean=self.pos_mean, pos_std=self.pos_std,
            va_mean=self.va_mean, va_std=self.va_std,
            # Lane 和 Class 通常不需要归一化，或者使用固定值，这里略过更新
            lane_mean=self.lane_mean, lane_std=self.lane_std,
            class_mean=self.class_mean, class_std=self.class_std,
        )
        print(f"Computed stats saved to {save_path}")
        print(f"Pos Mean: {self.pos_mean}, Pos Std: {self.pos_std}")

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