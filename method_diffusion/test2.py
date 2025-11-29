import numpy as np
import os

from dataset.ngsim_dataset import NgsimDataset
from utils.util import random_mask_traj, random_prefix_keep_traj, plot_traj, mask_traj
from torch.utils.data import DataLoader


def test(batch_size=1, row=3, col=3):
    for j in range(batch_size):
        hists = []
        nbrs = []
        for i in range(row):
            idx = np.random.randint(0, len(test_dataset))
            sample = test_dataset[idx]
            hist, fut, neighbors, lat_enc, lon_enc, va, neighborsva, lane, \
                neighborslane, refdistance, neighborsdistance, cclass, negihborsclass, nbrs_num = sample

            nbrs_ = []
            for i in range(len(neighbors)):
                nbr = neighbors[i]
                if nbr.shape[0] == 0:
                    continue
                nbrs_.append(nbr)
            hists.append(hist)
            nbrs.append(nbrs_)

            mask1 = random_mask_traj(hist, p=0.4)
            hist_masked1 = mask_traj(hist, mask1)
            hists.append(hist_masked1)
            nbrs_masked1 = []
            for i in range(len(nbrs_)):
                nbr = nbrs_[i]
                mask1 = random_mask_traj(nbr, p=0.4)
                nbr_masked = mask_traj(nbr, mask1)
                nbrs_masked1.append(nbr_masked)
            nbrs.append(nbrs_masked1)

            mask2 = random_prefix_keep_traj(hist, p=0.6)
            hist_masked2 = mask_traj(hist, mask2)
            hists.append(hist_masked2)
            nbrs_masked2 = []
            for i in range(len(nbrs_)):
                nbr = nbrs_[i]
                mask2 = random_prefix_keep_traj(nbr, p=0.6)
                nbr_masked = mask_traj(nbr, mask2)
                nbrs_masked2.append(nbr_masked)
            nbrs.append(nbrs_masked2)
            print(f"len of hists: {len(hists)}, len of nbrs_: {len(nbrs_)}, len of nbrs_masked1: {len(nbrs_masked1)}, len of nbrs_masked2: {len(nbrs_masked2)}")
        plot_traj(hists, nbrs=nbrs, fig_num1=row, fig_num2=col, is_compare=True)
        hists = []
        nbrs = []

# ---------------- 新增：归一化/反归一化验证 ----------------

def arr_str(a: np.ndarray) -> str:
    """Helper: format 1D array as space-separated string with 4 decimals."""
    return " ".join(f"{v:.4f}" for v in a.tolist())


def test_normalization(
    dataset: NgsimDataset,
    num_batch_samples: int = 4,
    num_samples_to_check: int = 5,
):
    """对比：原始 hist/fut vs 归一化 vs 反归一化，打印完整轨迹并可视化。

    Args:
        dataset: 已经配置好 normalization 的 NgsimDataset（normalize=True）。
        num_batch_samples: DataLoader 中一次取多少样本（用于触发 collate_fn）。
        num_samples_to_check: 从数据集中随机抽取多少个样本做数值和可视化检查。
    """
    # 先用 DataLoader 触发一次 collate_fn，主要为了确认批量维度/归一化行为正常
    loader = DataLoader(dataset, batch_size=num_batch_samples, shuffle=False, collate_fn=dataset.collate_fn)
    batch = next(iter(loader))
    (
        hist_batch,
        nbrs_batch,
        mask_batch,
        lat_enc_batch,
        lon_enc_batch,
        fut_batch,
        op_mask_batch,
        va_batch,
        nbrsva_batch,
        lane_batch,
        nbrslane_batch,
        distance_batch,
        nbrsdis_batch,
        class_batch,
        nbrsclass_batch,
        map_position,
        nbrs_num_batch,
        temporal_mask_batch,
    ) = batch

    print("DataLoader batch shapes:")
    print("  hist_batch:", hist_batch.shape)
    print("  fut_batch:", fut_batch.shape)

    # 随机抽样若干个样本索引做详细检查
    n_total = len(dataset)
    num_samples_to_check = min(num_samples_to_check, n_total)
    indices = np.random.choice(n_total, size=num_samples_to_check, replace=False)

    for idx_i, idx in enumerate(indices):
        print("\n================ Sample {} (dataset idx = {}) ================".format(idx_i, idx))
        # 从 __getitem__ 取得原始（未归一化）的数据
        raw_sample = dataset[idx]
        raw_hist, raw_fut, raw_neighbors, *_ = raw_sample

        raw_hist_np = raw_hist.astype(np.float32)
        raw_fut_np = raw_fut.astype(np.float32)

        if dataset.norm_method == 'fixed':
            scale = float(dataset.fixed_scale)
            hist_norm = raw_hist_np / scale
            hist_denorm = hist_norm * scale
            fut_norm = raw_fut_np / scale
            fut_denorm = fut_norm * scale
        else:  # zscore
            pos_mean = dataset.pos_mean.astype(np.float32)
            pos_std = dataset.pos_std.astype(np.float32)
            pos_std[pos_std == 0] = 1.0
            hist_norm = (raw_hist_np - pos_mean) / pos_std
            hist_denorm = hist_norm * pos_std + pos_mean
            fut_norm = (raw_fut_np - pos_mean) / pos_std
            fut_denorm = fut_norm * pos_std + pos_mean

        # 数值误差（hist 和 fut 分开统计）
        hist_diff = hist_denorm - raw_hist_np
        fut_diff = fut_denorm - raw_fut_np
        print("pos_mean:", dataset.pos_mean, "pos_std:", dataset.pos_std)
        print(f"HIST max abs error: {np.abs(hist_diff).max():.6f}, mean abs error: {np.abs(hist_diff).mean():.6f}")
        if raw_fut_np.size > 0:
            print(f"FUT  max abs error: {np.abs(fut_diff).max():.6f}, mean abs error: {np.abs(fut_diff).mean():.6f}")
        else:
            print("FUT  is empty for this sample.")

        # --------- 横向打印完整 hist 轨迹 ---------
        print("[HIST] length =", raw_hist_np.shape[0])
        print("原始X:", arr_str(raw_hist_np[:, 0]))
        print("原始Y:", arr_str(raw_hist_np[:, 1]))
        print("归一化X:", arr_str(hist_norm[:, 0]))
        print("归一化Y:", arr_str(hist_norm[:, 1]))
        print("反归一化X:", arr_str(hist_denorm[:, 0]))
        print("反归一化Y:", arr_str(hist_denorm[:, 1]))

        # --------- 横向打印完整 fut 轨迹（如果非空） ---------
        print("[FUT ] length =", raw_fut_np.shape[0])
        if raw_fut_np.size > 0:
            print("原始X:", arr_str(raw_fut_np[:, 0]))
            print("原始Y:", arr_str(raw_fut_np[:, 1]))
            print("归一化X:", arr_str(fut_norm[:, 0]))
            print("归一化Y:", arr_str(fut_norm[:, 1]))
            print("反归一化X:", arr_str(fut_denorm[:, 0]))
            print("反归一化Y:", arr_str(fut_denorm[:, 1]))
        else:
            print("FUT is empty, skip printing values.")

        # --------- 可视化：hist + fut ---------
        # 对于可视化，我们构建三组：原始 / 归一化 / 反归一化，每组用 (hist, fut) 拼接
        # 这里仍然沿用 plot_traj 接口：传入 hists 列表，每个元素是一条完整轨迹
        # 我们简单地画历史 + 未来拼接在一起的轨迹
        def concat_hist_fut(h: np.ndarray, f: np.ndarray) -> np.ndarray:
            if f.size == 0:
                return h
            return np.concatenate([h, f], axis=0)

        full_raw = concat_hist_fut(raw_hist_np, raw_fut_np)
        full_norm = concat_hist_fut(hist_norm, fut_norm)
        full_denorm = concat_hist_fut(hist_denorm, fut_denorm)

        hists = [full_raw, full_norm, full_denorm]

        # 邻居简单画原始邻居的历史+未来（这里只用历史，和之前保持一致）
        nbrs = []
        nbrs_single = []
        for n in raw_neighbors:
            if n.shape[0] == 0:
                continue
            nbrs_single.append(n)
        nbrs.append(nbrs_single)  # 对应 hists[0]
        nbrs.append(nbrs_single)  # 对应 hists[1]
        nbrs.append(nbrs_single)  # 对应 hists[2]

        plot_traj(hists, nbrs=nbrs, fig_num1=3, fig_num2=1, is_compare=True)
        print("Visualization done for this sample: row1=raw, row2=normalized, row3=denormalized.")


# ---------------- 原有小测试 ----------------

hist = np.random.rand(16, 2)
mask = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
traj = mask_traj(hist, mask)
hists = [hist, traj]
# plot_traj(hists, fig_num1=2, fig_num2=1)


test_dataset_path = os.path.join('../data/ngsimdata/TestSet.mat')
test_dataset = NgsimDataset(test_dataset_path)

test_dataset.compute_stats()
print(test_dataset.get_stats())



