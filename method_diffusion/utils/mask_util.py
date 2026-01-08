import numpy as np
import torch

# 轨迹随机掩码，保留比例 p， True表示保留
def random_mask(traj, p=0.4):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape

    mask = torch.rand((B, T, 1), device=traj.device) < p
    if T > 0:
        mask[:, -1, :] = True
    return mask

# 轨迹连续掩码， 丢弃比例为 p， True表示保留
def continuous_mask(traj, p=0.4):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape
    block_len = max(0, min(T - 1, int(round(T * p))))

    mask = torch.ones((B, T), dtype=torch.bool, device=traj.device)
    if block_len > 0:
        max_start = T - 1 - block_len
        starts = torch.randint(0, max_start + 1, (B,), device=traj.device)

        positions = torch.arange(T, device=traj.device).unsqueeze(0)
        starts = starts.unsqueeze(1)

        missing = (positions >= starts) & (positions < starts + block_len)
        mask = ~missing

    return mask.unsqueeze(-1)

# print(continuous_mask(torch.ones(5, 10, 2), p=0.4))
# print(random_mask(torch.ones(2, 10, 2), p=0.4))