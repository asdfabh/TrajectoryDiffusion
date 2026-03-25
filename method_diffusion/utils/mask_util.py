import numpy as np
import torch

# 轨迹随机掩码，p 表示掩码比例，True 表示保留
def random_mask(traj, p=0.4):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape
    p = float(max(0.0, min(1.0, p)))

    mask = torch.rand((B, T, 1), device=traj.device) >= p
    if T > 0:
        mask[:, -1, :] = True
    return mask

# 轨迹连续掩码，p 表示掩码比例，True 表示保留
def continuous_mask(traj, p=0.4, start_flag=False):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape
    p = float(max(0.0, min(1.0, p)))
    block_len = max(0, min(T - 1, int(round(T * p))))

    mask = torch.ones((B, T), dtype=torch.bool, device=traj.device)
    if block_len > 0:
        if start_flag:
            starts = torch.zeros((B,), dtype=torch.long, device=traj.device)
        else:
            max_start = T - 1 - block_len
            starts = torch.randint(0, max_start + 1, (B,), device=traj.device)

        positions = torch.arange(T, device=traj.device).unsqueeze(0)
        starts = starts.unsqueeze(1)

        missing = (positions >= starts) & (positions < starts + block_len)
        mask = ~missing

    return mask.unsqueeze(-1)


def mixed_mask(traj, p=0.4, random_ratio=0.7, block_start=False):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B = traj.shape[0]
    random_ratio = float(max(0.0, min(1.0, random_ratio)))

    random_hist_mask = random_mask(traj, p=p)
    block_hist_mask = continuous_mask(traj, p=p, start_flag=block_start)

    if random_ratio <= 0.0:
        return block_hist_mask
    if random_ratio >= 1.0:
        return random_hist_mask

    selector = (torch.rand((B, 1, 1), device=traj.device) < random_ratio)
    return torch.where(selector, random_hist_mask, block_hist_mask)

# print(continuous_mask(torch.ones(5, 10, 2), p=0.4))
# print(random_mask(torch.ones(2, 10, 2), p=0.4))
