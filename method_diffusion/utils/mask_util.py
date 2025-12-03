import numpy as np
import torch

def apply_mask_keep_length(traj, mask, fill_value=0):

    traj = traj.numpy() if isinstance(traj, torch.Tensor) else traj
    mask = mask.numpy() if isinstance(mask, torch.Tensor) else mask

    mask = mask.astype(bool)
    masked_traj = traj.copy()
    masked_traj[~mask] = fill_value
    return masked_traj


def random_mask_traj(traj, p=0.4):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    if traj.dim() == 2:
        T = traj.shape[0]
        if T == 0:
            return torch.empty(0, dtype=torch.bool, device=traj.device)
        mask = torch.rand(T, device=traj.device) < p
    else:  # [B, T, 2]
        B, T, _ = traj.shape
        mask = torch.rand(B, T, device=traj.device) < p

    return mask

def random_prefix_keep_traj(traj, p=0.6):
    T = traj.shape[0]
    if T == 0:
        return np.array([], dtype=bool)

    keep_len = np.random.randint(1, T * p)
    mask = np.zeros(T, dtype=bool)
    mask[T - keep_len:] = 1
    # print(f"keep_len: {keep_len}, mask = {mask}")
    return mask

def block_mask_traj(traj, missing_ratio=0.3):
    """支持单条轨迹 [T, 2]"""
    T = traj.shape[0]
    block_len = max(1, min(T, int(round(T * missing_ratio))))
    start = torch.randint(0, T - block_len + 1, (1,), device=traj.device).item()

    mask = torch.ones(T, dtype=torch.bool, device=traj.device)
    mask[start:start + block_len] = False
    return mask