import numpy as np
import torch

# 轨迹随机掩码，p 表示初始每帧缺失概率，True 表示保留；非空历史至少保留一帧。
def random_mask(traj, p=0.4):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape
    if T == 0:
        raise ValueError("History trajectories must contain at least one time step.")
    p = float(max(0.0, min(1.0, p)))

    draws = torch.rand((B, T, 1), device=traj.device)
    mask = draws >= p
    if T > 0:
        # 全缺失时保留随机分数最高的一帧，不固定当前帧，也不消耗额外随机数。
        keep_positions = draws[..., 0].argmax(dim=1)
        rows = torch.arange(B, device=traj.device)
        mask[rows, keep_positions, 0] |= ~mask.any(dim=1).squeeze(-1)
    return mask
