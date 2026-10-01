import numpy as np
import torch


def random_mask(traj, p=0.4):
    """逐点以概率 p 缺失，True 表示保留；每条轨迹至少保留一个点。

    所有时间步（包括当前状态）均参与独立随机采样。若全部缺失，
    则均匀随机恢复一个时间步，因此最多掩码 T-1 个点。
    """
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)

    B, T, _ = traj.shape
    if T == 0:
        raise ValueError("History trajectories must contain at least one time step.")
    p = float(max(0.0, min(1.0, p)))

    mask = torch.rand((B, T, 1), device=traj.device) >= p
    missing_rows = (~mask.any(dim=1).squeeze(-1)).nonzero(as_tuple=True)[0]
    retained_steps = torch.randint(T, (missing_rows.numel(),), device=traj.device)
    mask[missing_rows, retained_steps, 0] = True
    return mask
