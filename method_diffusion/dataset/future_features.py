import numpy as np


TRACK_DT = 0.1


def _build_delta_sequence(abs_positions, ref_position):
    if abs_positions.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    prev_positions = np.concatenate(
        [np.asarray(ref_position, dtype=np.float32).reshape(1, 2), abs_positions[:-1]],
        axis=0,
    )
    return (abs_positions - prev_positions).astype(np.float32, copy=False)


def derive_theta_from_positions(abs_positions, ref_position):
    delta = _build_delta_sequence(abs_positions, ref_position)
    if delta.size == 0:
        return np.empty((0, 1), dtype=np.float32)

    theta = np.arctan2(delta[:, 1], delta[:, 0]).astype(np.float32, copy=False)
    return theta.reshape(-1, 1)


def derive_speed_from_positions(abs_positions, ref_position, step_dt):
    delta = _build_delta_sequence(abs_positions, ref_position)
    if delta.size == 0:
        return np.empty((0, 1), dtype=np.float32)

    step_dt = max(float(step_dt), 1e-6)
    speed = np.linalg.norm(delta, axis=1, keepdims=True) / step_dt
    return speed.astype(np.float32, copy=False)


def build_future_xy_theta_v(track, frame_idx, t_f, d_s):
    ref_position = track[frame_idx, 1:3].astype(np.float32, copy=False)
    start_idx = frame_idx + int(d_s)
    end_idx = min(len(track), frame_idx + int(t_f) + 1)
    future_abs = track[start_idx:end_idx:int(d_s), 1:3].astype(np.float32, copy=False)
    if len(future_abs) == 0:
        return np.empty((0, 4), dtype=np.float32)

    future_xy = future_abs - ref_position
    theta = derive_theta_from_positions(future_abs, ref_position)
    future_v = track[start_idx:end_idx:int(d_s), 3:4].astype(np.float32, copy=False)
    return np.concatenate([future_xy, theta, future_v], axis=1).astype(np.float32, copy=False)


def build_anchor_xy_theta_v(anchor_xy, d_s):
    anchor_xy = np.asarray(anchor_xy, dtype=np.float32)
    if anchor_xy.ndim != 3 or anchor_xy.shape[-1] != 2:
        raise ValueError(f"Expected anchor_xy with shape [K, T, 2], got {anchor_xy.shape}")

    step_dt = TRACK_DT * int(d_s)
    anchor_aug = []
    zero_ref = np.zeros(2, dtype=np.float32)
    for traj_xy in anchor_xy:
        theta = derive_theta_from_positions(traj_xy, zero_ref)
        speed = derive_speed_from_positions(traj_xy, zero_ref, step_dt)
        anchor_aug.append(np.concatenate([traj_xy, theta, speed], axis=1))
    return np.stack(anchor_aug, axis=0).astype(np.float32, copy=False)
