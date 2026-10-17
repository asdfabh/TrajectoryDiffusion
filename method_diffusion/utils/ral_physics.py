"""SG motion evaluation and jerk P95 regularization for staged training."""
import json
import hashlib
import math
from pathlib import Path

import torch
import torch.distributed as dist

from method_diffusion.utils.physics_metrics import (
    PHYSICS_QUANTILE_SAMPLE_SIZE, PHYSICS_THRESHOLD_QUANTILE,
    MotionStatistics, load_physics_thresholds, print_motion_statistics,
    resolve_physics_threshold_file,
)
from method_diffusion.utils.sg_kinematics import MOTION_SPEED_MIN, MOTION_WINDOW, compute_future_motion, get_motion_method


def evaluation_collectors(args, dt, refined=False):
    dataset = getattr(args, "dataset", None)
    window = MOTION_WINDOW
    path = resolve_physics_threshold_file(dataset, window_length=window) if dataset else None
    thresholds = None
    if path is not None:
        thresholds, _ = load_physics_thresholds(path, dataset, dt, PHYSICS_THRESHOLD_QUANTILE,
                                               MOTION_SPEED_MIN, window)
    settings = dict(dt=dt, thresholds=thresholds, speed_min=MOTION_SPEED_MIN,
                    sample_size=PHYSICS_QUANTILE_SAMPLE_SIZE,
                    seed=getattr(args, "seed", 2026) + (1000 * dist.get_rank() if dist.is_available() and dist.is_initialized() else 0),
                    window_length=window)
    args._physics_evaluation = {
        **get_motion_method(window), "dt": dt, "candidate_scope": "all",
        "threshold_file": str(path) if path is not None else None,
        "threshold_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path is not None else None,
        "threshold_quantile": PHYSICS_THRESHOLD_QUANTILE if path is not None else None,
        "speed_min": MOTION_SPEED_MIN, "quantile_sample_size": PHYSICS_QUANTILE_SAMPLE_SIZE,
        "thresholds": thresholds,
    }
    names = ("proposal", "refined", "ground_truth") if refined else ("proposal", "ground_truth")
    return {name: MotionStatistics(**settings) for name in names}


def finish_physics(collectors, distributed=False, print_results=True):
    if distributed and dist.is_available() and dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, collectors)
        collectors = parts[0]
        for part in parts[1:]:
            for key in collectors:
                collectors[key].merge(part[key])
    result = {key: value.summary() for key, value in collectors.items()}
    if print_results:
        for key, summary in result.items():
            print_motion_statistics(summary, f"SG motion / {key}")
    return result


def save_evaluation(path, baseline, refined=None, metadata=None):
    if path is None:
        return
    def convert(value):
        if isinstance(value, torch.Tensor):
            return convert(value.detach().cpu().tolist())
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(convert({"metadata": metadata or {}, "proposal": baseline,
                                        "refined": refined}), ensure_ascii=False, indent=2,
                                allow_nan=False) + "\n", encoding="utf-8")


def configure_refiner_jerk(args, dt):
    weight = float(args.fut_refiner_jerk_weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("fut_refiner_jerk_weight must be finite and nonnegative")
    args.fut_refiner_jerk_threshold = None
    args.refiner_physics = None
    if weight == 0:
        return
    window = MOTION_WINDOW
    path = resolve_physics_threshold_file(args.dataset, window_length=window)
    if path is None:
        raise ValueError("Jerk regularization requires a training-set P95 calibration")
    thresholds, _ = load_physics_thresholds(path, args.dataset, dt, PHYSICS_THRESHOLD_QUANTILE,
                                          MOTION_SPEED_MIN, window)
    if thresholds['jerk'] <= 0:
        raise ValueError("Jerk P95 must be positive")
    args.fut_refiner_jerk_threshold = thresholds['jerk']
    args.refiner_physics = {
        **get_motion_method(window), "dt": dt, "threshold_quantile": PHYSICS_THRESHOLD_QUANTILE,
        "jerk_threshold": thresholds['jerk'], "threshold_file": str(path),
        "threshold_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "jerk_weight": weight, "candidate_scope": "all",
        "loss": "mean_squared_normalized_jerk_excess",
    }
    print(f"[RefinerJerk] SG{window}/P3 | P95={thresholds['jerk']:.6f} | weight={weight:g}")


def jerk_excess_loss(refined, valid, dt, threshold):
    if threshold is None or not math.isfinite(float(threshold)) or threshold <= 0:
        raise ValueError("Jerk regularization requires a positive training P95")
    motion = compute_future_motion(refined, dt, valid_mask=valid)
    support = motion['jerk_valid']
    if not torch.any(support):
        return refined.sum() * 0.
    return torch.relu(motion['jerk'][support] / float(threshold) - 1.).square().mean().to(refined.dtype)
