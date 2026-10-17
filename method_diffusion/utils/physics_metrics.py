"""Motion distributions and reusable, explicitly selected threshold files."""

import json
import math
from pathlib import Path

import numpy as np
import torch

from method_diffusion.utils.sg_kinematics import MOTION_SPEED_MIN, MOTION_WINDOW, compute_future_motion, get_motion_method


PHYSICS_KEYS = ("acceleration", "deceleration", "lateral_acceleration", "jerk")
PHYSICS_THRESHOLD_QUANTILE = 0.95
PHYSICS_QUANTILE_SAMPLE_SIZE = 100000


def resolve_physics_threshold_file(dataset, path=None, window_length=MOTION_WINDOW):
    """Resolve the shared default independently of the process working directory."""
    if path is not None and str(path).lower() == "none":
        return None
    if path is None:
        method = get_motion_method(window_length)
        path = Path(__file__).resolve().parents[2] / "outputs" / "physics" / f"{str(dataset).lower()}_train_physics_sg{method['window_length']}p3_derivative.json"
    return Path(path).expanduser().resolve()


class ScalarDistribution:
    """Exact moments, with exact or bounded uniform-sample quantiles.

    sample_size=0 stores every value (training-set calibration). Evaluation
    uses a bounded priority reservoir to avoid retaining every candidate's
    derivatives; quantile_sample_count/quantiles_exact document this choice.
    """

    def __init__(self, sample_size=PHYSICS_QUANTILE_SAMPLE_SIZE, seed=42):
        if int(sample_size) < 0:
            raise ValueError("sample_size must be nonnegative")
        self.sample_size = int(sample_size)
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.chunks = []
        self.values = np.empty(0, dtype=np.float64)
        self.priorities = np.empty(0, dtype=np.float64)

    def update(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if not values.size:
            return
        self.count += int(values.size)
        self.total += float(values.sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        if not self.sample_size:
            self.chunks.append(values.copy())
            return
        priorities = self.rng.random(values.size)
        # Retain the largest random priorities: a uniform sample of all points.
        if values.size > self.sample_size:
            keep = np.argpartition(priorities, -self.sample_size)[-self.sample_size:]
            values, priorities = values[keep], priorities[keep]
        values = np.concatenate((self.values, values))
        priorities = np.concatenate((self.priorities, priorities))
        if values.size > self.sample_size:
            keep = np.argpartition(priorities, -self.sample_size)[-self.sample_size:]
            values, priorities = values[keep], priorities[keep]
        self.values, self.priorities = values, priorities

    def merge(self, other):
        """Pool moments and the largest priorities from independent shards."""
        if self.sample_size != other.sample_size:
            raise ValueError("Cannot merge distributions with different sample budgets.")
        self.count += other.count
        self.total += other.total
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)
        if not self.sample_size:
            self.chunks.extend(other.chunks)
            return
        values = np.concatenate((self.values, other.values))
        priorities = np.concatenate((self.priorities, other.priorities))
        if values.size > self.sample_size:
            keep = np.argpartition(priorities, -self.sample_size)[-self.sample_size:]
            values, priorities = values[keep], priorities[keep]
        self.values, self.priorities = values, priorities

    def summary(self, quantiles=(0.5, 0.95, 0.98, 0.99)):
        values = np.concatenate(self.chunks) if self.chunks else self.values
        return {
            "count": self.count,
            "mean": self.total / self.count if self.count else None,
            "min": self.minimum if self.count else None,
            "max": self.maximum if self.count else None,
            "quantiles": {str(float(q)): float(np.quantile(values, q)) if values.size else None for q in quantiles},
            "quantile_sample_count": int(values.size),
            "quantiles_exact": values.size == self.count,
        }


class MotionStatistics:
    """Accumulate all candidates, rather than an oracle-selected trajectory.

    Acceleration/deceleration distributions condition on positive/negative
    longitudinal acceleration. Their point violation rates both use ALL
    valid longitudinal points, so their sum is the longitudinal violation
    rate. Trajectory violation rates use all candidates with valid points.
    """

    def __init__(self, dt, thresholds=None, speed_min=MOTION_SPEED_MIN, sample_size=PHYSICS_QUANTILE_SAMPLE_SIZE, seed=42, window_length=MOTION_WINDOW):
        self.dt, self.speed_min = float(dt), float(speed_min)
        self.method = get_motion_method(window_length)
        self.thresholds = thresholds
        self.distributions = {key: ScalarDistribution(sample_size, seed + i) for i, key in enumerate(PHYSICS_KEYS)}
        self.counts = {key: {"eligible_points": 0, "violating_points": 0, "eligible_trajectories": 0, "violating_trajectories": 0} for key in PHYSICS_KEYS}
        self.low_speed_points = 0

    def update(self, traj, valid_mask=None):
        motion = compute_future_motion(traj.detach(), self.dt, valid_mask, self.speed_min,
                                       window_length=self.method["window_length"])
        self.update_motion(motion)

    def update_motion(self, motion):
        lon, lat, jerk = (motion[key] for key in ("longitudinal_acceleration", "lateral_acceleration", "jerk"))
        acc_valid, jerk_valid = motion["acceleration_valid"], motion["jerk_valid"]
        self.low_speed_points += int((motion["acceleration_position_valid"] & ~acc_valid).sum().item())
        values = {"acceleration": lon.clamp(min=0), "deceleration": (-lon).clamp(min=0), "lateral_acceleration": lat.abs(), "jerk": jerk}
        for key, value in values.items():
            valid = jerk_valid if key == "jerk" else acc_valid
            distribution_valid = valid & (value > 0) if key in ("acceleration", "deceleration") else valid
            self.distributions[key].update(value[distribution_valid])
            counts = self.counts[key]
            counts["eligible_points"] += int(valid.sum().item())
            eligible = valid.any(dim=-1)
            counts["eligible_trajectories"] += int(eligible.sum().item())
            if self.thresholds is not None:
                violations = valid & (value > self.thresholds[key])
                counts["violating_points"] += int(violations.sum().item())
                counts["violating_trajectories"] += int(violations.any(dim=-1).sum().item())

    def merge(self, other):
        if (self.dt, self.speed_min, self.thresholds, self.method) != (other.dt, other.speed_min, other.thresholds, other.method):
            raise ValueError("Cannot merge motion statistics with different physics settings.")
        for key in PHYSICS_KEYS:
            self.distributions[key].merge(other.distributions[key])
            for name in self.counts[key]:
                self.counts[key][name] += other.counts[key][name]
        self.low_speed_points += other.low_speed_points

    def summary(self):
        result = {}
        for key in PHYSICS_KEYS:
            counts = self.counts[key]
            result[key] = {
                **self.distributions[key].summary(), **counts,
                "unit": "m/s^3" if key == "jerk" else "m/s^2",
                "threshold": self.thresholds[key] if self.thresholds is not None else None,
                "point_violation_rate": counts["violating_points"] / counts["eligible_points"] if self.thresholds is not None and counts["eligible_points"] else None,
                "trajectory_violation_rate": counts["violating_trajectories"] / counts["eligible_trajectories"] if self.thresholds is not None and counts["eligible_trajectories"] else None,
            }
        result["metadata"] = {
            **self.method, "dt": self.dt, "speed_min": self.speed_min,
            "kernel_span_seconds": (self.method["window_length"] - 1) * self.dt,
            "candidate_scope": "all", "low_speed_acceleration_points_excluded": self.low_speed_points,
            "longitudinal_distribution": "positive acceleration / magnitude of negative acceleration; zero excluded",
            "violation_denominator": "all valid points / candidates for the corresponding derivative",
        }
        return result


def load_physics_thresholds(path, dataset, dt, quantile=None, speed_min=MOTION_SPEED_MIN, window_length=MOTION_WINDOW):
    from method_diffusion.dataset.build import get_raw_dt, get_time_params

    with Path(path).open(encoding="utf-8") as handle:
        report = json.load(handle)
    metadata = report["metadata"]
    if metadata.get("partial", False):
        raise ValueError("Partial training statistics cannot be used as evaluation thresholds; run full calibration")
    method = get_motion_method(window_length)
    expected = {"dataset": str(dataset).lower(), "split": "train",
                **{key: method[key] for key in ("definition", "window_length", "polyorder", "future_only", "edge_policy")}}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Physics threshold {key} mismatch: expected {value}, got {metadata.get(key)}")
    for key, value in (("dt", dt), ("speed_min", speed_min), ("raw_dt", get_raw_dt(dataset)), ("d_s", get_time_params(dataset)[2])):
        if not math.isclose(float(metadata[key]), float(value), rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"Physics threshold {key} mismatch: {metadata[key]} vs {value}")
    # Calibration must use the same processed-MAT scale as the model loader.
    expected_scale = 1.0 if str(dataset).lower() == "round" else 0.3048
    if not math.isclose(float(metadata["position_scale"]), expected_scale, rel_tol=1e-8):
        raise ValueError("Calibration position_scale does not match the current dataset loader")
    if quantile is None:
        thresholds = report.get("thresholds")
        if thresholds is None:
            raise ValueError("Threshold file has candidates only; specify a quantile when loading the calibration")
        selected_quantile = metadata.get("selected_quantile")
    else:
        selected_quantile = float(quantile)
        thresholds = report["threshold_candidates"].get(str(selected_quantile))
        if thresholds is None:
            raise ValueError(f"Quantile {selected_quantile} is absent from the threshold file")
    for key in PHYSICS_KEYS:
        value = thresholds.get(key)
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"No usable {key} threshold; inspect its training distribution")
    return {key: float(thresholds[key]) for key in PHYSICS_KEYS}, selected_quantile


def print_motion_statistics(summary, title):
    print(f"\n{title}: all candidate trajectories, future only")
    print(f"{'Metric':<20}|{'Mean':>9}|{'P95':>9}|{'P98':>9}|{'P99':>9}|{'Point VR':>9}|{'Traj VR':>9}")
    for key in PHYSICS_KEYS:
        item = summary[key]
        values = (item["mean"], *(item["quantiles"].get(str(q)) for q in (0.95, 0.98, 0.99)),
                  item["point_violation_rate"], item["trajectory_violation_rate"])
        rendered = ["-" if value is None else f"{value:.6f}" for value in values]
        rendered = [text if len(text) <= 9 else f"{value:.3e}" for value, text in zip(values, rendered)]
        print(f"{key:<20}|" + "|".join(f"{value:>9}" for value in rendered))
