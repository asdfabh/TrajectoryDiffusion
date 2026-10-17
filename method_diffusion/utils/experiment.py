"""Checkpoint roots and provenance for the original staged training pipeline."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASK_PROTOCOL = {
    "definition": "iid_timestamp_missing_keep_random_one_v1",
    "all_state_channels_together": True,
    "current_timestamp_can_be_missing": True,
    "empty_history_guard": "retain maximum uniform draw; no extra RNG consumption",
    "neighbor_histories": "complete",
}


def checkpoint_directory(args, *parts):
    root = getattr(args, "_checkpoint_root", None)
    if root is None:
        root = Path(args.checkpoint_dir).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        args._checkpoint_root = str(root.resolve())
    return Path(root).joinpath(*parts)


def ensure_training_destination(directory, resume=None):
    directory = Path(directory)
    if resume in (None, "", "none") and directory.exists() and any(directory.glob("*.pth")):
        raise FileExistsError(f"Weights already exist in {directory}. Use a new --checkpoint_dir or explicit resume.")


def seed_run(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def record_checkpoint_source(model, path, state=None):
    path = Path(path).resolve()
    model.checkpoint_source = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "epoch": (state or {}).get("epoch"),
        "mask_protocol": (state or {}).get("mask_protocol"),
    }


def experiment_metadata(args, hist_model=None, fut_model=None, history_source="completed", refiner_model=None):
    def source(model):
        model = model.module if hasattr(model, "module") else model
        return getattr(model, "checkpoint_source", None)
    return {
        "args": vars(args).copy(),
        "history_source": history_source,
        "mask_protocol": {**MASK_PROTOCOL, "missing_probability": 0.0 if history_source == "full" else float(args.mask_prob)},
        "training_recipe": "pretrain_hist_then_freeze_hist_train_future_then_freeze_upstream_train_refiner",
        "upstream_hist": source(hist_model), "upstream_future": source(fut_model),
        "refiner_checkpoint": source(refiner_model),
        "physics_evaluation": getattr(args, "_physics_evaluation", None),
        "source_sha256": {
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in ("method_diffusion/models/hist_model.py", "method_diffusion/models/fut_model.py",
                         "method_diffusion/models/trajectory_refiner.py", "method_diffusion/utils/mask_util.py",
                         "method_diffusion/dataset/round_dataset.py", "method_diffusion/utils/fut_utils.py",
                         "method_diffusion/utils/sg_kinematics.py", "method_diffusion/utils/ral_physics.py",
                         "method_diffusion/run/train_refine.py")
        },
    }
