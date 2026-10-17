"""Optional, bounded profiling for the joint DDP training loop."""

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch


def profile_region(profiler, name):
    return torch.profiler.record_function(name) if profiler is not None else contextlib.nullcontext()


def profiled_batches(dataloader, profiler):
    if profiler is None:
        yield from dataloader
        return
    iterator = iter(dataloader)
    while True:
        with profile_region(profiler, "train/data_wait"):
            try:
                batch = next(iterator)
            except StopIteration:
                return
        yield batch


def make_train_profiler(directory, device, rank, epoch, wait, warmup, active, num_batches, metadata=None):
    if wait < 0 or warmup < 1 or active < 1:
        raise ValueError("Profiler requires wait >= 0, warmup >= 1 and active >= 1")
    if wait + warmup + active > num_batches:
        raise ValueError("Profiler window exceeds batches in this epoch; reduce profile_wait/warmup/active")

    device = torch.device(device)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = directory / f"epoch{epoch}_rank{rank}_{stamp}_pid{os.getpid()}"
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def save_trace(profiler):
        averages = profiler.key_averages()
        profiler.export_chrome_trace(str(stem) + ".trace.json")
        # PyTorch 2.0 calls these CUDA times; newer versions use device times.
        device_key = "self_device_time_total" if any(hasattr(event, "self_device_time_total") for event in averages) else "self_cuda_time_total"
        table = averages.table(sort_by="self_cpu_time_total", row_limit=40)
        if device.type == "cuda":
            table += "\n\n" + averages.table(sort_by=device_key, row_limit=40)
        Path(str(stem) + ".operators.txt").write_text(table, encoding="utf-8")
        stages = {}
        for event in averages:
            if event.key.startswith("train/"):
                device_us = getattr(event, "device_time_total", None)
                if device_us is None:
                    device_us = getattr(event, "cuda_time_total", 0.0)
                stages[event.key] = {
                    "calls": event.count,
                    "cpu_total_ms": event.cpu_time_total / 1000.0,
                    "device_total_ms": device_us / 1000.0,
                }
        report = {
            "rank": rank,
            "epoch": epoch,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_version": str(torch.__version__),
            "window": {"wait": wait, "warmup": warmup, "active": active},
            "config": metadata or {},
            "stages": stages,
            "timing_note": "Inclusive CPU/device durations may overlap; do not sum them as wall time. DDP communication is visible in the trace inside/around backward.",
        }
        Path(str(stem) + ".summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"[TrainProfile] rank={rank} saved {stem}.{{trace.json,operators.txt,summary.json}}", flush=True)

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=save_trace,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )
