#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
ADE_RE = re.compile(rf"Overall ADE:\s*({FLOAT_RE})\s*m\s*\|\s*({FLOAT_RE})\s*ft")
FDE_RE = re.compile(rf"Overall FDE:\s*({FLOAT_RE})\s*m\s*\|\s*({FLOAT_RE})\s*ft")
TIME_METRIC_RE = re.compile(rf"([1-5])s:\s*({FLOAT_RE})\s*m\s*/\s*({FLOAT_RE})\s*ft")


def parse_epoch_list(text):
    vals = []
    for s in str(text).split(","):
        s = s.strip()
        if not s:
            continue
        vals.append(int(s))
    return vals


def range_epochs(start, end, step):
    if step == 0:
        raise ValueError("epoch_step must be non-zero")
    vals = []
    if step > 0:
        cur = start
        while cur <= end:
            vals.append(cur)
            cur += step
    else:
        cur = start
        while cur >= end:
            vals.append(cur)
            cur += step
    return vals


def resolve_fut_ckpt_dir(project_root, checkpoint_dir_arg):
    """
    Match evaluate_fut.py ckpt resolution semantics:
    - absolute -> itself
    - relative -> method_diffusion/<name>
    then append /fut
    """
    script_dir = (project_root / "method_diffusion").resolve()
    p = Path(checkpoint_dir_arg)
    if p.is_absolute():
        base_ckpt_dir = p
    else:
        base_ckpt_dir = script_dir / p.name
    return (base_ckpt_dir / "fut").resolve()


def run_and_log(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
        code = proc.wait()
        f.write(f"\n[EXIT_CODE] {code}\n")
    return code


def parse_time_block(text, header):
    out_m = {}
    out_ft = {}
    idx = text.find(header)
    if idx < 0:
        return out_m, out_ft
    block = text[idx: idx + 900]
    for sec, m_val, ft_val in TIME_METRIC_RE.findall(block):
        out_m[int(sec)] = float(m_val)
        out_ft[int(sec)] = float(ft_val)
    return out_m, out_ft


def parse_eval_metrics(log_path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    ade_m = ade_ft = None
    fde_m = fde_ft = None

    m = ADE_RE.search(text)
    if m:
        ade_m = float(m.group(1))
        ade_ft = float(m.group(2))

    m = FDE_RE.search(text)
    if m:
        fde_m = float(m.group(1))
        fde_ft = float(m.group(2))

    rmse_m, rmse_ft = parse_time_block(text, "RMSE at specific timesteps:")
    de_m, de_ft = parse_time_block(text, "Displacement Error at specific timesteps:")
    return {
        "overall_ade_m": ade_m,
        "overall_ade_ft": ade_ft,
        "overall_fde_m": fde_m,
        "overall_fde_ft": fde_ft,
        "rmse_m": rmse_m,
        "rmse_ft": rmse_ft,
        "de_m": de_m,
        "de_ft": de_ft,
    }


def sort_rows(rows, sort_by):
    huge = 1e18
    if sort_by == "ade":
        key_fn = lambda r: r["overall_ade_m"] if r["overall_ade_m"] is not None else huge
    elif sort_by == "fde":
        key_fn = lambda r: r["overall_fde_m"] if r["overall_fde_m"] is not None else huge
    elif sort_by == "rmse5":
        key_fn = lambda r: r["rmse_5s_m"] if r["rmse_5s_m"] is not None else huge
    else:
        key_fn = lambda r: (
            r["overall_ade_m"] if r["overall_ade_m"] is not None else huge,
            r["overall_fde_m"] if r["overall_fde_m"] is not None else huge,
            r["rmse_5s_m"] if r["rmse_5s_m"] is not None else huge,
        )
    return sorted(rows, key=key_fn)


def main():
    parser = argparse.ArgumentParser("Evaluate existing fut checkpoints by epoch range and rank metrics")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--project_root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--evaluate_script", type=str, default="method_diffusion/evaluate_fut.py")
    parser.add_argument("--checkpoint_dir", type=str, default="method_diffusion/checkpoints")

    parser.add_argument("--epochs", type=str, default="",
                        help="Explicit epoch list, e.g. 20,25,30. If set, range args ignored.")
    parser.add_argument("--epoch_start", type=int, default=20)
    parser.add_argument("--epoch_end", type=int, default=29)
    parser.add_argument("--epoch_step", type=int, default=1)

    parser.add_argument("--eval_ratio", type=float, default=0.03, help="Passed to evaluate_fut --test_ratio")
    parser.add_argument("--eval_mode", type=str, default="fut_only", choices=["fut_only", "joint"])
    parser.add_argument("--eval_extra_args", type=str, default="", help="Extra args for evaluate_fut")
    parser.add_argument("--sort_by", type=str, default="composite", choices=["composite", "ade", "fde", "rmse5"])
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--print_topk", type=int, default=15, help="Print top-k ranked rows to stdout")

    parser.add_argument("--save_root", type=str, default="method_diffusion/eval_runs")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    eval_script = (project_root / args.evaluate_script).resolve()
    if not eval_script.exists():
        raise FileNotFoundError(f"evaluate script not found: {eval_script}")

    if args.epochs.strip():
        wanted_epochs = parse_epoch_list(args.epochs)
    else:
        wanted_epochs = range_epochs(args.epoch_start, args.epoch_end, args.epoch_step)

    fut_ckpt_dir = resolve_fut_ckpt_dir(project_root, args.checkpoint_dir)
    if not fut_ckpt_dir.exists():
        raise FileNotFoundError(f"fut checkpoint dir not found: {fut_ckpt_dir}")

    existing = []
    for ep in wanted_epochs:
        ckpt = fut_ckpt_dir / f"checkpoint_epoch_{ep}.pth"
        if ckpt.exists():
            existing.append(ep)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip() if args.tag else f"fut_eval_epoch_range_{ts}"
    run_dir = (project_root / args.save_root / tag).resolve()
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[EvalSweep] project_root={project_root}")
    print(f"[EvalSweep] evaluate_script={eval_script}")
    print(f"[EvalSweep] checkpoint_dir_arg={args.checkpoint_dir}")
    print(f"[EvalSweep] resolved_fut_ckpt_dir={fut_ckpt_dir}")
    print(f"[EvalSweep] requested_epochs={wanted_epochs}")
    print(f"[EvalSweep] existing_epochs={existing}")
    print(f"[EvalSweep] eval_ratio(test_ratio)={args.eval_ratio}")
    print(f"[EvalSweep] save_dir={run_dir}")

    if not existing:
        print("[EvalSweep] no checkpoint found in requested range.")
        return

    rows = []
    for i, ep in enumerate(existing, start=1):
        resume_arg = f"epoch{ep}"
        log_path = logs_dir / f"eval_epoch_{ep}.log"

        cmd = [
            args.python_bin,
            str(eval_script),
            "--checkpoint_dir", args.checkpoint_dir,
            "--resume_fut", resume_arg,
            "--eval_mode", args.eval_mode,
            "--test_ratio", str(args.eval_ratio),
            "--visualize_samples", "0",
        ]
        if args.eval_extra_args:
            cmd.extend(shlex.split(args.eval_extra_args))

        print(f"\n[{i}/{len(existing)}] evaluating {resume_arg} ...")
        if args.dry_run:
            print("[DryRun]", " ".join(shlex.quote(c) for c in cmd))
            code = 0
            metrics = {
                "overall_ade_m": None, "overall_ade_ft": None,
                "overall_fde_m": None, "overall_fde_ft": None,
                "rmse_m": {}, "rmse_ft": {}, "de_m": {}, "de_ft": {},
            }
        else:
            code = run_and_log(cmd, log_path, cwd=project_root)
            print(f"[Done] epoch={ep}, exit_code={code}, log={log_path}")
            metrics = parse_eval_metrics(log_path) if code == 0 else {
                "overall_ade_m": None, "overall_ade_ft": None,
                "overall_fde_m": None, "overall_fde_ft": None,
                "rmse_m": {}, "rmse_ft": {}, "de_m": {}, "de_ft": {},
            }
            if code != 0 and args.fail_fast:
                raise RuntimeError(f"evaluate failed at epoch{ep}, see {log_path}")

        row = {
            "epoch": ep,
            "resume_fut": resume_arg,
            "exit_code": code,
            "overall_ade_m": metrics["overall_ade_m"],
            "overall_ade_ft": metrics["overall_ade_ft"],
            "overall_fde_m": metrics["overall_fde_m"],
            "overall_fde_ft": metrics["overall_fde_ft"],
            "rmse_1s_m": metrics["rmse_m"].get(1),
            "rmse_2s_m": metrics["rmse_m"].get(2),
            "rmse_3s_m": metrics["rmse_m"].get(3),
            "rmse_4s_m": metrics["rmse_m"].get(4),
            "rmse_5s_m": metrics["rmse_m"].get(5),
            "de_1s_m": metrics["de_m"].get(1),
            "de_2s_m": metrics["de_m"].get(2),
            "de_3s_m": metrics["de_m"].get(3),
            "de_4s_m": metrics["de_m"].get(4),
            "de_5s_m": metrics["de_m"].get(5),
            "log_path": str(log_path),
        }
        rows.append(row)

    valid_rows = [r for r in rows if r["exit_code"] == 0 and r["overall_ade_m"] is not None]
    sorted_rows = sort_rows(valid_rows, args.sort_by)

    all_csv = run_dir / "summary_all.csv"
    sorted_csv = run_dir / "summary_sorted.csv"
    fields = [
        "epoch", "resume_fut", "exit_code",
        "overall_ade_m", "overall_ade_ft", "overall_fde_m", "overall_fde_ft",
        "rmse_1s_m", "rmse_2s_m", "rmse_3s_m", "rmse_4s_m", "rmse_5s_m",
        "de_1s_m", "de_2s_m", "de_3s_m", "de_4s_m", "de_5s_m",
        "log_path",
    ]
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with sorted_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted_rows)

    summary_json = run_dir / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "checkpoint_dir_arg": args.checkpoint_dir,
                    "resolved_fut_ckpt_dir": str(fut_ckpt_dir),
                    "requested_epochs": wanted_epochs,
                    "existing_epochs": existing,
                    "eval_ratio": args.eval_ratio,
                    "eval_mode": args.eval_mode,
                    "sort_by": args.sort_by,
                },
                "all_rows": rows,
                "sorted_valid_rows": sorted_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n================ Eval Sweep Summary ================")
    print(f"[Save] all_csv={all_csv}")
    print(f"[Save] sorted_csv={sorted_csv}")
    print(f"[Save] summary_json={summary_json}")
    if sorted_rows:
        topk = max(1, min(args.print_topk, len(sorted_rows)))
        print(f"[Rank] Top-{topk} by {args.sort_by}:")
        print("rank | epoch | ADE(m) | FDE(m) | RMSE@5s(m)")
        print("-" * 48)
        for i, r in enumerate(sorted_rows[:topk], start=1):
            ade_s = f"{r['overall_ade_m']:.4f}" if r["overall_ade_m"] is not None else "None"
            fde_s = f"{r['overall_fde_m']:.4f}" if r["overall_fde_m"] is not None else "None"
            rmse5_s = f"{r['rmse_5s_m']:.4f}" if r["rmse_5s_m"] is not None else "None"
            print(f"{i:>4d} | {r['epoch']:>5d} | {ade_s:>6} | {fde_s:>6} | {rmse5_s:>10}")
        top = sorted_rows[0]
        print(
            f"[Best] epoch={top['epoch']} | ADE={top['overall_ade_m']:.4f}m | "
            f"FDE={top['overall_fde_m']:.4f}m | RMSE@5s={top['rmse_5s_m']:.4f}m"
        )
    else:
        print("[Best] no valid parsed result.")


if __name__ == "__main__":
    main()
