import os
import sys
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers.schedulers import DDIMScheduler

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut


@dataclass
class ScalarMetric:
    dist_sum: float = 0.0
    valid_count: float = 0.0
    final_sum: float = 0.0
    final_count: float = 0.0
    meter_per_foot: float = 0.3048

    def update(self, pred, target, op_mask):
        pred = pred[..., :2]
        target = target[..., :2]
        valid = op_mask[..., 0] if op_mask.dim() == 3 else op_mask
        valid = (valid > 0.5).float().to(pred.device)

        dist = torch.norm(pred - target, dim=-1)  # [B, T]
        self.dist_sum += float((dist * valid).sum().item())
        self.valid_count += float(valid.sum().item())

        valid_counts = valid.sum(dim=1).long()
        has_valid = valid_counts > 0
        last_idx = torch.clamp(valid_counts - 1, min=0)
        final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        self.final_sum += float((final_dist * has_valid.float()).sum().item())
        self.final_count += float(has_valid.float().sum().item())

    def summary(self):
        ade_ft = self.dist_sum / (self.valid_count + 1e-6)
        fde_ft = self.final_sum / (self.final_count + 1e-6)
        return {
            "ade_ft": ade_ft,
            "fde_ft": fde_ft,
            "ade_m": ade_ft * self.meter_per_foot,
            "fde_m": fde_ft * self.meter_per_foot,
        }


def parse_int_list(text):
    items = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        items.append(int(s))
    return items


def parse_float_list(text):
    items = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        items.append(float(s))
    return items


def parse_str_list(text):
    items = []
    for s in text.split(","):
        s = s.strip()
        if not s:
            continue
        items.append(s)
    return items


def _set_timesteps_safe(scheduler, num_steps):
    try:
        scheduler.set_timesteps(num_steps)
    except TypeError:
        scheduler.set_timesteps(num_steps=num_steps)


def build_ddim_scheduler(model, timestep_spacing=None):
    if timestep_spacing is None or str(timestep_spacing).strip() == "":
        return model.diffusion_scheduler
    try:
        return DDIMScheduler.from_config(
            model.diffusion_scheduler.config,
            timestep_spacing=str(timestep_spacing),
        )
    except Exception as exc:
        print(
            f"[Warn] Failed to create scheduler with timestep_spacing='{timestep_spacing}' "
            f"({type(exc).__name__}: {exc}). Falling back to model default scheduler."
        )
        return model.diffusion_scheduler


def prepare_fut_inputs(batch, feature_dim, device):
    hist = batch["hist"]
    va = batch["va"]
    lane = batch["lane"]
    cclass = batch["cclass"]
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    hist_nbrs = batch["nbrs"]
    va_nbrs = batch["nbrs_va"]
    lane_nbrs = batch["nbrs_lane"]
    cclass_nbrs = batch["nbrs_class"]
    mask = batch["mask"]
    temporal_mask = batch["temporal_mask"]

    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs, cclass_nbrs), dim=-1).to(device)
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)
    else:
        hist = hist.to(device)
        hist_nbrs = hist_nbrs.to(device)

    return (
        hist,
        hist_nbrs,
        fut.to(device),
        op_mask.to(device),
        mask.to(device),
        temporal_mask.to(device),
    )


def load_fut_checkpoint(model, resume_arg, default_dir, device):
    ckpt_path = None
    default_dir = Path(default_dir)

    if resume_arg in ("none", "", None):
        print("[FutModel] No checkpoint specified; using random init.")
        return model

    if Path(resume_arg).exists():
        ckpt_path = Path(resume_arg)
    elif (default_dir / resume_arg).exists():
        ckpt_path = default_dir / resume_arg
    elif resume_arg == "latest":
        ckpts = sorted(default_dir.glob("checkpoint_epoch_*.pth"))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif resume_arg == "best":
        cand = default_dir / "checkpoint_best.pth"
        if cand.exists():
            ckpt_path = cand
    elif resume_arg.startswith("epoch"):
        try:
            ep = int(resume_arg.replace("epoch", ""))
            ckpt_path = default_dir / f"checkpoint_epoch_{ep}.pth"
        except Exception:
            pass

    if ckpt_path and ckpt_path.exists():
        print(f"[FutModel] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        new_state = {}
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
            new_state[clean_k] = v

        model_keys = set(model.state_dict().keys())
        matched_keys = len(model_keys.intersection(new_state.keys()))
        if matched_keys == 0:
            raise RuntimeError(
                "[FutModel] Checkpoint keys do not match model keys at all. "
                "Please verify model config/checkpoint compatibility."
            )

        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[FutModel] Matched keys: {matched_keys}/{len(model_keys)}")
        if missing:
            print(f"[FutModel] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[FutModel] Unexpected keys: {len(unexpected)}")
    else:
        print(f"[FutModel] Warning: checkpoint '{resume_arg}' not found in {default_dir}.")

    model.eval()
    return model


def get_test_loader(args):
    if getattr(args, "test_path", None):
        test_path = args.test_path
    elif os.path.exists(os.path.join(args.data_root, "TestSet.mat")):
        test_path = os.path.join(args.data_root, "TestSet.mat")
    else:
        test_path = str(Path(args.data_root) / "TestSet.mat")

    print(f"[Data] Test path: {test_path}")
    dataset = NgsimDataset(
        test_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )


@torch.no_grad()
def encode_context(model, hist, hist_nbrs, mask, temporal_mask):
    hist_norm = model.norm(hist)
    hist_nbrs_norm = model.norm(hist_nbrs)
    context, hist_enc = model.hist_encoder(hist_norm, hist_nbrs_norm, mask, temporal_mask)
    enc_emb = model.enc_embedding(hist_enc[:, -1, :])
    return context, enc_emb


@torch.no_grad()
def predict_x0(model, x_t, context, enc_emb, t):
    bsz = x_t.size(0)
    timesteps = torch.full((bsz,), int(t), device=x_t.device, dtype=torch.long)
    y = model.timestep_embedder(timesteps) + enc_emb
    input_embedded = model.input_embedding(x_t) + model.pos_embedding(x_t)
    return model.dit(x=input_embedded, y=y, cross=context)


@torch.no_grad()
def run_diagnostics(model, dataloader, args, device, num_steps):
    meter = 0.3048
    one_step = ScalarMetric(meter_per_foot=meter)
    mid_rollout = ScalarMetric(meter_per_foot=meter)
    pure_rollout = ScalarMetric(meter_per_foot=meter)

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Diagnose@steps={num_steps}", ncols=130)
    for batch_idx, batch in pbar:
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        hist, hist_nbrs, fut, op_mask, mask, temporal_mask = prepare_fut_inputs(batch, args.feature_dim, device)
        bsz, t_fut, dim = fut.shape

        context, enc_emb = encode_context(model, hist, hist_nbrs, mask, temporal_mask)
        fut_norm = model.norm(fut)

        scheduler = model.diffusion_scheduler
        _set_timesteps_safe(scheduler, num_steps)
        ts = scheduler.timesteps
        mid_idx = len(ts) // 2
        t_high = int(ts[0].item())
        t_mid = int(ts[mid_idx].item())

        # 1) one-step oracle at high noise
        noise_1 = torch.randn_like(fut_norm)
        tvec_high = torch.full((bsz,), t_high, device=device, dtype=torch.long)
        x_high = scheduler.add_noise(fut_norm, noise_1, tvec_high)
        pred_x0_high = predict_x0(model, x_high, context, enc_emb, t_high)
        pred_high = model.denorm(pred_x0_high)
        one_step.update(pred_high, fut, op_mask)

        # 2) rollout from mid-noise (GT-noised start)
        noise_2 = torch.randn_like(fut_norm)
        tvec_mid = torch.full((bsz,), t_mid, device=device, dtype=torch.long)
        x_mid = scheduler.add_noise(fut_norm, noise_2, tvec_mid)
        for t in ts[mid_idx:]:
            pred_x0_mid = predict_x0(model, x_mid, context, enc_emb, t)
            x_mid = scheduler.step(pred_x0_mid, t, x_mid).prev_sample
        pred_mid = model.denorm(x_mid)
        mid_rollout.update(pred_mid, fut, op_mask)

        # 3) pure-noise full rollout
        x_noise = torch.randn((bsz, t_fut, dim), device=device)
        for t in ts:
            pred_x0_noise = predict_x0(model, x_noise, context, enc_emb, t)
            x_noise = scheduler.step(pred_x0_noise, t, x_noise).prev_sample
        pred_noise = model.denorm(x_noise)
        pure_rollout.update(pred_noise, fut, op_mask)

    return {
        "one_step_oracle_high_t": one_step.summary(),
        "mid_noise_rollout": mid_rollout.summary(),
        "pure_noise_full_rollout": pure_rollout.summary(),
    }


@torch.no_grad()
def run_one_step_t_profile(model, dataloader, args, device, num_steps, probe_count=9):
    scheduler = model.diffusion_scheduler
    _set_timesteps_safe(scheduler, num_steps)
    ts = scheduler.timesteps
    n_steps = len(ts)
    if n_steps == 0:
        return [], {}

    probe_count = min(max(int(probe_count), 1), n_steps)
    lin = torch.linspace(0, n_steps - 1, steps=probe_count)
    probe_indices = sorted(set(int(round(v.item())) for v in lin))
    if probe_indices[0] != 0:
        probe_indices = [0] + probe_indices
    if probe_indices[-1] != n_steps - 1:
        probe_indices.append(n_steps - 1)

    def noise_band(step_idx):
        ratio = step_idx / max(1, n_steps - 1)
        if ratio < 1.0 / 3.0:
            return "high_noise"
        if ratio < 2.0 / 3.0:
            return "mid_noise"
        return "low_noise"

    per_t_metrics = {}
    for idx in probe_indices:
        t = int(ts[idx].item())
        per_t_metrics[(idx, t)] = ScalarMetric()

    band_metrics = {
        "high_noise": ScalarMetric(),
        "mid_noise": ScalarMetric(),
        "low_noise": ScalarMetric(),
    }

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"OneStepProfile@steps={num_steps}",
        ncols=130,
    )
    for batch_idx, batch in pbar:
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        hist, hist_nbrs, fut, op_mask, mask, temporal_mask = prepare_fut_inputs(batch, args.feature_dim, device)
        bsz, _, _ = fut.shape

        context, enc_emb = encode_context(model, hist, hist_nbrs, mask, temporal_mask)
        fut_norm = model.norm(fut)
        shared_noise = torch.randn_like(fut_norm)

        for idx in probe_indices:
            t = int(ts[idx].item())
            tvec = torch.full((bsz,), t, device=device, dtype=torch.long)
            x_t = scheduler.add_noise(fut_norm, shared_noise, tvec)
            pred_x0 = predict_x0(model, x_t, context, enc_emb, t)
            pred = model.denorm(pred_x0)

            per_t_metrics[(idx, t)].update(pred, fut, op_mask)
            band_metrics[noise_band(idx)].update(pred, fut, op_mask)

    curve = []
    for idx in probe_indices:
        t = int(ts[idx].item())
        s = per_t_metrics[(idx, t)].summary()
        curve.append(
            {
                "step_index": idx,
                "timestep": t,
                "noise_band": noise_band(idx),
                "ade_ft": s["ade_ft"],
                "fde_ft": s["fde_ft"],
                "ade_m": s["ade_m"],
                "fde_m": s["fde_m"],
            }
        )

    band_summary = {k: v.summary() for k, v in band_metrics.items()}
    return curve, band_summary


@torch.no_grad()
def run_pure_noise_eval(model, dataloader, args, device, num_steps, eta=0.0, timestep_spacing=None):
    metric = ScalarMetric()
    scheduler = build_ddim_scheduler(model, timestep_spacing=timestep_spacing)
    _set_timesteps_safe(scheduler, num_steps)
    ts = scheduler.timesteps

    pbar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"PureNoise@steps={num_steps},eta={eta},spacing={timestep_spacing}",
        ncols=130,
    )
    for batch_idx, batch in pbar:
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break

        hist, hist_nbrs, fut, op_mask, mask, temporal_mask = prepare_fut_inputs(batch, args.feature_dim, device)
        bsz, t_fut, dim = fut.shape

        context, enc_emb = encode_context(model, hist, hist_nbrs, mask, temporal_mask)

        x_t = torch.randn((bsz, t_fut, dim), device=device)
        for t in ts:
            pred_x0 = predict_x0(model, x_t, context, enc_emb, t)
            try:
                x_t = scheduler.step(pred_x0, t, x_t, eta=float(eta)).prev_sample
            except TypeError:
                x_t = scheduler.step(pred_x0, t, x_t).prev_sample
        pred = model.denorm(x_t)
        metric.update(pred, fut, op_mask)

    return metric.summary()


def print_summary_block(title, summary):
    print(f"\n{'=' * 24} {title} {'=' * 24}")
    for k, v in summary.items():
        print(
            f"{k:<28} | "
            f"ADE: {v['ade_m']:.4f} m / {v['ade_ft']:.4f} ft | "
            f"FDE: {v['fde_m']:.4f} m / {v['fde_ft']:.4f} ft"
        )


def print_sweep_table(results):
    print(f"\n{'=' * 24} Pure-Noise Step Sweep {'=' * 24}")
    print(f"{'steps':>8} | {'ADE(m)':>10} | {'FDE(m)':>10} | {'ADE(ft)':>10} | {'FDE(ft)':>10}")
    print("-" * 64)
    for steps, val in results:
        print(
            f"{steps:>8d} | {val['ade_m']:>10.4f} | {val['fde_m']:>10.4f} | "
            f"{val['ade_ft']:>10.4f} | {val['fde_ft']:>10.4f}"
        )


def print_t_profile_table(rows):
    print(f"\n{'=' * 24} One-Step T Profile {'=' * 24}")
    print(
        f"{'idx':>6} | {'timestep':>8} | {'band':>11} | "
        f"{'ADE(m)':>10} | {'FDE(m)':>10} | {'ADE(ft)':>10} | {'FDE(ft)':>10}"
    )
    print("-" * 90)
    for r in rows:
        print(
            f"{r['step_index']:>6d} | {r['timestep']:>8d} | {r['noise_band']:>11} | "
            f"{r['ade_m']:>10.4f} | {r['fde_m']:>10.4f} | {r['ade_ft']:>10.4f} | {r['fde_ft']:>10.4f}"
        )


def print_sampler_grid_table(results):
    print(f"\n{'=' * 24} Sampler Grid (Pure Noise) {'=' * 24}")
    print(
        f"{'steps':>8} | {'spacing':>10} | {'eta':>6} | "
        f"{'ADE(m)':>10} | {'FDE(m)':>10} | {'ADE(ft)':>10} | {'FDE(ft)':>10}"
    )
    print("-" * 96)
    for row in results:
        print(
            f"{row['steps']:>8d} | {row['spacing']:>10} | {row['eta']:>6.2f} | "
            f"{row['ade_m']:>10.4f} | {row['fde_m']:>10.4f} | {row['ade_ft']:>10.4f} | {row['fde_ft']:>10.4f}"
        )
    if results:
        best = min(results, key=lambda x: x["ade_m"])
        print(
            f"\n[SamplerBest] steps={best['steps']}, spacing={best['spacing']}, eta={best['eta']:.2f} | "
            f"ADE={best['ade_m']:.4f} m ({best['ade_ft']:.4f} ft), "
            f"FDE={best['fde_m']:.4f} m ({best['fde_ft']:.4f} ft)"
        )


def main():
    parser = get_args_parser()
    parser.add_argument("--test_path", type=str, default=None, help="Optional explicit TestSet.mat path")
    parser.add_argument("--diag_steps", type=int, default=None, help="Steps used for diagnosis block")
    parser.add_argument("--sweep_steps", type=str, default="20,30,50,100", help="Comma-separated step list")
    parser.add_argument("--disable_sweep", action="store_true", help="Disable pure-noise sweep/grid (legacy flag)")
    parser.add_argument("--disable_sampler_grid", action="store_true", help="Disable sampler-grid evaluation")
    parser.add_argument("--sampler_spacings", type=str, default="leading,trailing", help="Comma-separated timestep spacing list")
    parser.add_argument("--sampler_etas", type=str, default="0.0,0.1,0.3", help="Comma-separated eta list for DDIM step")
    parser.add_argument("--disable_t_profile", action="store_true", help="Disable one-step t profile diagnostics")
    parser.add_argument("--t_profile_probe_count", type=int, default=9, help="Number of probe points across timesteps")
    parser.add_argument("--max_batches", type=int, default=100, help="0 means full test set")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Env] Device: {device}")
    print(f"[Env] Seed: {args.seed}")

    test_loader = get_test_loader(args)

    script_dir = Path(__file__).resolve().parent
    ckpt_root = Path(args.checkpoint_dir)
    if ckpt_root.is_absolute():
        base_ckpt_dir = ckpt_root
    else:
        base_ckpt_dir = script_dir / ckpt_root.name
    fut_ckpt_dir = base_ckpt_dir / "fut"

    model = DiffusionFut(args).to(device)
    model = load_fut_checkpoint(model, args.resume_fut, fut_ckpt_dir, device)
    model.eval()

    diag_steps = args.num_inference_steps if args.diag_steps is None else args.diag_steps
    diag_summary = run_diagnostics(model, test_loader, args, device, num_steps=diag_steps)
    print_summary_block(f"Diffusion Gap Diagnosis (steps={diag_steps})", diag_summary)

    if not args.disable_t_profile:
        profile_rows, profile_band_summary = run_one_step_t_profile(
            model,
            test_loader,
            args,
            device,
            num_steps=diag_steps,
            probe_count=args.t_profile_probe_count,
        )
        print_t_profile_table(profile_rows)
        print_summary_block("One-Step Band Summary", profile_band_summary)

    disable_grid = args.disable_sweep or args.disable_sampler_grid
    if not disable_grid:
        step_list = parse_int_list(args.sweep_steps)
        spacing_list = parse_str_list(args.sampler_spacings)
        eta_list = parse_float_list(args.sampler_etas)
        sweep_results = []
        for steps in step_list:
            for spacing in spacing_list:
                for eta in eta_list:
                    summary = run_pure_noise_eval(
                        model,
                        test_loader,
                        args,
                        device,
                        num_steps=steps,
                        eta=eta,
                        timestep_spacing=spacing,
                    )
                    sweep_results.append(
                        {
                            "steps": steps,
                            "spacing": spacing,
                            "eta": eta,
                            **summary,
                        }
                    )
        print_sampler_grid_table(sweep_results)


if __name__ == "__main__":
    main()
