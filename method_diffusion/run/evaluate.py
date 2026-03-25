import sys
import os
import re
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask

METER_PER_FOOT = 0.3048
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HIST_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "hist"


def get_eval_args():
    parser = get_args_parser()
    parser.add_argument("--hist_eval_mask_type", default="random", type=str)
    parser.add_argument("--hist_eval_mask_prob", default=0.5, type=float)
    return parser.parse_args()


def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.5, device='cuda'):
    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    lane = batch['lane']  # [B, T, 1]
    cclass = batch['cclass']  # [B, T, 1]

    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
    else:
        hist = hist.to(device)

    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob)
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob)
    else:
        hist_mask = random_mask(hist, p=mask_prob)

    hist_mask = hist_mask.to(device)

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1)

    return hist, hist_masked, hist_mask


def filter_valid_batch(batch):
    sample_valid = batch.get("sample_valid", None)
    if sample_valid is None:
        return batch
    sample_valid = sample_valid.bool()
    if sample_valid.numel() == 0 or bool(sample_valid.all()):
        return batch

    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == sample_valid.shape[0]:
            filtered[key] = value[sample_valid]
        else:
            filtered[key] = value
    return filtered


def resolve_checkpoint_path(resume_arg, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if resume_arg in ("none", "", None):
        resume_arg = "best"

    resume_path = Path(str(resume_arg))
    if resume_path.exists():
        return resume_path
    if resume_arg == "best":
        return checkpoint_dir / "checkpoint_best.pth"
    if resume_arg == "latest":
        ckpts = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        return ckpts[-1] if ckpts else None
    if re.fullmatch(r"epoch\d+", str(resume_arg)):
        epoch_num = int(str(resume_arg).replace("epoch", ""))
        return checkpoint_dir / f"checkpoint_epoch_{epoch_num}.pth"
    return None


def load_checkpoint(model, resume_arg, checkpoint_dir, device, model_name="Model"):
    ckpt_path = resolve_checkpoint_path(resume_arg, checkpoint_dir)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"[{model_name}] Checkpoint not found: resume_hist={resume_arg}, dir={checkpoint_dir}"
        )

    print(f"[{model_name}] Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    new_state_dict = {}
    for k, v in state_dict.items():
        if k in ["pos_mean", "pos_std", "va_mean", "va_std"]:
            continue
        new_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


class HistReconstructionMetrics:
    def __init__(self, meter_per_foot=METER_PER_FOOT):
        self.meter_per_foot = float(meter_per_foot)
        self.regions = ("all", "unmasked", "masked")

        self.xy_dist_sum = {key: 0.0 for key in self.regions}
        self.xy_point_count = {key: 0.0 for key in self.regions}
        self.xy_coord_se = {key: 0.0 for key in self.regions}
        self.xy_coord_count = {key: 0.0 for key in self.regions}

        self.va_abs_sum = {key: torch.zeros(2, dtype=torch.float64) for key in self.regions}
        self.va_coord_se = {key: torch.zeros(2, dtype=torch.float64) for key in self.regions}
        self.va_point_count = {key: 0.0 for key in self.regions}

    def update(self, pred, target, mask):
        pred = pred.detach()
        target = target.detach()
        unmasked = (mask[..., 0] > 0.5)
        masked = ~unmasked
        all_mask = torch.ones_like(unmasked, dtype=torch.bool)
        region_masks = {"all": all_mask, "unmasked": unmasked, "masked": masked}

        xy_diff = pred[..., :2] - target[..., :2]
        xy_dist = torch.norm(xy_diff, dim=-1)

        for region, region_mask in region_masks.items():
            region_float = region_mask.float()
            coord_mask = region_float.unsqueeze(-1)
            self.xy_dist_sum[region] += float((xy_dist * region_float).sum().item())
            self.xy_point_count[region] += float(region_float.sum().item())
            self.xy_coord_se[region] += float(((xy_diff ** 2) * coord_mask).sum().item())
            self.xy_coord_count[region] += float(coord_mask.sum().item())

        if pred.shape[-1] >= 4 and target.shape[-1] >= 4:
            va_diff = pred[..., 2:4] - target[..., 2:4]
            for region, region_mask in region_masks.items():
                region_float = region_mask.float()
                coord_mask = region_float.unsqueeze(-1)
                self.va_abs_sum[region] += torch.sum(torch.abs(va_diff) * coord_mask, dim=(0, 1)).double().cpu()
                self.va_coord_se[region] += torch.sum((va_diff ** 2) * coord_mask, dim=(0, 1)).double().cpu()
                self.va_point_count[region] += float(region_float.sum().item())

    @staticmethod
    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    def summary(self):
        xy_ade_m = {}
        xy_rmse_m = {}
        for region in self.regions:
            xy_ade_ft = self.safe_div(self.xy_dist_sum[region], self.xy_point_count[region])
            xy_rmse_ft = self.safe_div(self.xy_coord_se[region], self.xy_coord_count[region]) ** 0.5
            xy_ade_m[region] = xy_ade_ft * self.meter_per_foot
            xy_rmse_m[region] = xy_rmse_ft * self.meter_per_foot

        va_l1 = {}
        va_rmse = {}
        for region in self.regions:
            point_count = self.va_point_count[region]
            if point_count > 0:
                l1_ft = self.va_abs_sum[region] / point_count
                rmse_ft = torch.sqrt(self.va_coord_se[region] / point_count)
                va_l1[region] = (l1_ft * self.meter_per_foot).tolist()
                va_rmse[region] = (rmse_ft * self.meter_per_foot).tolist()
            else:
                va_l1[region] = [0.0, 0.0]
                va_rmse[region] = [0.0, 0.0]

        xy_ade_m["known"] = xy_ade_m["unmasked"]
        xy_rmse_m["known"] = xy_rmse_m["unmasked"]
        va_l1["known"] = va_l1["unmasked"]
        va_rmse["known"] = va_rmse["unmasked"]

        return {
            "xy_ade_m": xy_ade_m,
            "xy_rmse_m": xy_rmse_m,
            "va_l1": va_l1,
            "va_rmse": va_rmse,
        }


def print_metrics(summary, title, mask_type, mask_prob):
    print('\n' + '=' * 20 + f' {title} ' + '=' * 20)
    print(f"Mask Type: {mask_type} | Mask Probability: {mask_prob:.4f}")
    print(f"Masked XY ADE (m): {summary['xy_ade_m']['masked']:.6f}")
    print(f"Masked XY RMSE (m): {summary['xy_rmse_m']['masked']:.6f}")
    print('-' * 96)
    print(f"{'XY Region':<12} | {'ADE (m)':<12} | {'RMSE (m)':<12}")
    print('-' * 96)
    print(f"{'All':<12} | {summary['xy_ade_m']['all']:<12.6f} | {summary['xy_rmse_m']['all']:<12.6f}")
    print(f"{'Unmasked':<12} | {summary['xy_ade_m']['unmasked']:<12.6f} | {summary['xy_rmse_m']['unmasked']:<12.6f}")
    print(f"{'Masked':<12} | {summary['xy_ade_m']['masked']:<12.6f} | {summary['xy_rmse_m']['masked']:<12.6f}")
    print('-' * 96)
    print(f"{'VA Region':<12} | {'V L1 (m/s)':<14} | {'V RMSE (m/s)':<16} | {'A L1 (m/s^2)':<16} | {'A RMSE (m/s^2)':<16}")
    print('-' * 96)
    print(f"{'All':<12} | {summary['va_l1']['all'][0]:<14.6f} | {summary['va_rmse']['all'][0]:<16.6f} | {summary['va_l1']['all'][1]:<16.6f} | {summary['va_rmse']['all'][1]:<16.6f}")
    print(f"{'Unmasked':<12} | {summary['va_l1']['unmasked'][0]:<14.6f} | {summary['va_rmse']['unmasked'][0]:<16.6f} | {summary['va_l1']['unmasked'][1]:<16.6f} | {summary['va_rmse']['unmasked'][1]:<16.6f}")
    print(f"{'Masked':<12} | {summary['va_l1']['masked'][0]:<14.6f} | {summary['va_rmse']['masked'][0]:<16.6f} | {summary['va_l1']['masked'][1]:<16.6f} | {summary['va_rmse']['masked'][1]:<16.6f}")
    print('=' * 96)


def main():
    args = get_eval_args()
    args.checkpoint_dir = str(HIST_CHECKPOINT_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"[HistEval] Checkpoint dir: {args.checkpoint_dir}")

    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    test_path = data_root / "TestSet.mat"
    if not test_path.exists():
        test_path = data_root / "ValSet.mat"
    print(f"[HistEval] Dataset: {dataset_name}")
    print(f"[HistEval] Test path: {test_path}")

    test_dataset = NgsimHistDataset(str(test_path), t_h=30, d_s=2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True
    )

    model = DiffusionPast(args).to(device)
    load_checkpoint(model, args.resume_hist, args.checkpoint_dir, device, model_name="HistModel")
    metrics = HistReconstructionMetrics()
    eval_mask_type = str(args.hist_eval_mask_type).lower()
    eval_mask_prob = float(args.hist_eval_mask_prob)

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader, start=1), total=len(test_loader), desc="Eval Hist Recon", dynamic_ncols=True)

        for batch_idx, batch in pbar:
            batch = filter_valid_batch(batch)
            if batch["hist"].shape[0] == 0:
                continue

            hist, hist_masked, hist_mask = prepare_input_data(
                batch, args.feature_dim, mask_type=eval_mask_type, mask_prob=eval_mask_prob, device=device
            )

            _, pred = model.forward_eval(hist, hist_masked, device)
            metrics.update(pred, hist, hist_mask)

            summary = metrics.summary()
            pbar.set_postfix({
                "xy_mask_ade": f"{summary['xy_ade_m']['masked']:.4f}",
                "xy_mask_rmse": f"{summary['xy_rmse_m']['masked']:.4f}",
                "v_mask_l1": f"{summary['va_l1']['masked'][0]:.4f}",
                "a_mask_l1": f"{summary['va_l1']['masked'][1]:.4f}",
            })

            if batch_idx % 100 == 0:
                print_metrics(summary, f"Hist Test Iteration {batch_idx}", eval_mask_type, eval_mask_prob)

    final_summary = metrics.summary()
    print_metrics(final_summary, "Hist Reconstruction Result", eval_mask_type, eval_mask_prob)


if __name__ == '__main__':
    main()
