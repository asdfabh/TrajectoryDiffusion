import sys
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask

METER_PER_FOOT = 0.3048


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


def load_checkpoint(model, resume_arg, default_dir, device, model_name="Model"):
    ckpt_path = None
    default_dir = Path(default_dir)

    if Path(resume_arg).is_absolute() and Path(resume_arg).exists():
        ckpt_path = Path(resume_arg)
    elif (default_dir / resume_arg).exists():
        ckpt_path = default_dir / resume_arg
    elif resume_arg == 'latest':
        ckpts = sorted(default_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts: ckpt_path = ckpts[-1]
    elif resume_arg == 'best':
        best_cand = default_dir / 'checkpoint_best.pth'
        if best_cand.exists(): ckpt_path = best_cand
    elif resume_arg.startswith('epoch'):
        try:
            ep = int(resume_arg.replace('epoch', ''))
            ckpt_path = default_dir / f'checkpoint_epoch_{ep}.pth'
        except:
            pass

    if ckpt_path and ckpt_path.exists():
        print(f"[{model_name}] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k in ['pos_mean', 'pos_std', 'va_mean', 'va_std']: continue
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
    else:
        print(f"[{model_name}] [Error] Checkpoint '{resume_arg}' not found in {default_dir}. Using random weights!")

    return model


class HistReconstructionMetrics:
    def __init__(self, meter_per_foot=METER_PER_FOOT, hist_dt=0.2):
        self.meter_per_foot = float(meter_per_foot)
        self.hist_dt = float(hist_dt)

        self.xy_dist_sum = {"all": 0.0, "known": 0.0, "masked": 0.0}
        self.xy_point_count = {"all": 0.0, "known": 0.0, "masked": 0.0}
        self.xy_coord_se = {"all": 0.0, "known": 0.0, "masked": 0.0}
        self.xy_coord_count = {"all": 0.0, "known": 0.0, "masked": 0.0}

        self.va_coord_se = {key: torch.zeros(2, dtype=torch.float64) for key in ("all", "known", "masked")}
        self.va_point_count = {"all": 0.0, "known": 0.0, "masked": 0.0}

        self.masked_fde_sum = 0.0
        self.masked_fde_count = 0.0

        self.dxy_coord_se = 0.0
        self.dxy_coord_count = 0.0
        self.v_cons_se = 0.0
        self.v_cons_count = 0.0
        self.a_cons_se = 0.0
        self.a_cons_count = 0.0

    def update(self, pred, target, mask):
        pred = pred.detach()
        target = target.detach()
        known = (mask[..., 0] > 0.5)
        masked = ~known
        all_mask = torch.ones_like(known, dtype=torch.bool)
        region_masks = {"all": all_mask, "known": known, "masked": masked}

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
                self.va_coord_se[region] += torch.sum((va_diff ** 2) * region_float.unsqueeze(-1), dim=(0, 1)).double().cpu()
                self.va_point_count[region] += float(region_float.sum().item())

            pair_mask = torch.maximum(masked[:, 1:].float(), masked[:, :-1].float())
            pred_dxy = pred[:, 1:, :2] - pred[:, :-1, :2]
            target_dxy = target[:, 1:, :2] - target[:, :-1, :2]
            self.dxy_coord_se += float((((pred_dxy - target_dxy) ** 2) * pair_mask.unsqueeze(-1)).sum().item())
            self.dxy_coord_count += float(pair_mask.sum().item() * 2.0)

            pred_v = pred[..., 2]
            pred_a = pred[..., 3]
            pred_v_from_y = pred_dxy[..., 1] / self.hist_dt
            pred_a_from_v = (pred_v[:, 1:] - pred_v[:, :-1]) / self.hist_dt
            self.v_cons_se += float((((pred_v_from_y - pred_v[:, 1:]) ** 2) * pair_mask).sum().item())
            self.v_cons_count += float(pair_mask.sum().item())
            self.a_cons_se += float((((pred_a_from_v - pred_a[:, 1:]) ** 2) * pair_mask).sum().item())
            self.a_cons_count += float(pair_mask.sum().item())

        t_idx = torch.arange(masked.shape[1], device=masked.device).unsqueeze(0).expand_as(masked)
        last_mask_idx = torch.where(masked, t_idx, t_idx.new_full(t_idx.shape, -1)).max(dim=1).values
        has_masked = last_mask_idx >= 0
        if bool(has_masked.any()):
            last_err = xy_dist.gather(1, last_mask_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
            self.masked_fde_sum += float((last_err * has_masked.float()).sum().item())
            self.masked_fde_count += float(has_masked.float().sum().item())

    @staticmethod
    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    def summary(self):
        xy_ade_m = {}
        xy_rmse_m = {}
        for region in ("all", "known", "masked"):
            xy_ade_ft = self.safe_div(self.xy_dist_sum[region], self.xy_point_count[region])
            xy_rmse_ft = self.safe_div(self.xy_coord_se[region], self.xy_coord_count[region]) ** 0.5
            xy_ade_m[region] = xy_ade_ft * self.meter_per_foot
            xy_rmse_m[region] = xy_rmse_ft * self.meter_per_foot

        va_rmse = {}
        for region in ("all", "known", "masked"):
            point_count = self.va_point_count[region]
            if point_count > 0:
                rmse_ft = torch.sqrt(self.va_coord_se[region] / point_count)
                va_rmse[region] = (rmse_ft * self.meter_per_foot).tolist()
            else:
                va_rmse[region] = [0.0, 0.0]

        return {
            "xy_ade_m": xy_ade_m,
            "xy_rmse_m": xy_rmse_m,
            "masked_fde_m": self.safe_div(self.masked_fde_sum, self.masked_fde_count) * self.meter_per_foot,
            "dxy_rmse_m": self.safe_div(self.dxy_coord_se, self.dxy_coord_count) ** 0.5 * self.meter_per_foot,
            "v_cons_rmse_mps": self.safe_div(self.v_cons_se, self.v_cons_count) ** 0.5 * self.meter_per_foot,
            "a_cons_rmse_mps2": self.safe_div(self.a_cons_se, self.a_cons_count) ** 0.5 * self.meter_per_foot,
            "va_rmse": va_rmse,
        }


def print_metrics(summary, title, mask_type, mask_prob):
    print('\n' + '=' * 28 + f' {title} ' + '=' * 28)
    print(f"Mask Type: {mask_type} | Mask Probability: {mask_prob:.4f}")
    print(f"Masked XY ADE (m): {summary['xy_ade_m']['masked']:.6f}")
    print(f"Masked XY FDE (m): {summary['masked_fde_m']:.6f}")
    print(f"Masked XY RMSE (m): {summary['xy_rmse_m']['masked']:.6f}")
    print('-' * 90)
    print(f"{'XY Region':<12} | {'ADE (m)':<12} | {'RMSE (m)':<12}")
    print('-' * 90)
    print(f"{'All':<12} | {summary['xy_ade_m']['all']:<12.6f} | {summary['xy_rmse_m']['all']:<12.6f}")
    print(f"{'Known':<12} | {summary['xy_ade_m']['known']:<12.6f} | {summary['xy_rmse_m']['known']:<12.6f}")
    print(f"{'Masked':<12} | {summary['xy_ade_m']['masked']:<12.6f} | {summary['xy_rmse_m']['masked']:<12.6f}")
    print('-' * 90)
    print(f"{'VA Region':<12} | {'V RMSE (m/s)':<16} | {'A RMSE (m/s^2)':<16}")
    print('-' * 90)
    print(f"{'All':<12} | {summary['va_rmse']['all'][0]:<16.6f} | {summary['va_rmse']['all'][1]:<16.6f}")
    print(f"{'Known':<12} | {summary['va_rmse']['known'][0]:<16.6f} | {summary['va_rmse']['known'][1]:<16.6f}")
    print(f"{'Masked':<12} | {summary['va_rmse']['masked'][0]:<16.6f} | {summary['va_rmse']['masked'][1]:<16.6f}")
    print('-' * 90)
    print(f"Masked dXY RMSE (m): {summary['dxy_rmse_m']:.6f}")
    print(f"V Consistency RMSE (m/s): {summary['v_cons_rmse_mps']:.6f}")
    print(f"A Consistency RMSE (m/s^2): {summary['a_cons_rmse_mps2']:.6f}")
    print('=' * 90)


def main():
    args = get_eval_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    script_dir = Path(__file__).resolve().parent
    arg_ckpt_path = Path(args.checkpoint_dir)
    if arg_ckpt_path.is_absolute():
        base_ckpt_dir = arg_ckpt_path
    else:
        base_ckpt_dir = script_dir / arg_ckpt_path.name

    hist_ckpt_dir = base_ckpt_dir / 'hist'
    print(f"[Info] Checkpoint Search Dir: {hist_ckpt_dir}")

    if os.path.exists(os.path.join(args.data_root, 'TestSet.mat')):
        test_path = os.path.join(args.data_root, 'TestSet.mat')
    else:
        data_root = script_dir.parent / 'mnt/datasets/ngsimdata'
        if not data_root.exists():
            data_root = Path('/mnt/datasets/ngsimdata')
        test_path = str(data_root / 'TestSet.mat')

    print(f"Loading test data from: {test_path}")

    test_dataset = NgsimHistDataset(test_path, t_h=30, d_s=2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True
    )

    model = DiffusionPast(args).to(device)
    resume_target = args.resume_hist if args.resume_hist != 'none' else 'best'
    load_checkpoint(model, resume_target, hist_ckpt_dir, device, model_name="HistModel")
    metrics = HistReconstructionMetrics(hist_dt=getattr(model, "hist_dt", 0.2))
    eval_mask_type = str(args.hist_eval_mask_type).lower()
    eval_mask_prob = float(args.hist_eval_mask_prob)

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader, start=1), total=len(test_loader), desc="Eval Hist Recon", ncols=120)

        for batch_idx, batch in pbar:
            batch = filter_valid_batch(batch)
            if batch["hist"].shape[0] == 0:
                continue

            hist, hist_masked, hist_mask = prepare_input_data(
                batch, args.feature_dim, mask_type=eval_mask_type, mask_prob=eval_mask_prob, device=device
            )

            _, pred, _, _ = model.forward_eval(hist, hist_masked, device)
            metrics.update(pred, hist, hist_mask)

            summary = metrics.summary()
            pbar.set_postfix({
                "xy_mask_ade": f"{summary['xy_ade_m']['masked']:.4f}",
                "xy_mask_fde": f"{summary['masked_fde_m']:.4f}",
                "xy_mask_rmse": f"{summary['xy_rmse_m']['masked']:.4f}",
            })

            if batch_idx % 100 == 0:
                print_metrics(summary, f"Hist Test Iteration {batch_idx}", eval_mask_type, eval_mask_prob)

    final_summary = metrics.summary()
    print_metrics(final_summary, "Hist Reconstruction Result", eval_mask_type, eval_mask_prob)


if __name__ == '__main__':
    main()
