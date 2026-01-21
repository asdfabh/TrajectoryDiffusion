import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask


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


def main():
    args = get_args_parser().parse_args()

    args.batch_size = 512
    args.num_workers = 8

    EVAL_MASK_PROB = 0.5

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

    test_dataset = NgsimDataset(test_path, t_h=30, t_f=50, d_s=2)
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

    # 指标统计
    global_mse_sum = 0.0
    global_count = 0
    obs_mse_sum = 0.0
    obs_count = 0
    masked_mse_sum = 0.0
    masked_count = 0
    total_ade = 0.0
    total_fde = 0.0
    total_traj_count = 0

    UNIT_CONVERSION = 0.3048

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Eval Hist", ncols=120)

        for batch_idx, batch in pbar:
            hist, hist_masked, hist_mask = prepare_input_data(
                batch, args.feature_dim, mask_type='random', mask_prob=EVAL_MASK_PROB, device=device
            )

            _, pred, ade, fde = model.forward_eval(hist, hist_masked, device)

            pred_pos = pred[..., :2]
            target_pos = hist[..., :2]

            diff_sq = (pred_pos - target_pos) ** 2

            mask_float = hist_mask.float().expand_as(diff_sq)

            # 1. Global
            global_mse_sum += diff_sq.sum().item()
            global_count += diff_sq.numel()

            # 2. Observed (mask == 1)
            obs_diff_sq = diff_sq * mask_float
            obs_mse_sum += obs_diff_sq.sum().item()
            obs_count += mask_float.sum().item()

            # 3. Masked (mask == 0)
            inv_mask = 1.0 - mask_float
            masked_diff_sq = diff_sq * inv_mask
            masked_mse_sum += masked_diff_sq.sum().item()
            masked_count += inv_mask.sum().item()

            current_bs = hist.shape[0]
            total_ade += ade.item() * current_bs
            total_fde += fde.item() * current_bs
            total_traj_count += current_bs

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    final_global_mse = safe_div(global_mse_sum, global_count)
    final_obs_mse = safe_div(obs_mse_sum, obs_count)
    final_masked_mse = safe_div(masked_mse_sum, masked_count)

    scale_sq = UNIT_CONVERSION ** 2
    mse_m_global = final_global_mse * scale_sq
    mse_m_obs = final_obs_mse * scale_sq
    mse_m_masked = final_masked_mse * scale_sq

    rmse_m_global = np.sqrt(mse_m_global)
    rmse_m_obs = np.sqrt(mse_m_obs)
    rmse_m_masked = np.sqrt(mse_m_masked)

    ade_m = (total_ade / total_traj_count) * UNIT_CONVERSION
    fde_m = (total_fde / total_traj_count) * UNIT_CONVERSION

    print('\n' + '=' * 30 + ' Hist Reconstruction Results ' + '=' * 30)
    print(f"Mask Probability (Keep Ratio): {EVAL_MASK_PROB}")
    print('-' * 75)
    print(f"{'Metric':<20} | {'MSE (m^2)':<15} | {'RMSE (m)':<15}")
    print('-' * 75)
    print(f"{'Global (All)':<20} | {mse_m_global:.6f}        | {rmse_m_global:.6f}")
    print(f"{'Observed (Seen)':<20} | {mse_m_obs:.6f}        | {rmse_m_obs:.6f}")
    print(f"{'Masked (Unseen)':<20} | {mse_m_masked:.6f}        | {rmse_m_masked:.6f}")
    print('-' * 75)
    print(f"Overall ADE (m): {ade_m:.6f}")
    print(f"Overall FDE (m): {fde_m:.6f}")
    print('=' * 75)


if __name__ == '__main__':
    main()