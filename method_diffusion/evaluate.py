import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.train_fut import prepare_input_data


def main():
    args = get_args_parser().parse_args()

    args.batch_size = 512
    args.num_workers = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if os.path.exists(os.path.join(args.data_root, 'TestSet.mat')):
        test_path = os.path.join(args.data_root, 'TestSet.mat')
    else:
        data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
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

    model = DiffusionFut(args).to(device)

    ckpt_dir = Path(args.checkpoint_dir) / 'fut'
    ckpt_path = ckpt_dir / 'checkpoint_best.pth'
    # ckpt_path = ckpt_dir / 'checkpoint_epoch_5.pth'

    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    state_dict = {k: v for k, v in state_dict.items() if k not in ['pos_mean', 'pos_std']}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    T_pred = args.T_f  # 25

    total_se = torch.zeros(T_pred).to(device)
    total_de = torch.zeros(T_pred).to(device)
    total_counts = torch.zeros(T_pred).to(device)

    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing", ncols=100)

        for batch_idx, batch in pbar:
            hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
                batch, args.feature_dim, mask_type='random', mask_prob=0.0, device=device
            )

            _, pred, _, _ = model.forward_eval(hist, hist_nbrs, mask, temporal_mask, fut, device)

            pred = pred[:, :T_pred, :2]
            target = fut[:, :T_pred, :2]

            diff = pred - target  # [B, T, 2]
            dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, T] squared error
            dist = torch.sqrt(dist_sq)  # [B, T] displacement error (L2)

            current_batch_size = dist.shape[0]
            total_se += torch.sum(dist_sq, dim=0)
            total_de += torch.sum(dist, dim=0)
            total_counts += current_batch_size

            num_batches += 1

    UNIT_CONVERSION = 0.3048

    rmse = torch.sqrt(total_se / total_counts) * UNIT_CONVERSION

    de_avg = (total_de / total_counts) * UNIT_CONVERSION

    print('\n' + '=' * 30 + ' Test Results ' + '=' * 30)

    # 时间间隔为 0.2s，所以索引 4=1s, 9=2s, 14=3s, 19=4s, 24=5s
    times = [4, 9, 14, 19, 24]  # indices
    time_labels = ['1s', '2s', '3s', '4s', '5s']

    print('RMSE (m):')
    rmse_str = " | ".join([f"{time_labels[i]}: {rmse[t].item():.2f}" for i, t in enumerate(times) if t < len(rmse)])
    print(rmse_str)

    print('FDE (m) (ErrorMessage at specific time):')
    fde_str = " | ".join([f"{time_labels[i]}: {de_avg[t].item():.2f}" for i, t in enumerate(times) if t < len(de_avg)])
    print(fde_str)

    print('ADE (m):')
    ade_vals = []
    for i, t in enumerate(times):
        if t < len(de_avg):

            ade_val = torch.sum(total_de[:t + 1]) / torch.sum(total_counts[:t + 1]) * UNIT_CONVERSION
            ade_vals.append(f"{time_labels[i]}: {ade_val.item():.2f}")

    print(" | ".join(ade_vals))
    print('=' * 74)


if __name__ == '__main__':
    main()
