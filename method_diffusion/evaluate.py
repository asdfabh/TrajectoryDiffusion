import math
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.net import DiffusionPast
from method_diffusion.train import prepare_input_data


def build_dataloader(args) -> Tuple[NgsimDataset, DataLoader]:
    data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'
    dataset = NgsimDataset(str(data_root / 'TestSet.mat'), t_h=30, t_f=50, d_s=2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
    )
    return dataset, loader


def load_checkpoint(model: DiffusionPast, checkpoint_dir: Path, device: torch.device):
    ckpt_path = checkpoint_dir / 'checkpoint_best.pth'
    # ckpt_path = checkpoint_dir / 'checkpoint_epoch_1.pth'
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        print(f'Loaded checkpoint: {ckpt_path} (epoch={state.get("epoch", "unknown")}, '
              f'best_loss={state.get("best_loss", float("nan")):.4f})')
    else:
        print(f'Warning: checkpoint {ckpt_path} not found, using random weights.')


def build_sparse_neighbor_mask(nbrs_masked: torch.Tensor, nbrs_num: torch.Tensor) -> torch.Tensor:
    masks: List[torch.Tensor] = []
    B, T, _, _ = nbrs_masked.shape
    for b in range(B):
        count = int(nbrs_num[b].item())
        if count == 0:
            continue
        obs_flag = nbrs_masked[b, :, :count, -1:].permute(1, 0, 2)  # [count, T, 1]
        masks.append(1.0 - obs_flag)
    if not masks:
        return torch.empty(0, T, 1, device=nbrs_masked.device)
    return torch.cat(masks, dim=0)


def masked_mse_sum(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    if mask.numel() == 0:
        return 0.0, 0.0
    if mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred).float()
    valid = mask.sum().item()
    if valid == 0:
        return 0.0, 0.0
    mse = ((pred - target) ** 2 * mask).sum().item()
    return mse, valid


@torch.no_grad()
def run_inference():
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, dataloader = build_dataloader(args)

    model = DiffusionPast(args).to(device)
    load_checkpoint(model, Path(args.checkpoint_dir), device)
    model.eval()

    mask_prob = args.mask_prob
    num_inference_steps = args.num_inference_steps
    preview_limit = max(0, args.preview_limit)
    preview_gt, preview_pred, preview_mask = [], [], []

    ego_err_sum = nbr_err_sum = 0.0
    ego_weight = nbr_weight = 0.0

    pbar = tqdm(dataloader, desc='Inference', total=len(dataloader))
    for batch in pbar:
        hist_masked, nbrs_masked, nbrs_num = prepare_input_data(
            batch,
            args.feature_dim,
            mask_type='random',
            mask_prob=mask_prob,
            device=device,
        )

        pred_ego, pred_nbrs = model.forward_eval(
            hist_masked, nbrs_masked, nbrs_num, num_inference_steps=num_inference_steps
        )

        gt_hist = batch['hist'].to(device=device, dtype=pred_ego.dtype).unsqueeze(2)  # [B, T, 1, 2]
        gt_nbrs = batch['nbrs'].to(device=device, dtype=pred_nbrs.dtype)  # [N_total, T, 2]

        ego_missing = (1.0 - hist_masked[..., -1:])  # [B, T, 1, 1]
        nbr_missing = build_sparse_neighbor_mask(nbrs_masked, nbrs_num)

        ego_pred_xy = pred_ego[..., :2]
        nbr_pred_xy = pred_nbrs[..., :2]

        ego_err, ego_cnt = masked_mse_sum(ego_pred_xy, gt_hist, ego_missing)
        nbr_err, nbr_cnt = masked_mse_sum(nbr_pred_xy, gt_nbrs, nbr_missing)

        ego_err_sum += ego_err
        ego_weight += ego_cnt
        nbr_err_sum += nbr_err
        nbr_weight += nbr_cnt

        if preview_limit > 0 and len(preview_gt) < preview_limit:
            for b in range(min(gt_hist.size(0), preview_limit - len(preview_gt))):
                preview_gt.append(gt_hist[b, :, 0, :2].detach().cpu().numpy())
                preview_pred.append(ego_pred_xy[b, :, 0, :2].detach().cpu().numpy())
                preview_mask.append(ego_missing[b, :, 0, 0].detach().cpu().numpy())

    ego_rmse = math.sqrt(ego_err_sum / ego_weight) if ego_weight else float('nan')
    nbr_rmse = math.sqrt(nbr_err_sum / nbr_weight) if nbr_weight else float('nan')

    print('=' * 80)
    print(f'Ego RMSE (masked positions): {ego_rmse:.3f}')
    print(f'Neighbor RMSE (masked positions): {nbr_rmse:.3f}')
    print('=' * 80)

    if preview_limit > 0 and preview_gt:
        out_path = Path(args.checkpoint_dir) / 'inference_preview.npz'
        np.savez_compressed(
            out_path,
            gt=np.stack(preview_gt),
            pred=np.stack(preview_pred),
            missing_mask=np.stack(preview_mask),
        )
        print(f'Saved {len(preview_gt)} preview samples to {out_path}')


if __name__ == '__main__':
    run_inference()
