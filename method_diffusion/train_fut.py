import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.visualization import plot_traj_with_mask
from method_diffusion.utils.mask_util import random_mask, continuous_mask
import numpy as np
from tqdm import tqdm

def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    lane = batch['lane']  # [B, T, 1]
    cclass = batch['cclass']  # [B, T, 1]
    fut = batch['fut']  # [B, T, 2]
    hist_nbrs = batch['nbrs']  # [B, N, T, 2]
    va_nbrs = batch['nbrs_va']  # [B, N, T, 2]
    lane_nbrs = batch['nbrs_lane']  # [B, N, T, 1]
    cclass_nbrs = batch['nbrs_class']
    mask = batch['mask']  # [B, 3, 13, h]
    temporal_mask = batch['temporal_mask']  # [B, 3, 13, dim]


    # 根据 feature_dim 拼接特征
    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)  # [B, T, 6]
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs, cclass_nbrs), dim=-1).to(device)  # [B, N, T, 6]
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device) # [B, T, 5]
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)
    else:  # feature_dim == 2
        hist = hist.to(device)
        hist_nbrs = hist_nbrs.to(device)
    fut = fut.to(device)

    # 生成掩码并应用掩码
    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob).to(device)
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob).to(device)
    else:
        hist_mask = random_mask(hist, p=mask_prob).to(device)

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1) # [B, T, feature_dim+1]

    hist_masked = hist_masked.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim,
                mask_type='random', mask_prob=0.4):

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=150)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        loss, pred, ade, fde = model.forward_train(hist, hist_nbrs, mask, temporal_mask, fut, device)
        # _, _, _, _ = model.forward_eval(hist, hist_nbrs, mask, temporal_mask, fut, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.8f}',
            'avg_loss': f'{total_loss/num_batches:.8f}',
            'ade': f'{ade.mean().item():.4f}',
            'fde': f'{fde.mean().item():.4f}',
        })

    avg_loss = total_loss / num_batches
    return avg_loss


def load_checkpoint_if_needed(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = None
    if args.resume == 'latest':
        ckpts = sorted(Path(args.checkpoint_dir).glob('checkpoint_epoch_*.pth'))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif args.resume == 'best':
        best_candidate = Path(args.checkpoint_dir) / 'checkpoint_best.pth'
        if best_candidate.exists():
            ckpt_path = best_candidate
    elif args.resume.startswith('epoch'):
        try:
            epoch_num = int(args.resume.replace('epoch', ''))
            ckpt_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch_num}.pth'
            if not ckpt_path.exists():
                print(f"Warning: {ckpt_path} not found")
                ckpt_path = None
        except ValueError:
            print(f"Invalid epoch format: {args.resume}")
            ckpt_path = None
    elif args.resume not in ('none', ''):
        ckpt_path = Path(args.resume)

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)
        print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")
    return start_epoch, best_loss

def main():
    args = get_args_parser().parse_args()

    # Set checkpoint directory for FutModel
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'fut')
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
    train_path = str(data_root / 'TrainSet.mat')

    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )

    model = DiffusionFut(args).to(device)

    # Freezing parameters if needed (placeholder for hist_encoder)
    # model.hist_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    start_epoch, best_loss = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device
    )

    for epoch in range(args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        mask_type = 'random' # Can be 'block'
        mask_prob = 0.5

        avg_loss = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, mask_type=mask_type, mask_prob=mask_prob
        )

        print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")
        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if avg_loss < best_loss:
            best_loss = avg_loss
            state['best_loss'] = best_loss
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

if __name__ == '__main__':
    main()

