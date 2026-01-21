# import sys
# import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
# import torch
# from torch.utils.data import DataLoader
# from pathlib import Path
# from method_diffusion.models.fut_model import DiffusionFut
# from method_diffusion.dataset.ngsim_dataset import NgsimDataset
# from method_diffusion.config import get_args_parser
# from method_diffusion.utils.mask_util import random_mask, continuous_mask
# from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
#
# def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
#     hist = batch['hist']  # [B, T, 2]
#     va = batch['va']  # [B, T, 2]
#     lane = batch['lane']  # [B, T, 1]
#     cclass = batch['cclass']  # [B, T, 1]
#     fut = batch['fut']  # [B, T, 2]
#     hist_nbrs = batch['nbrs']  # [B, N, T, 2]
#     va_nbrs = batch['nbrs_va']  # [B, N, T, 2]
#     lane_nbrs = batch['nbrs_lane']  # [B, N, T, 1]
#     cclass_nbrs = batch['nbrs_class']
#     mask = batch['mask']  # [B, 3, 13, h]
#     temporal_mask = batch['temporal_mask']  # [B, 3, 13, dim]
#
#
#     # 根据 feature_dim 拼接特征
#     if feature_dim == 6:
#         hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)  # [B, T, 6]
#         hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs, cclass_nbrs), dim=-1).to(device)  # [B, N, T, 6]
#     elif feature_dim == 5:
#         hist = torch.cat((hist, va, lane), dim=-1).to(device) # [B, T, 5]
#         hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs), dim=-1).to(device)
#     elif feature_dim == 4:
#         hist = torch.cat((hist, va), dim=-1).to(device)
#         hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)
#     else:  # feature_dim == 2
#         hist = hist.to(device)
#         hist_nbrs = hist_nbrs.to(device)
#     fut = fut.to(device)
#
#     # 生成掩码并应用掩码
#     if mask_type == 'random':
#         hist_mask = random_mask(hist, p=mask_prob).to(device)
#     elif mask_type == 'block':
#         hist_mask = continuous_mask(hist, p=mask_prob).to(device)
#     else:
#         hist_mask = random_mask(hist, p=mask_prob).to(device)
#
#     hist_masked_val = hist_mask * hist
#     hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1) # [B, T, feature_dim+1]
#
#     hist_masked = hist_masked.to(device)
#     mask = mask.to(device)
#     temporal_mask = temporal_mask.to(device)
#
#     return hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask
#
# def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim,
#                 mask_type='random', mask_prob=0.4):
#
#     model.train()
#     total_loss = 0.0
#     num_batches = 0
#
#     pbar = tqdm(enumerate(dataloader), total=len(dataloader),
#                 desc=f"Epoch {epoch}", ncols=150)
#
#
#     for batch_idx, batch in pbar:
#         hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
#             batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
#         )
#
#         loss, pred, ade, fde = model.forward_train(hist, hist_nbrs, mask, temporal_mask, fut, device)
#         # _, _, _, _ = model.forward_eval(hist, hist_nbrs, mask, temporal_mask, fut, device)
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#
#         total_loss += loss.item()
#         num_batches += 1
#
#         pbar.set_postfix({
#             'loss': f'{loss.item():.8f}',
#             'avg_loss': f'{total_loss/num_batches:.8f}',
#             'ade': f'{ade.mean().item():.4f}',
#             'fde': f'{fde.mean().item():.4f}',
#         })
#
#     avg_loss = total_loss / num_batches
#
#     model.eval()
#     with torch.no_grad():
#         hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
#             batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
#         )
#         eval_loss, eval_pred, eval_ade, eval_fde = model.forward_eval(hist, hist_nbrs, mask, temporal_mask, fut, device)
#         print(f"EVAL at Epoch {epoch}: ADE: {eval_ade:.4f}, FDE: {eval_fde:.4f}")
#     model.train()
#
#     return avg_loss
#
#
# def load_checkpoint_if_needed(args, model, optimizer, scheduler, device):
#     start_epoch = 0
#     best_loss = float('inf')
#     ckpt_path = None
#     if args.resume_fut == 'latest':
#         ckpts = sorted(Path(args.checkpoint_dir).glob('checkpoint_epoch_*.pth'))
#         if ckpts:
#             ckpt_path = ckpts[-1]
#     elif args.resume_fut == 'best':
#         best_candidate = Path(args.checkpoint_dir) / 'checkpoint_best.pth'
#         if best_candidate.exists():
#             ckpt_path = best_candidate
#     elif args.resume_fut.startswith('epoch'):
#         try:
#             epoch_num = int(args.resume_fut.replace('epoch', ''))
#             ckpt_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch_num}.pth'
#             if not ckpt_path.exists():
#                 print(f"Warning: {ckpt_path} not found")
#                 ckpt_path = None
#         except ValueError:
#             print(f"Invalid epoch format: {args.resume_fut}")
#             ckpt_path = None
#     elif args.resume_fut not in ('none', ''):
#         ckpt_path = Path(args.resume_fut)
#
#     if ckpt_path and ckpt_path.exists():
#         state = torch.load(ckpt_path, map_location=device)
#         state_dict = state['model_state_dict']
#         state_dict = {k: v for k, v in state_dict.items() if k not in ['pos_mean', 'pos_std']}
#
#         model.load_state_dict(state_dict, strict=False)
#         optimizer.load_state_dict(state['optimizer_state_dict'])
#         scheduler.load_state_dict(state['scheduler_state_dict'])
#         start_epoch = state.get('epoch', 0)
#         best_loss = state.get('best_loss', best_loss)
#         print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")
#     return start_epoch, best_loss
#
#
# def main():
#     args = get_args_parser().parse_args()
#
#     # Set checkpoint directory for FutModel
#     args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'fut')
#     Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
#
#     log_dir = Path(args.checkpoint_dir) / 'logs'
#     writer = SummaryWriter(log_dir=str(log_dir))
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
#     train_path = str(data_root / 'TrainSet.mat')
#
#     train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=train_dataset.collate_fn,
#         pin_memory=True,
#         persistent_workers=True,
#         drop_last=True
#     )
#
#     model = DiffusionFut(args).to(device)
#
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=args.learning_rate,
#         weight_decay=1e-5
#     )
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=args.num_epochs
#     )
#
#     start_epoch, best_loss = load_checkpoint_if_needed(
#         args, model, optimizer, scheduler, device
#     )
#
#     for epoch in range(start_epoch, args.num_epochs):
#         print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")
#
#         mask_type = 'random' # Can be 'block'
#         mask_prob = 0.5
#
#         avg_loss = train_epoch(
#             model, train_loader, optimizer, device, epoch + 1,
#             args.feature_dim, mask_type=mask_type, mask_prob=mask_prob
#         )
#
#         print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")
#
#         writer.add_scalar('Train/Loss', avg_loss, epoch + 1)
#         current_lr = optimizer.param_groups[0]['lr']
#         writer.add_scalar('Train/LR', current_lr, epoch + 1)
#
#         scheduler.step()
#
#         state = {
#             'epoch': epoch + 1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#             'loss': avg_loss,
#             'best_loss': best_loss,
#         }
#
#         if (epoch + 1) % args.save_interval == 0:
#             torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
#         if avg_loss < best_loss:
#             best_loss = avg_loss
#             state['best_loss'] = best_loss
#             torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")
#
#     writer.close()
#
# if __name__ == '__main__':
#     main()

"""
==========================================================================================================================
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        hist = torch.cat((hist, va, lane), dim=-1).to(device)  # [B, T, 5]
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
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1)  # [B, T, feature_dim+1]

    hist_masked = hist_masked.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask


def train_epoch(model_fut, model_hist, dataloader, optimizer, device, epoch, feature_dim,
                mask_type='random', mask_prob=0.4):
    model_fut.train()
    # Ensure Hist model is in eval mode
    model_hist.eval()

    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=150)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )


        with torch.no_grad():
            loss_h, Ego_hist, ade_h, fde_h = model_hist.forward_eval(hist, hist_masked, device)

        loss, pred, ade, fde = model_fut.forward_train(Ego_hist, hist_nbrs, mask, temporal_mask, fut, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fut.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.8f}',
            'avg_loss': f'{total_loss / num_batches:.8f}',
            'ade': f'{ade.mean().item():.4f}',
            'fde': f'{fde.mean().item():.4f}',
        })

    avg_loss = total_loss / num_batches

    return avg_loss


def load_hist_checkpoint(args, model, device):
    """Specifically loads the pre-trained history model parameters and freezes them."""
    ckpt_path = None

    # Logic to find the hist checkpoint based on resume_hist arg
    if args.resume_hist == 'latest':
        # Assuming hist model checkpoints are in a 'hist' subdirectory or similar convention
        hist_ckpt_dir = Path(args.checkpoint_dir).parent / 'hist'
        ckpts = sorted(hist_ckpt_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif args.resume_hist == 'best':
        hist_ckpt_dir = Path(args.checkpoint_dir).parent / 'hist'
        best_candidate = hist_ckpt_dir / 'checkpoint_best.pth'
        if best_candidate.exists():
            ckpt_path = best_candidate
    elif args.resume_hist and args.resume_hist != 'none':
        ckpt_path = Path(args.resume_hist)

    if ckpt_path and ckpt_path.exists():
        print(f"Loading Hist Model from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        print(f"WARNING: No Hist checkpoint found at {ckpt_path}. Model initiated with random weights!")

    # Freeze Parameters
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("Hist Model parameters frozen.")
    return model


def load_fut_checkpoint_if_needed(args, model, optimizer, scheduler, device):
    start_epoch = 0
    best_loss = float('inf')
    ckpt_path = None
    if args.resume_fut == 'latest':
        ckpts = sorted(Path(args.checkpoint_dir).glob('checkpoint_epoch_*.pth'))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif args.resume_fut == 'best':
        best_candidate = Path(args.checkpoint_dir) / 'checkpoint_best.pth'
        if best_candidate.exists():
            ckpt_path = best_candidate
    elif args.resume_fut.startswith('epoch'):
        try:
            epoch_num = int(args.resume_fut.replace('epoch', ''))
            ckpt_path = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch_num}.pth'
            if not ckpt_path.exists():
                print(f"Warning: {ckpt_path} not found")
                ckpt_path = None
        except ValueError:
            print(f"Invalid epoch format: {args.resume_fut}")
            ckpt_path = None
    elif args.resume_fut not in ('none', ''):
        ckpt_path = Path(args.resume_fut)

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        # Load logic specific to Fut model
        model.load_state_dict(state['model_state_dict'], strict=False)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        scheduler.load_state_dict(state['scheduler_state_dict'])
        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)
        print(f"Resumed Fut Model from {ckpt_path} @ epoch {start_epoch}")
    return start_epoch, best_loss


def main():
    args = get_args_parser().parse_args()

    log_dir = Path(args.checkpoint_dir) / 'logs'
    writer = SummaryWriter(log_dir=str(log_dir))

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

    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'hist')
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print("Initializing Hist Model...")
    model_hist = DiffusionPast(args).to(device)
    model_hist = load_hist_checkpoint(args, model_hist, device)


    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'fut')
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print("Initializing Fut Model...")
    model_fut = DiffusionFut(args).to(device)

    optimizer = torch.optim.AdamW(
        model_fut.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    start_epoch, best_loss = load_fut_checkpoint_if_needed(
        args, model_fut, optimizer, scheduler, device
    )

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        mask_type = 'random'  # Can be 'block'
        mask_prob = 0.5

        avg_loss = train_epoch(
            model_fut, model_hist, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, mask_type=mask_type, mask_prob=mask_prob
        )

        print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")

        writer.add_scalar('Train/Loss', avg_loss, epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch + 1)

        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model_fut.state_dict(),
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

    writer.close()


if __name__ == '__main__':
    main()