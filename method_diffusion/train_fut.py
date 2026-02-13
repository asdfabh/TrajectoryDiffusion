import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from method_diffusion.utils.traj_vis_metrics import visualize_hist_nbrs_fut_pred

UNIT_CONVERSION = 0.3048

def prepare_input_data(batch, feature_dim, device='cuda'):
    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    lane = batch['lane']  # [B, T, 1]
    cclass = batch['cclass']  # [B, T, 1]
    fut = batch['fut']  # [B, T, 2]
    op_mask = batch['op_mask']  # [B, T, 2]
    hist_nbrs = batch['nbrs']  # [B, T, N, 2] or [B, N, T, 2]
    va_nbrs = batch['nbrs_va']  # [B, T, N, 2] or [B, N, T, 2]
    lane_nbrs = batch['nbrs_lane']  # [B, T, N, 1] or [B, N, T, 1]
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
    op_mask = op_mask.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, fut, op_mask, hist_nbrs, mask, temporal_mask

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, vis_cfg=None):
    model.train()
    total_loss_sum = 0.0
    total_ade_sum = 0.0  # feet
    total_fde_sum = 0.0  # feet
    total_ade_m_sum = 0.0
    total_fde_m_sum = 0.0
    num_batches = 0
    vis_result = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Train Epoch {epoch}", ncols=150)

    for batch_idx, batch in pbar:
        hist, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, device=device
        )

        loss, pred, ade, fde = model.forward_train(hist, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask)
        # _, _, _, _ = model.forward_eval(hist, hist_nbrs, mask, temporal_mask, fut, device)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        ade_ft = ade.item()
        fde_ft = fde.item()
        ade_m = ade_ft * UNIT_CONVERSION
        fde_m = fde_ft * UNIT_CONVERSION

        total_loss_sum += loss.item()
        total_ade_sum += ade_ft
        total_fde_sum += fde_ft
        total_ade_m_sum += ade_m
        total_fde_m_sum += fde_m
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.8f}',
            'avg_loss': f'{total_loss_sum / num_batches:.8f}',
            'avg_ade_ft': f'{total_ade_sum / num_batches:.4f}',
            'avg_fde_ft': f'{total_fde_sum / num_batches:.4f}',
            'avg_ade_m': f'{total_ade_m_sum / num_batches:.4f}',
            'avg_fde_m': f'{total_fde_m_sum / num_batches:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
        })

        if vis_cfg is not None and vis_result is None:
            save_path = vis_cfg["save_dir"] / f"train_epoch_{epoch:03d}.png"
            title = f"Train Epoch {epoch}"
            sample_metrics, saved_file = visualize_hist_nbrs_fut_pred(
                hist=hist,
                nbrs=hist_nbrs,
                fut=fut,
                pred=pred,
                op_mask=op_mask,
                sample_index=vis_cfg["sample_index"],
                save_path=str(save_path),
                title_prefix=title,
                temporal_mask=temporal_mask,
            )
            vis_result = {
                "metrics": sample_metrics,
                "file": saved_file,
            }

    result = {
        "loss": total_loss_sum / max(num_batches, 1),
        "ade": total_ade_sum / max(num_batches, 1),
        "fde": total_fde_sum / max(num_batches, 1),
        "ade_m": total_ade_m_sum / max(num_batches, 1),
        "fde_m": total_fde_m_sum / max(num_batches, 1),
    }
    if vis_result is not None:
        result["vis"] = vis_result
    return result


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, epoch, feature_dim, vis_cfg=None):
    model.eval()
    total_loss_sum = 0.0
    total_ade_sum = 0.0  # feet
    total_fde_sum = 0.0  # feet
    total_ade_m_sum = 0.0
    total_fde_m_sum = 0.0
    num_batches = 0
    vis_result = None

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Val   Epoch {epoch}", ncols=150)

    for _, batch in pbar:
        hist, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, device=device
        )
        eval_loss, eval_pred, eval_ade, eval_fde = model.forward_eval(
            hist, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask
        )
        ade_ft = eval_ade.item()
        fde_ft = eval_fde.item()
        ade_m = ade_ft * UNIT_CONVERSION
        fde_m = fde_ft * UNIT_CONVERSION
        total_loss_sum += eval_loss.item()
        total_ade_sum += ade_ft
        total_fde_sum += fde_ft
        total_ade_m_sum += ade_m
        total_fde_m_sum += fde_m
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{eval_loss.item():.8f}',
            'avg_loss': f'{total_loss_sum / num_batches:.8f}',
            'avg_ade_ft': f'{total_ade_sum / num_batches:.4f}',
            'avg_fde_ft': f'{total_fde_sum / num_batches:.4f}',
            'avg_ade_m': f'{total_ade_m_sum / num_batches:.4f}',
            'avg_fde_m': f'{total_fde_m_sum / num_batches:.4f}',
        })

        if vis_cfg is not None and vis_result is None:
            save_path = vis_cfg["save_dir"] / f"val_epoch_{epoch:03d}.png"
            title = f"Val Epoch {epoch}"
            sample_metrics, saved_file = visualize_hist_nbrs_fut_pred(
                hist=hist,
                nbrs=hist_nbrs,
                fut=fut,
                pred=eval_pred,
                op_mask=op_mask,
                sample_index=vis_cfg["sample_index"],
                save_path=str(save_path),
                title_prefix=title,
                temporal_mask=temporal_mask,
            )
            vis_result = {
                "metrics": sample_metrics,
                "file": saved_file,
            }

    result = {
        "loss": total_loss_sum / max(num_batches, 1),
        "ade": total_ade_sum / max(num_batches, 1),
        "fde": total_fde_sum / max(num_batches, 1),
        "ade_m": total_ade_m_sum / max(num_batches, 1),
        "fde_m": total_fde_m_sum / max(num_batches, 1),
    }
    if vis_result is not None:
        result["vis"] = vis_result
    return result


def load_checkpoint_if_needed(args, model, optimizer, scheduler, device):
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
        state_dict = state['model_state_dict']
        filtered_state_dict = {}
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if key in ['pos_mean', 'pos_std', 'va_mean', 'va_std']:
                continue
            filtered_state_dict[key] = v

        model.load_state_dict(filtered_state_dict, strict=False)
        if 'optimizer_state_dict' in state:
            optimizer.load_state_dict(state['optimizer_state_dict'])
        if 'scheduler_state_dict' in state:
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

    log_dir = Path(args.checkpoint_dir) / 'logs'
    writer = SummaryWriter(log_dir=str(log_dir))
    vis_train_cfg = None
    vis_val_cfg = None
    if args.save_epoch_vis:
        vis_root = Path(args.checkpoint_dir) / "epoch_vis"
        vis_train_cfg = {"save_dir": vis_root / "train", "sample_index": int(args.epoch_vis_sample_idx)}
        vis_val_cfg = {"save_dir": vis_root / "val", "sample_index": int(args.epoch_vis_sample_idx)}
        vis_train_cfg["save_dir"].mkdir(parents=True, exist_ok=True)
        vis_val_cfg["save_dir"].mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_root = Path(args.data_root)
    train_path = str(data_root / 'TrainSet.mat')
    val_path = str(data_root / 'ValSet.mat')

    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)
    val_dataset = NgsimDataset(val_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False
    )

    model = DiffusionFut(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )

    start_epoch, best_loss = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device
    )

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, vis_cfg=vis_train_cfg
        )
        val_metrics = evaluate_epoch(
            model, val_loader, device, epoch + 1,
            args.feature_dim, vis_cfg=vis_val_cfg
        )

        print(
            f"Epoch [{epoch + 1}] "
            f"Train Loss: {train_metrics['loss']:.6f} | "
            f"Train ADE/FDE: {train_metrics['ade']:.4f}/{train_metrics['fde']:.4f} ft "
            f"({train_metrics['ade_m']:.4f}/{train_metrics['fde_m']:.4f} m) || "
            f"Val Loss: {val_metrics['loss']:.6f} | "
            f"Val ADE/FDE: {val_metrics['ade']:.4f}/{val_metrics['fde']:.4f} ft "
            f"({val_metrics['ade_m']:.4f}/{val_metrics['fde_m']:.4f} m)"
        )
        if "vis" in train_metrics:
            print(
                f"[VIS][Train] {train_metrics['vis']['file']} | "
                f"ADE: {train_metrics['vis']['metrics']['ade_m']:.4f} m | "
                f"FDE: {train_metrics['vis']['metrics']['fde_m']:.4f} m"
            )
        if "vis" in val_metrics:
            print(
                f"[VIS][Val] {val_metrics['vis']['file']} | "
                f"ADE: {val_metrics['vis']['metrics']['ade_m']:.4f} m | "
                f"FDE: {val_metrics['vis']['metrics']['fde_m']:.4f} m"
            )

        writer.add_scalar('Train/Loss', train_metrics['loss'], epoch + 1)
        writer.add_scalar('Train/ADE_ft', train_metrics['ade'], epoch + 1)
        writer.add_scalar('Train/FDE_ft', train_metrics['fde'], epoch + 1)
        writer.add_scalar('Train/ADE_m', train_metrics['ade_m'], epoch + 1)
        writer.add_scalar('Train/FDE_m', train_metrics['fde_m'], epoch + 1)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch + 1)
        writer.add_scalar('Val/ADE_ft', val_metrics['ade'], epoch + 1)
        writer.add_scalar('Val/FDE_ft', val_metrics['fde'], epoch + 1)
        writer.add_scalar('Val/ADE_m', val_metrics['ade_m'], epoch + 1)
        writer.add_scalar('Val/FDE_m', val_metrics['fde_m'], epoch + 1)
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch + 1)

        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'best_loss': best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            state['best_loss'] = best_loss
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

    writer.close()

if __name__ == '__main__':
    main()
