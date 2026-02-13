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

UNIT_CONVERSION = 0.3048


def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    hist = batch['hist']
    va = batch['va']
    lane = batch['lane']
    cclass = batch['cclass']
    fut = batch['fut']
    op_mask = batch['op_mask']
    hist_nbrs = batch['nbrs']
    va_nbrs = batch['nbrs_va']
    lane_nbrs = batch['nbrs_lane']
    cclass_nbrs = batch['nbrs_class']
    mask = batch['mask']
    temporal_mask = batch['temporal_mask']

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
    fut = fut.to(device)
    op_mask = op_mask.to(device)

    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob).to(device)
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob).to(device)
    else:
        hist_mask = random_mask(hist, p=mask_prob).to(device)

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1)

    hist_masked = hist_masked.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, hist_masked, hist_mask, fut, op_mask, hist_nbrs, mask, temporal_mask


def train_epoch(model_fut, model_hist, dataloader, optimizer, device, epoch, feature_dim,
                mask_type='random', mask_prob=0.4, freeze_hist=True):
    model_fut.train()

    if freeze_hist:
        model_hist.eval()
    else:
        model_hist.train()

    total_loss = 0.0
    total_loss_hist = 0.0
    total_loss_fut = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=160)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        if freeze_hist:
            with torch.no_grad():
                loss_h, pred_hist, _, _ = model_hist.forward_eval(hist, hist_masked, device)
                loss_h = 0.0
        else:
            loss_h, pred_hist, _, _ = model_hist.forward_train(hist, hist_masked, device)

        loss_f, pred, ade, fde = model_fut.forward_train(
            pred_hist, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask
        )

        loss = loss_f + loss_h

        optimizer.zero_grad()
        loss.backward()

        params = list(model_fut.parameters())
        if not freeze_hist:
            params += list(model_hist.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        curr_loss_h = loss_h.item() if not isinstance(loss_h, float) else 0.0
        total_loss += loss.item()
        total_loss_hist += curr_loss_h
        total_loss_fut += loss_f.item()
        num_batches += 1

        pbar.set_postfix({
            'L_all': f'{total_loss / num_batches:.4f}',
            'L_hist': f'{total_loss_hist / num_batches:.4f}',
            'L_fut': f'{total_loss_fut / num_batches:.4f}',
            'ade_ft': f'{ade.mean().item():.2f}',
            'fde_ft': f'{fde.mean().item():.2f}',
            'ade_m': f'{(ade.mean().item() * UNIT_CONVERSION):.2f}',
            'fde_m': f'{(fde.mean().item() * UNIT_CONVERSION):.2f}',
        })

    return total_loss / num_batches


def load_checkpoint_generic(resume_arg, default_dir, model, device, model_name="Model"):
    """通用的 Checkpoint 加载函数，处理路径逻辑"""
    ckpt_path = None
    default_dir = Path(default_dir)

    if resume_arg == 'latest':
        ckpts = sorted(default_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts: ckpt_path = ckpts[-1]
    elif resume_arg == 'best':
        best_candidate = default_dir / 'checkpoint_best.pth'
        if best_candidate.exists(): ckpt_path = best_candidate
    elif resume_arg.startswith('epoch'):
        try:
            ep = int(resume_arg.replace('epoch', ''))
            ckpt_path = default_dir / f'checkpoint_epoch_{ep}.pth'
        except:
            pass
    elif resume_arg not in ('none', '', None):
        ckpt_path = Path(resume_arg)

    start_epoch = 0
    best_loss = float('inf')

    if ckpt_path and ckpt_path.exists():
        print(f"Loading {model_name} from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=False)
        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)
    else:
        print(f"Warning: No checkpoint found for {model_name}, initializing randomly.")

    return start_epoch, best_loss


def main():
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_ckpt_dir = Path(args.checkpoint_dir)
    hist_ckpt_dir = base_ckpt_dir / 'hist'
    fut_ckpt_dir = base_ckpt_dir / 'fut'

    hist_ckpt_dir.mkdir(parents=True, exist_ok=True)
    fut_ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(fut_ckpt_dir / 'logs'))

    data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
    train_path = str(data_root / 'TrainSet.mat')
    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)
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

    print("\n[Hist Model] Initializing...")
    model_hist = DiffusionPast(args).to(device)
    load_checkpoint_generic(args.resume_hist, hist_ckpt_dir, model_hist, device, "Hist")

    print("\n[Fut Model] Initializing...")
    model_fut = DiffusionFut(args).to(device)
    start_epoch, best_loss = load_checkpoint_generic(args.resume_fut, fut_ckpt_dir, model_fut, device, "Fut")

    freeze_hist = True
    print(f"Training Mode: Joint Training | Freeze Hist: {freeze_hist}")

    if freeze_hist:
        model_hist.eval()
        for param in model_hist.parameters():
            param.requires_grad = False

    params = list(model_fut.parameters())
    if not freeze_hist:
        params += list(model_hist.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        avg_loss = train_epoch(
            model_fut, model_hist, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, mask_type='random', mask_prob=0.5, freeze_hist=freeze_hist
        )

        print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")
        writer.add_scalar('Train/Loss', avg_loss, epoch + 1)
        scheduler.step()

        state_fut = {
            'epoch': epoch + 1,
            'model_state_dict': model_fut.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # 如果联合训练，这里包含了 hist 的 optimizer 状态
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state_fut, fut_ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth")

            # 如果 Hist 也在训练，也保存 Hist 的状态
            if not freeze_hist:
                state_hist = {'model_state_dict': model_hist.state_dict(), 'epoch': epoch + 1}
                torch.save(state_hist, hist_ckpt_dir / f"joint_checkpoint_epoch_{epoch + 1}.pth")

        if avg_loss < best_loss:
            best_loss = avg_loss
            state_fut['best_loss'] = best_loss
            torch.save(state_fut, fut_ckpt_dir / "checkpoint_best.pth")
            if not freeze_hist:
                torch.save({'model_state_dict': model_hist.state_dict()}, hist_ckpt_dir / "joint_checkpoint_best.pth")

    writer.close()


if __name__ == '__main__':
    main()
