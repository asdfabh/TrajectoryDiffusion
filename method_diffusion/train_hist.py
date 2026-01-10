import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.visualization import plot_traj_with_mask, plot_traj
from method_diffusion.utils.mask_util import random_mask, continuous_mask
import numpy as np
from tqdm import tqdm
from einops import repeat

def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    # type is torch
    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    lane = batch['lane']  # [B, T, 1]
    cclass = batch['cclass']  # [B, T, 1]
    mask_type = mask_type

    # 根据 feature_dim 拼接特征
    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)  # [B, T, 6]
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device) # [B, T, 5]
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
    else:  # feature_dim == 2
        hist = hist.to(device)

    # 生成掩码并应用掩码
    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob) # 保留位置为 True，p为至少保留比例 [B, T, 1]
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob) # 保留位置为 True，p为丢弃比例 [B, T, 1]
    else:
        print(f'Unknown mask type: {mask_type}, defaulting to random mask.')
        hist_mask = random_mask(hist, p=mask_prob)  # 保留位置为 True，p为至少保留比例 [B, T, 1]

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1) # [B, T, feature_dim+1]

    # hist: [B, T, feature_dim] 完整历史轨迹
    # hist_masked: [B, T, feature_dim+1] 掩码后历史轨迹，最后一维为掩码标记
    # hist_mask: [B, T, 1] 掩码标记，True表示保留，False表示掩码
    return hist, hist_masked, hist_mask

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, mask_type='random', mask_prob=0.4):

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=150)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        # 前向传播，输入为掩码后的轨迹 hist [B, T, dim]
        loss, pred, ade, fde = model.forward_train(hist, hist_masked, device)
        # _, _, _, _ = model.forward_eval(hist, hist_masked, device)

        # hist = hist[0, :, :2].detach().cpu().numpy()
        # hist_masked = hist_masked[0, :, :2].detach().cpu().numpy()
        # pred_ego = pred[0, :, :2].detach().cpu().numpy()
        #
        # plot_traj_with_mask(
        #     hist_original=[hist],
        #     hist_masked=[hist_masked],
        #     hist_pred=[pred_ego],
        #     fig_num1=1,
        #     fig_num2=1,
        # )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
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

    # Set checkpoint directory validation for HistModel
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'hist')
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集路径
    data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
    train_path = str(data_root / 'TrainSet.mat')

    # 创建数据集和 DataLoader
    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,                # GPU 训练强烈建议打开
        persistent_workers=True,        # PyTorch>=1.7，epoch 间复用 worker
        drop_last=True
    )

    # 创建模型
    model = DiffusionPast(args).to(device)

    # 优化器和学习率调度器
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

    # 训练循环
    for epoch in range(args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        # 动态切换掩码策略 (可选)
        # mask_type = 'random' if epoch < args.num_epochs // 2 else 'block'
        mask_type = 'random'
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
