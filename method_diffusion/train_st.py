import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.net_st import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask_traj, block_mask_traj
import numpy as np
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.4, device='cuda'):

    # torch
    hist = batch['hist']  # [B, T, 2]
    nbrs = batch['nbrs']  # [N_total, T, 2]
    va = batch['va']  # [B, T, 2]
    nbrs_va = batch['nbrs_va']  # [N_total, T, 2]
    lane = batch['lane']  # [B, T, 1]
    nbrs_lane = batch['nbrs_lane']  # [N_total, T, 1]
    cclass = batch['cclass']  # [B, T, 1]
    nbrs_class = batch['nbrs_class']  # [N_total, T, 1]
    nbrs_num = batch['nbrs_num'].squeeze(-1)  # [B]

    # 根据 input_dim 拼接特征
    if input_dim == 6:
        src = torch.cat((hist, cclass, va, lane), dim=-1).to(device)  # [B, T, 6]
        nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va, nbrs_lane), dim=-1).to(device)  # [N_total, T, 6]
    elif input_dim == 5:
        src = torch.cat((hist, cclass, va), dim=-1)  # [B, T, 5]
        nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va), dim=-1)  # [N_total, T, 5]
    else:  # input_dim == 2
        src = hist
        nbrs_src = nbrs

    B, T, _ = hist.shape
    N_total = nbrs.shape[0]

    if mask_type == 'random':
        hist_mask = random_mask_traj(hist, p=mask_prob)  # [B, T]
    else:  # 'block'
        hist_mask = torch.stack([
            block_mask_traj(hist[b], missing_ratio=mask_prob)
            for b in range(B)
        ])  # [B, T]

    hist_mask = hist_mask.float()

    # 检测空轨迹: [N_total, T, 2] -> [N_total]
    nbrs_valid = (nbrs != 0).any(dim=-1).any(dim=-1)  # 至少有一个非零点

    if mask_type == 'random':
        nbrs_mask_raw = random_mask_traj(nbrs, p=mask_prob)
    else:  # 'block'
        nbrs_mask_raw = torch.stack([
            block_mask_traj(nbrs[n], missing_ratio=mask_prob)
            for n in range(N_total)
        ])

    # 将空轨迹的掩码全部置 0
    nbrs_mask = nbrs_mask_raw.float() * nbrs_valid.unsqueeze(-1).float()

    src = src.to(device)
    nbrs_src = nbrs_src.to(device)
    hist_mask = hist_mask.to(device)
    nbrs_mask = nbrs_mask.to(device)
    nbrs_num = nbrs_num.to(device)

    # print(f"device: hist = {src.device}, nbrs = {nbrs_src.device}, hist_mask = {hist_mask.device}, nbrs_mask = {nbrs_mask.device}, nbrs_num = {nbrs_num.device}")
    return src, nbrs_src, hist_mask, nbrs_mask, nbrs_num

def train_epoch(model, dataloader, optimizer, device, epoch, input_dim,
                mask_type='random', mask_prob=0.4):

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=100)

    for batch_idx, batch in pbar:
        # 数据拼接和掩码生成
        hist, nbrs, hist_mask, nbrs_mask, nbrs_num = prepare_input_data(
            batch, input_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        # 前向传播
        loss, pred_ego, pred_nbrs = model.forward_train(hist, hist_mask, nbrs, nbrs_mask, nbrs_num)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'mask': f'{mask_type}({mask_prob})'
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

# def main():
#     args = get_args_parser().parse_args()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 数据集路径
#     data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'
#     train_path = str(data_root / 'TrainSet.mat')
#
#     # 创建数据集和 DataLoader
#     train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=train_dataset.collate_fn
#     )
#
#     # 创建模型
#     model = DiffusionPast(args).to(device)
#
#     # 优化器和学习率调度器
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
#     # 训练循环
#     for epoch in range(args.num_epochs):
#         print(f"\n========== Epoch {epoch+1}/{args.num_epochs} ==========")
#
#         # 动态切换掩码策略 (可选)
#         # mask_type = 'random' if epoch < args.num_epochs // 2 else 'block'
#         mask_type = 'random'
#         mask_prob = 0.4
#
#         avg_loss = train_epoch(
#             model, train_loader, optimizer, device, epoch+1,
#             args.feature_dim, mask_type=mask_type, mask_prob=mask_prob
#         )
#
#         print(f"Epoch [{epoch+1}] Average Loss: {avg_loss:.4f}")
#         scheduler.step()
#
#         # 保存检查点
#         if (epoch + 1) % args.save_interval == 0:
#             checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pth"
#             checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
#             torch.save({
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'loss': avg_loss,
#             }, checkpoint_path)
#             print(f"Checkpoint saved to {checkpoint_path}")

def main():
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集路径
    data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'
    train_path = str(data_root / 'TrainSet.mat')

    # 创建数据集和 DataLoader
    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn
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
        mask_prob = 0.4

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
