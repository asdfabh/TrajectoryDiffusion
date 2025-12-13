import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.net import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.visualization import plot_traj_with_mask, plot_traj
from method_diffusion.utils.mask_util import random_mask_traj, block_mask_traj, block_mask
import numpy as np
from tqdm import tqdm
from einops import repeat

# def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.7, device='cuda'):
#
#     # type is torch
#     hist = batch['hist']  # [B, T, 2]
#     nbrs = batch['nbrs']  # [N_total, T, 2]
#     va = batch['va']  # [B, T, 2]
#     nbrs_va = batch['nbrs_va']  # [N_total, T, 2]
#     lane = batch['lane']  # [B, T, 1]
#     nbrs_lane = batch['nbrs_lane']  # [N_total, T, 1]
#     cclass = batch['cclass']  # [B, T, 1]
#     nbrs_class = batch['nbrs_class']  # [N_total, T, 1]
#     nbrs_num = batch['nbrs_num'].squeeze(-1)  # [B]
#     mask = batch['mask'].to(device) # [B, 3, 13]
#
#     # 根据 input_dim 拼接特征
#     if input_dim == 6:
#         src = torch.cat((hist, cclass, va, lane), dim=-1).to(device)  # [B, T, 6]
#         nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va, nbrs_lane), dim=-1).to(device)  # [N_total, T, 6]
#     elif input_dim == 5:
#         src = torch.cat((hist, cclass, va), dim=-1).to(device)  # [B, T, 5]
#         nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va), dim=-1).to(device)  # [N_total, T, 5]
#     else:  # input_dim == 2
#         src = hist.to(device)
#         nbrs_src = nbrs.to(device)
#
#     B, T, _ = hist.shape
#     N_total = nbrs.shape[0]
#
#     # 在mask的基础上生成历史轨迹掩码，并应用掩码
#     mask = mask.view(mask.shape[0], mask.shape[1] * mask.shape[2])  # [B, 39]
#     mask = mask.unsqueeze(-1).expand(-1, -1, input_dim)  # [B, 39, dim]
#     mask = repeat(mask, 'b c n -> b t c n', t=T)  # [B, T, 39, 6]
#     nbrs_grid = torch.zeros_like(mask).float()
#     nbrs_grid = nbrs_grid.masked_scatter_(mask.bool(), nbrs_src)  # size [B, T, 39, dim]
#
#     has_trajectory = mask[:, :, :, 0].bool()  # 提取有轨迹标记
#     time_mask = torch.rand(B, T, 39, device=device) < mask_prob
#     time_mask_final = has_trajectory & time_mask
#     time_mask_expanded = time_mask_final.unsqueeze(-1).expand(-1, -1, -1, input_dim)
#
#     nbrs_grid_masked = nbrs_grid * (~time_mask_expanded).float()  # 时间掩码应用 size [B, T, 39, dim]
#     obs_nbrs = (has_trajectory & ~time_mask).float().unsqueeze(-1)  #
#     nbrs_grid_masked = torch.cat([nbrs_grid_masked, obs_nbrs], dim=-1)
#
#     # 处理自车历史
#     hist_mask = torch.rand(B, T, device=device) < mask_prob  # [B, T]
#     hist_masked = src.masked_fill(hist_mask.unsqueeze(-1), 0.0)
#     obs_hist = (~hist_mask).float().unsqueeze(-1)  # [B, T, 1]
#     hist_masked = torch.cat([hist_masked, obs_hist], dim=-1).unsqueeze(2)  # [B, T, 1, input_dim+1]
#     src = src.unsqueeze(2)
#
#     return hist_masked, nbrs_grid_masked, nbrs_num, src, nbrs_grid

def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    # type is torch
    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    lane = batch['lane']  # [B, T, 1]
    cclass = batch['cclass']  # [B, T, 1]
    mask_type = mask_type

    # 根据 input_dim 拼接特征
    if input_dim == 6:
        hist = torch.cat((hist, cclass, va, lane), dim=-1).to(device)  # [B, T, 6]
    elif input_dim == 5:
        hist = torch.cat((hist, cclass, va), dim=-1).to(device) # [B, T, 5]
    else:  # input_dim == 2
        hist = hist.to(device)

    B, T, dim = hist.shape

    # 处理自车历史
    if mask_type == 'random':
        hist_mask = torch.rand(B, T, device=device) < mask_prob  # [B, T], True 表示观测位置
    elif mask_type == 'block':
        hist_mask = block_mask(B, T, device=device)
    else:
        print(f'Unknown mask type: {mask_type}, defaulting to random mask.')
        hist_mask = torch.rand(B, T, device=device) < mask_prob

    hist_masked = hist.masked_fill(~hist_mask.unsqueeze(-1), 0.0)  # 被掩码位置置 0
    obs_hist = hist_mask.float().unsqueeze(-1)  # 观测位置为 1，掩码位置为 0 -> [B, T, 1]
    hist_masked = torch.cat([hist_masked, obs_hist], dim=-1).unsqueeze(2)  # [B, T, 1, input_dim+1]
    hist = hist.unsqueeze(2) # [B, T, 1, dim]

    return hist, hist_masked

def train_epoch(model, dataloader, optimizer, device, epoch, input_dim,
                mask_type='random', mask_prob=0.4):

    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}", ncols=100)

    for batch_idx, batch in pbar:
        # 数据拼接、掩码处理 hist: [B, T, 1, dim]，不进行归一化
        hist, hist_masked = prepare_input_data(
            batch, input_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        # 前向传播，输入为掩码后的轨迹 hist [B, T, 1, dim]
        loss, pred_ego = model.forward_train(hist, hist_masked, device)

        hist = hist[0, :, 0, :2].detach().cpu().numpy()
        hist_masked = hist_masked[0, :, 0, :2].detach().cpu().numpy()
        pred_ego = pred_ego[0, :, 0, :2].detach().cpu().numpy()

        plot_traj_with_mask(
            hist_original=[hist],
            hist_masked=[hist_masked],
            hist_pred=[pred_ego],
            fig_num1=1,
            fig_num2=1,
        )

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
    elif args.resume.startswith('epoch'):
        # 支持 epoch3, epoch10 等格式
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
        mask_type = 'block'
        mask_prob = 0.55

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
