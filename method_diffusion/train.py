import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.net import TrajectoryModel
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import block_mask
from tqdm import tqdm
from method_diffusion.utils.visualization import visualize_batch_trajectories

def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    """
    准备输入数据，处理 Ego 和 Neighbors，并生成掩码。
    适配 Dataset 修改：Neighbors 现在是最近的 K 个，而非固定网格。

    Returns:
        hist: [B, T, 1, dim] 真实自车轨迹
        hist_nbrs: [B, T, N, dim] 真实邻居轨迹 (N为最大邻居数)
        hist_masked: [B, T, 1, dim+1] 掩码后的自车输入 (含 mask 通道)
        hist_nbrs_masked: [B, T, N, dim+1] 掩码后的邻居输入 (含 mask 通道)
        mask: [B, T, 1, 1] 自车掩码 (1=Keep, 0=Drop)
        nbrs_mask: [B, T, N, 1] 邻居掩码 (1=Keep, 0=Drop)，True表示存在
    """

    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    nbrs = batch['nbrs']  # [Total_Nbrs, T, 2]
    nbrs_va = batch['nbrs_va']
    ego_fut = batch['fut']  # [B, T_f, 2]
    nbrs_fut = batch['nbrs_fut']  # [Total_Nbrs, T_f, 2]
    nbr_valid_mask = batch['mask'].to(device) # [B, N] 表示位置是否存在nbrs

    src = torch.cat((hist, va), dim=-1).to(device)
    nbrs_src = torch.cat((nbrs, nbrs_va), dim=-1).to(device)
    ego_fut = ego_fut.to(device).unsqueeze(2)  # [B, T_f, 1, 2]
    nbrs_fut = nbrs_fut.to(device)  # [Total_Nbrs, T_f, 2]

    B, T, dim = src.shape
    _, T_f, _, dim_fut = ego_fut.shape
    N = nbr_valid_mask.shape[1]

    mask_flat = nbr_valid_mask.view(B, -1)

    scatter_mask = nbr_valid_mask.view(B, N, 1, 1).expand(B, N, T, dim)
    hist_nbrs_temp = torch.zeros(B, N, T, dim, device=device)
    hist_nbrs_temp = hist_nbrs_temp.masked_scatter_(scatter_mask.bool(), nbrs_src)
    hist_nbrs = hist_nbrs_temp.permute(0, 2, 1, 3).contiguous() # [B, T, N, dim]

    scatter_mask_fut = nbr_valid_mask.view(B, N, 1, 1).expand(B, N, T_f, dim_fut)
    fut_nbrs_temp = torch.zeros(B, N, T_f, dim_fut, device=device)
    fut_nbrs_temp = fut_nbrs_temp.masked_scatter_(scatter_mask_fut.bool(), nbrs_fut)
    fut_nbrs = fut_nbrs_temp.permute(0, 2, 1, 3).contiguous() # [B, T_f, N, dim_fut]
    future = torch.cat([ego_fut, fut_nbrs], dim=2)

    if mask_type == 'random':
        mask = torch.rand(B, T, 1, 1, device=device) < mask_prob
        mask = mask.float()
        nbrs_mask_rand = torch.rand(B, T, N, 1, device=device) < mask_prob
        nbrs_exists = mask_flat.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1).float()
        nbrs_mask = nbrs_mask_rand.float() * nbrs_exists

    elif mask_type == 'block':
        m = block_mask(B, T, device=device).unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        mask = m.float()
        nbrs_exists = mask_flat.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1).float()
        nbrs_mask = mask.expand(-1, -1, N, -1) * nbrs_exists
    else:
        mask = torch.ones(B, T, 1, 1, device=device)
        nbrs_exists = mask_flat.unsqueeze(1).unsqueeze(-1).expand(B, T, N, 1).float()
        nbrs_mask = nbrs_exists

    # Ego: [B, T, 1, dim] -> [B, T, 1, dim+1]
    src = src.unsqueeze(2)
    hist_masked_val = src * mask
    hist_masked = torch.cat([hist_masked_val, mask], dim=-1)
    hist_masked[:, -1, :, :] = 1  # 确保最后一个时间步不被mask

    # Nbrs: [B, T, N, dim] -> [B, T, N, dim+1]
    hist_nbrs_masked_val = hist_nbrs * nbrs_mask
    hist_nbrs_masked = torch.cat([hist_nbrs_masked_val, nbrs_mask], dim=-1)

    # Construct agent_mask [B, 1+N] (Existence Mask)
    ego_mask = torch.ones(B, 1, device=device).bool()
    agent_mask = torch.cat([ego_mask, nbr_valid_mask.bool()], dim=1) # [B, N]

    return src, hist_nbrs, hist_masked, hist_nbrs_masked, mask, nbrs_mask, future, agent_mask

def train_epoch(model, dataloader, optimizer, device, epoch, input_dim,
                mask_type='random', mask_prob=0.4):

    model.train()
    total_loss_past = 0.0
    total_loss_fut = 0.0
    num_batches = 0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                desc=f"Epoch {epoch}")

    for batch_idx, batch in pbar:
        mask_type = 'random' if torch.rand(1).item() < 0.65 else 'block'
        hist, hist_nbrs, hist_masked, hist_nbrs_masked, mask, nbrs_mask, future, agent_mask = prepare_input_data(
            batch, input_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        hist = torch.cat([hist, hist_nbrs], dim=2)  # [B, T, 40, dim]
        hist_masked = torch.cat([hist_masked, hist_nbrs_masked], dim=2)  # [B, T, 40, dim+1]

        loss_past, loss_fut, pred_hist, pred_fut, hist_ade, hist_fde, fut_ade, fut_fde = model.forward(hist, hist_masked, future, device, agent_mask=agent_mask)
        model.inference(hist, hist_masked, future, device, agent_mask=agent_mask)
        loss = loss_past + loss_fut

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计
        total_loss_past += loss_past.item()
        total_loss_fut += loss_fut.item()
        num_batches += 1

        pbar.set_postfix({
            'Loss_past': f'{loss_past.item():.4f}',
            'Loss_fut': f'{loss_fut.item():.4f}',
            'Avg_past': f'{total_loss_past/num_batches:.4f}',
            'Avg_fut': f'{total_loss_fut/num_batches:.4f}',
            'hist_ade': f'{hist_ade:.4f}',
            'hist_fde': f'{hist_fde:.4f}',
            'fut_ade': f'{fut_ade:.4f}',
            'fut_fde': f'{fut_fde:.4f}',
        })

    avg_loss_past = total_loss_past / num_batches
    avg_loss_fut = total_loss_fut / num_batches
    return avg_loss_past, avg_loss_fut

def load_pretrained_parts(args, model, device):
    if args.pretrained_past:
        print(f"Loading pretrained Past model from {args.pretrained_past}")
        ckpt = torch.load(args.pretrained_past, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        # 尝试提取 past_model 部分
        past_dict = {}
        for k, v in state_dict.items():
            if k.startswith('past_model.'):
                past_dict[k.replace('past_model.', '')] = v
            elif not k.startswith('fut_model.'): # 假设没有前缀的情况
                past_dict[k] = v

        if past_dict:
            msg = model.past_model.load_state_dict(past_dict, strict=False)
            print(f"Loaded Past model: {msg}")
        else:
            print("Warning: No past_model weights found in checkpoint")

    if args.pretrained_fut:
        print(f"Loading pretrained Fut model from {args.pretrained_fut}")
        ckpt = torch.load(args.pretrained_fut, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        fut_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fut_model.'):
                fut_dict[k.replace('fut_model.', '')] = v
            elif not k.startswith('past_model.'):
                fut_dict[k] = v

        if fut_dict:
            msg = model.fut_model.load_state_dict(fut_dict, strict=False)
            print(f"Loaded Fut model: {msg}")
        else:
            print("Warning: No fut_model weights found in checkpoint")

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
    model = TrajectoryModel(args).to(device)

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

    # 加载预训练部分 (如果指定)
    load_pretrained_parts(args, model, device)

    start_epoch, best_loss = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device
    )

    # 训练循环
    for epoch in range(args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        # 动态切换掩码策略 (可选)
        # mask_type = 'random' if epoch < args.num_epochs // 2 else 'block'
        mask_type = 'random'
        mask_prob = 0.6

        avg_loss_past, avg_loss_fut = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, mask_type=mask_type, mask_prob=mask_prob
        )

        current_total_loss = avg_loss_past + avg_loss_fut
        print(f"Epoch [{epoch + 1}] Total Loss: {current_total_loss:.4f} "
              f"(Past: {avg_loss_past:.4f}, Fut: {avg_loss_fut:.4f})")

        scheduler.step()

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss_past': avg_loss_past,
            'loss_fut': avg_loss_fut,
            'best_loss': best_loss,
        }

        if (epoch + 1) % args.save_interval == 0:
            save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(state, save_path)
            print(f"Saved checkpoint to {save_path}")

        # 保存最佳模型 (基于总 Loss)
        if current_total_loss < best_loss:
            best_loss = current_total_loss
            state['best_loss'] = best_loss
            save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
            torch.save(state, save_path)
            print(f"Saved best model (Loss: {best_loss:.4f}) to {save_path}")


if __name__ == '__main__':
    main()
