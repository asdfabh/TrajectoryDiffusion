import sys
import os

# 添加项目根目录到环境变量
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
import numpy as np
from tqdm import tqdm
import builtins

from method_diffusion.models.net import TrajectoryModel
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import block_mask


# -----------------------------------------------------------------------------
# Functions for DDP
# -----------------------------------------------------------------------------

def setup_ddp():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return rank, local_rank, world_size
    else:
        print("Not using distributed mode")
        return 0, 0, 1


def cleanup_ddp():
    dist.destroy_process_group()


def reduce_value(value, average=True):
    """聚合所有进程的数值用于日志显示"""
    if not dist.is_initialized():
        return value
    world_size = dist.get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value


def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    """
    准备输入数据，处理 Ego 和 Neighbors，并生成掩码。
    """

    hist = batch['hist']  # [B, T, 2]
    va = batch['va']  # [B, T, 2]
    nbrs = batch['nbrs']  # [Total_Nbrs, T, 2]
    nbrs_va = batch['nbrs_va']
    ego_fut = batch['fut']  # [B, T_f, 2]
    nbrs_fut = batch['nbrs_fut']  # [Total_Nbrs, T_f, 2]
    nbr_valid_mask = batch['mask'].to(device)  # [B, N] 表示位置是否存在nbrs

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
    hist_nbrs = hist_nbrs_temp.permute(0, 2, 1, 3).contiguous()  # [B, T, N, dim]

    scatter_mask_fut = nbr_valid_mask.view(B, N, 1, 1).expand(B, N, T_f, dim_fut)
    fut_nbrs_temp = torch.zeros(B, N, T_f, dim_fut, device=device)
    fut_nbrs_temp = fut_nbrs_temp.masked_scatter_(scatter_mask_fut.bool(), nbrs_fut)
    fut_nbrs = fut_nbrs_temp.permute(0, 2, 1, 3).contiguous()  # [B, T_f, N, dim_fut]
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
    agent_mask = torch.cat([ego_mask, nbr_valid_mask.bool()], dim=1)  # [B, N]

    return src, hist_nbrs, hist_masked, hist_nbrs_masked, mask, nbrs_mask, future, agent_mask


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def train_epoch(model, dataloader, optimizer, device, epoch, input_dim, rank,
                mask_type='random', mask_prob=0.4):
    model.train()
    total_loss_past = torch.zeros(1).to(device)
    total_loss_fut = torch.zeros(1).to(device)
    num_batches = 0

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        # mask_type = 'random' if torch.rand(1).item() < 0.65 else 'block'
        hist, hist_nbrs, hist_masked, hist_nbrs_masked, mask, nbrs_mask, future, agent_mask = prepare_input_data(
            batch, input_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        # 拼接 Ego 和 Neighbors
        hist = torch.cat([hist, hist_nbrs], dim=2)  # [B, T, N+1, dim]
        hist_masked = torch.cat([hist_masked, hist_nbrs_masked], dim=2)  # [B, T, N+1, dim+1]

        # Forward
        loss_past, loss_fut, pred_hist, pred_fut, hist_ade, hist_fde, fut_ade, fut_fde = model.forward(hist,
                                                                                                       hist_masked,
                                                                                                       future, device,
                                                                                                       agent_mask=agent_mask)
        loss = loss_fut + loss_past

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 统计 (累加 Tensor 以便后续 reduce)
        total_loss_past += loss_past.detach()
        total_loss_fut += loss_fut.detach()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({
                # 'L_p': f'{loss_past.item():.4f}', # Past loss is dummy 0.0
                'L_f': f'{loss_fut.item():.4f}',
                # 'Av_p': f'{total_loss_past.item() / num_batches:.4f}',
                'Av_f': f'{total_loss_fut.item() / num_batches:.4f}',
                # 'hADE': f'{hist_ade:.4f}',
                # 'hFDE': f'{hist_fde:.4f}',
                'fADE': f'{fut_ade:.4f}',
                'fFDE': f'{fut_fde:.4f}',
            })

    # 聚合所有 GPU 的 Loss
    avg_loss_past = reduce_value(total_loss_past, average=True).item() / num_batches
    avg_loss_fut = reduce_value(total_loss_fut, average=True).item() / num_batches

    return avg_loss_past, avg_loss_fut


def load_pretrained_parts(args, model, device, rank):
    # 【修改点】移除 past_model 的加载逻辑，因为模型结构已变更
    # if args.pretrained_past: ... (Deleted)

    if args.pretrained_fut:
        if rank == 0:
            print(f"Loading pretrained Fut model from {args.pretrained_fut}")
        ckpt = torch.load(args.pretrained_fut, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

        fut_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fut_model.'):
                fut_dict[k.replace('fut_model.', '')] = v
            # 兼容旧权重：如果旧权重没有前缀但结构匹配，也可以加载（视情况而定）
            elif not k.startswith('past_model.'):
                fut_dict[k] = v

        if fut_dict:
            msg = model.fut_model.load_state_dict(fut_dict, strict=False)
            if rank == 0:
                print(f"Loaded Fut model: {msg}")
        else:
            if rank == 0:
                print("Warning: No fut_model weights found in checkpoint")


def load_checkpoint_if_needed(args, model, optimizer, scheduler, device, rank):
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
        except ValueError:
            pass
    elif args.resume not in ('none', ''):
        ckpt_path = Path(args.resume)

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)

        model_dict = state['model_state_dict']
        new_state_dict = {}
        for k, v in model_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # 使用 strict=False，因为可能存在新旧模型结构差异
        model.load_state_dict(new_state_dict, strict=False)

        # 尝试加载 optimizer，如果参数不匹配可能会失败，建议在架构大改后重置 optimizer
        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load optimizer state (expected due to architecture change): {e}")

        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)

        if rank == 0:
            print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_loss


def main():
    # 1. DDP Setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    args = get_args_parser().parse_args()

    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'  # 注意路径修正，根据你的环境可能需要调整
    # data_root = Path('/mnt/datasets/ngsimdata') # 如果你在服务器上用绝对路径
    train_path = str(data_root / 'TrainSet.mat')

    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 每个 GPU 的 batch size
        shuffle=False,  # Sampler 负责 shuffle
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    model = TrajectoryModel(args).to(device)

    # 【修改点】移除冻结参数的逻辑，因为只有 fut_model
    # if args.train_mode == 'past_only': ... (Deleted)
    # elif args.train_mode == 'fut_only': ... (Deleted)

    # 仅在分布式环境下使用 DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        print("Running in non-distributed mode (Single GPU/CPU)")

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
    load_pretrained_parts(args, model, device, rank)

    start_epoch, best_loss = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device, rank
    )

    for epoch in range(start_epoch, args.num_epochs):
        # 重要：设置 epoch 以保证每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        mask_type = 'random'
        mask_prob = 0.55

        avg_loss_past, avg_loss_fut = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, mask_type=mask_type, mask_prob=mask_prob
        )

        current_total_loss = avg_loss_past + avg_loss_fut

        if rank == 0:
            print(f"Epoch [{epoch + 1}] Total Loss: {current_total_loss:.4f} "
                  f"(Past: {avg_loss_past:.4f}, Fut: {avg_loss_fut:.4f})")

        scheduler.step()

        if rank == 0:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),  # 注意使用 .module
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

            if current_total_loss < best_loss:
                best_loss = current_total_loss
                state['best_loss'] = best_loss
                save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
                torch.save(state, save_path)
                print(f"Saved best model (Loss: {best_loss:.4f}) to {save_path}")

    cleanup_ddp()


if __name__ == '__main__':
    main()