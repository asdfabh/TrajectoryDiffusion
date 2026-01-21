import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import builtins
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask
from method_diffusion.models.fut_model import DiffusionFut


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

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank, mask_type='random', mask_prob=0.4):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        # loss, pred, ade, fde = model(hist, hist_masked, device)
        loss, pred, ade, fde = model(hist, hist_nbrs, mask, temporal_mask, fut, device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.8f}',
                'avg_loss': f'{total_loss / num_batches:.8f}',
                'ade': f'{ade.mean().item():.4f}',
                'fde': f'{fde.mean().item():.4f}',
            })

    avg_loss = total_loss / num_batches
    return avg_loss

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

    # Use args.data_root
    data_root = Path(args.data_root)
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

    model = DiffusionFut(args).to(device)

    # 仅在分布式环境下使用 DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
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

    start_epoch, best_loss = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device, rank
    )

    for epoch in range(start_epoch, args.num_epochs):
        # 重要：设置 epoch 以保证每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        mask_type = 'random'
        mask_prob = args.mask_prob

        avg_loss = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, mask_type=mask_type, mask_prob=mask_prob
        )

        if rank == 0:
            print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")

        scheduler.step()

        if rank == 0:
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }

            if (epoch + 1) % args.save_interval == 0:
                save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(state, save_path)
                print(f"Saved checkpoint to {save_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                state['best_loss'] = best_loss
                save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
                torch.save(state, save_path)
                print(f"Saved best model (Loss: {best_loss:.4f}) to {save_path}")

    cleanup_ddp()


if __name__ == '__main__':
    main()

