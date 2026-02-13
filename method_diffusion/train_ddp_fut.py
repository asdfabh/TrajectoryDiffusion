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
from torch.utils.tensorboard import SummaryWriter
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.utils.traj_vis_metrics import visualize_hist_nbrs_fut_pred

UNIT_CONVERSION = 0.3048


def setup_ddp():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return rank, local_rank, world_size
    else:
        print("Not using distributed mode")
        return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
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

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank, vis_cfg=None):
    model.train()
    total_loss_sum = 0.0
    total_ade_sum = 0.0  # feet
    total_fde_sum = 0.0  # feet
    total_ade_m_sum = 0.0
    total_fde_m_sum = 0.0
    num_batches = 0
    vis_result = None

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        hist, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, device=device
        )

        # loss, pred, ade, fde = model(hist, hist_masked, device)
        loss, pred, ade, fde = model(hist, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Align with TAME training style.
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

        if rank == 0:
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

    metrics = torch.tensor(
        [total_loss_sum, total_ade_sum, total_fde_sum, total_ade_m_sum, total_fde_m_sum, float(num_batches)],
        device=device, dtype=torch.float64
    )
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    denom = max(metrics[5].item(), 1.0)
    result = {
        "loss": metrics[0].item() / denom,
        "ade": metrics[1].item() / denom,
        "fde": metrics[2].item() / denom,
        "ade_m": metrics[3].item() / denom,
        "fde_m": metrics[4].item() / denom,
    }
    if rank == 0 and vis_result is not None:
        result["vis"] = vis_result
    return result


@torch.no_grad()
def evaluate_epoch(model, dataloader, device, epoch, feature_dim, rank, vis_cfg=None):
    model.eval()
    total_loss_sum = 0.0
    total_ade_sum = 0.0  # feet
    total_fde_sum = 0.0  # feet
    total_ade_m_sum = 0.0
    total_fde_m_sum = 0.0
    num_batches = 0
    vis_result = None

    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Val   Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for _, batch in pbar:
        hist, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, device=device
        )
        base_model = model.module if isinstance(model, DDP) else model
        loss, pred, ade, fde = base_model.forward_eval(
            hist, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask
        )
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

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.8f}',
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

    metrics = torch.tensor(
        [total_loss_sum, total_ade_sum, total_fde_sum, total_ade_m_sum, total_fde_m_sum, float(num_batches)],
        device=device, dtype=torch.float64
    )
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    denom = max(metrics[5].item(), 1.0)
    result = {
        "loss": metrics[0].item() / denom,
        "ade": metrics[1].item() / denom,
        "fde": metrics[2].item() / denom,
        "ade_m": metrics[3].item() / denom,
        "fde_m": metrics[4].item() / denom,
    }
    if rank == 0 and vis_result is not None:
        result["vis"] = vis_result
    return result

def load_checkpoint_if_needed(args, model, optimizer, scheduler, device, rank):
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
        except ValueError:
            pass
    elif args.resume_fut not in ('none', ''):
        ckpt_path = Path(args.resume_fut)

    if ckpt_path and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)

        model_dict = state['model_state_dict']
        new_state_dict = {}
        for k, v in model_dict.items():
            key = k[7:] if k.startswith('module.') else k
            if key in ['pos_mean', 'pos_std', 'va_mean', 'va_std']:
                continue
            new_state_dict[key] = v

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
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / "fut")

    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(Path(args.checkpoint_dir) / "logs"))
        vis_train_cfg = None
        vis_val_cfg = None
        if args.save_epoch_vis:
            vis_root = Path(args.checkpoint_dir) / "epoch_vis"
            vis_train_cfg = {"save_dir": vis_root / "train", "sample_index": int(args.epoch_vis_sample_idx)}
            vis_val_cfg = {"save_dir": vis_root / "val", "sample_index": int(args.epoch_vis_sample_idx)}
            vis_train_cfg["save_dir"].mkdir(parents=True, exist_ok=True)
            vis_val_cfg["save_dir"].mkdir(parents=True, exist_ok=True)
    else:
        writer = None
        vis_train_cfg = None
        vis_val_cfg = None

    # Use args.data_root
    data_root = Path(args.data_root)
    train_path = str(data_root / 'TrainSet.mat')
    val_path = str(data_root / 'ValSet.mat')

    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)
    val_dataset = NgsimDataset(val_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

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
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False
    )

    model = DiffusionFut(args).to(device)

    # 仅在分布式环境下使用 DDP
    if dist.is_initialized():
        if device.type == "cuda":
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            model = DDP(model, find_unused_parameters=False)
    else:
        print("Running in non-distributed mode (Single GPU/CPU)")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
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
        val_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, vis_cfg=vis_train_cfg
        )
        val_metrics = evaluate_epoch(
            model, val_loader, device, epoch + 1,
            args.feature_dim, rank, vis_cfg=vis_val_cfg
        )

        if rank == 0:
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
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch + 1)

        scheduler.step()

        if rank == 0:
            model_to_save = model.module if isinstance(model, DDP) else model
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'best_loss': best_loss,
            }

            if (epoch + 1) % args.save_interval == 0:
                save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(state, save_path)
                print(f"Saved checkpoint to {save_path}")

            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                state['best_loss'] = best_loss
                save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
                torch.save(state, save_path)
                print(f"Saved best model (Loss: {best_loss:.4f}) to {save_path}")

    if writer is not None:
        writer.close()
    cleanup_ddp()


if __name__ == '__main__':
    main()
