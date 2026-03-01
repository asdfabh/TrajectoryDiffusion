import sys
import os
import math
import csv
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
from method_diffusion.models.fut_model import DiffusionFut


def setup_ddp():
    """初始化分布式训练环境"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(local_rank)
        try:
            dist.init_process_group(
                backend="nccl", init_method="env://", world_size=world_size, rank=rank, device_id=local_rank
            )
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier(device_ids=[local_rank])
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
    op_mask = op_mask.to(device)

    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, hist_nbrs, mask, temporal_mask, fut, op_mask


def get_model_loss_stats(model):
    module = model.module if hasattr(model, "module") else model
    stats = getattr(module, "latest_loss_stats", None)
    if not isinstance(stats, dict):
        return None
    return stats


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank):
    model.train()
    total_loss = 0.0
    total_loss_vel = 0.0
    total_loss_pos = 0.0
    stat_batches = 0
    num_batches = 0

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss = model(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        stats = get_model_loss_stats(model)
        if stats is not None:
            total_loss_vel += float(stats.get("loss_vel", 0.0))
            total_loss_pos += float(stats.get("loss_pos", 0.0))
            stat_batches += 1

        if rank == 0:
            if stats is None:
                pbar.set_postfix({
                    'loss': f'{loss.item():.8f}',
                    'avg_loss': f'{total_loss / num_batches:.8f}',
                })
            else:
                pbar.set_postfix({
                    'loss': f'{loss.item():.8f}',
                    'avg_loss': f'{total_loss / num_batches:.8f}',
                    'vel': f'{float(stats.get("loss_vel", 0.0)):.8f}',
                    'pos': f'{float(stats.get("loss_pos", 0.0)):.8f}',
                })

    # Aggregate loss/count across all ranks for a true global average.
    loss_count = torch.tensor(
        [total_loss, float(num_batches), total_loss_vel, total_loss_pos, float(stat_batches)],
        device=device
    )
    if dist.is_initialized():
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
    global_total_loss = float(loss_count[0].item())
    global_total_batches = max(float(loss_count[1].item()), 1.0)
    global_total_vel = float(loss_count[2].item())
    global_total_pos = float(loss_count[3].item())
    global_stat_batches = max(float(loss_count[4].item()), 1.0)
    avg_loss = global_total_loss / global_total_batches
    avg_loss_vel = global_total_vel / global_stat_batches
    avg_loss_pos = global_total_pos / global_stat_batches
    return avg_loss, avg_loss_vel, avg_loss_pos


@torch.no_grad()
def evaluate_on_testset(model, dataloader, device, epoch, feature_dim, eval_ratio=0.1, max_batches=0):
    fut_model = model.module if hasattr(model, "module") else model
    fut_model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    total_loss_vel = 0.0
    total_loss_pos = 0.0
    stat_batches = 0
    num_batches = 0

    total_batches = len(dataloader)
    if total_batches == 0:
        fut_model.train()
        return 0.0, 0.0, 0.0, 0.0, 0.0

    target_batches = total_batches
    if eval_ratio > 0:
        target_batches = max(1, int(math.ceil(total_batches * float(eval_ratio))))
    if max_batches > 0:
        target_batches = min(target_batches, int(max_batches))

    pbar = tqdm(enumerate(dataloader), total=target_batches, desc=f"Eval(TestSet) Ep{epoch}", dynamic_ncols=True)
    for batch_idx, batch in pbar:
        if batch_idx >= target_batches:
            break

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        eval_loss, _, eval_ade, eval_fde = fut_model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        num_batches += 1
        stats = get_model_loss_stats(fut_model)
        if stats is not None:
            total_loss_vel += float(stats.get("loss_vel", 0.0))
            total_loss_pos += float(stats.get("loss_pos", 0.0))
            stat_batches += 1
        pbar.set_postfix({
            'eval_loss': f'{eval_loss.item():.8f}',
            'avg_ade_ft': f'{(total_ade / num_batches):.4f}',
            'avg_fde_ft': f'{(total_fde / num_batches):.4f}',
        })

    fut_model.train()
    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        total_loss / num_batches,
        total_ade / num_batches,
        total_fde / num_batches,
        total_loss_vel / max(stat_batches, 1),
        total_loss_pos / max(stat_batches, 1),
    )

def load_checkpoint_if_needed(args, model, optimizer, scheduler, device, rank):
    start_epoch = 0
    best_ade = float('inf')
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
        best_ade = state.get('best_ade', state.get('best_loss', best_ade))

        if rank == 0:
            print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_ade


def main():
    # 1. DDP Setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    args = get_args_parser().parse_args()

    # Keep the same checkpoint layout as train_fut.py.
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'fut')
    if rank == 0:
        log_dir = Path(args.checkpoint_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "train_eval_metrics.csv"
        if not csv_path.exists():
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer_csv = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss_total",
                        "train_loss_vel",
                        "train_loss_pos",
                        "eval_loss_total",
                        "eval_loss_vel",
                        "eval_loss_pos",
                        "eval_ade_ft",
                        "eval_fde_ft",
                        "eval_ade_m",
                        "eval_fde_m",
                        "lr",
                    ],
                )
                writer_csv.writeheader()

    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        fixed_eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))
        if fixed_eval_ratio <= 0.0:
            fixed_eval_ratio = 0.1
        print(
            f"[FutModel] Inference sampler: steps={args.num_inference_steps}, "
            f"spacing={args.inference_timestep_spacing}, eta={args.ddim_eta}, x0_clip={args.x0_clip}"
        )
        print(
            f"[FutModel] Train strategy: self_condition_prob={args.self_condition_prob}, "
            f"loss=smooth_l1_residual_anchor, y_weight={args.fut_y_loss_weight}, huber_delta={args.fut_huber_delta}"
        )
        print(
            f"[FutModel] PosLoss warmup: min={args.fut_pos_loss_weight_min}, "
            f"max={args.fut_pos_loss_weight_max}, warmup_ratio={args.fut_pos_loss_warmup_ratio}"
        )
        print(
            f"[FutModel] CFG: enabled={int(args.cfg_enabled) > 0}, "
            f"drop_prob={args.cfg_drop_prob}, guidance_scale={args.cfg_guidance_scale}"
        )
        print(
            f"[FutModel] Architecture: hidden_dim_fut={args.hidden_dim_fut}, depth_fut={args.depth_fut}"
        )
        print(
            f"[FutModel] TestSet eval sampling: eval_ratio={fixed_eval_ratio}, "
            f"eval_max_batches={args.eval_max_batches}"
        )
    else:
        fixed_eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))
        if fixed_eval_ratio <= 0.0:
            fixed_eval_ratio = 0.1

    # Use args.data_root
    data_root = Path(args.data_root)
    train_path = str(data_root / 'TrainSet.mat')

    train_dataset = NgsimDataset(
        train_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim
    )

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

    test_loader = None
    if rank == 0:
        test_path = str(data_root / 'TestSet.mat')
        test_dataset = NgsimDataset(
            test_path,
            t_h=30,
            t_f=50,
            d_s=2,
            enc_size=args.encoder_input_dim,
            feature_dim=args.feature_dim
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
            drop_last=False
        )

    model = DiffusionFut(args).to(device)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[FutModel] Parameters: {num_params / 1e6:.3f} M")

    # 仅在分布式环境下使用 DDP
    if dist.is_initialized():
        # CFG 关闭时，无条件分支参数会长期不参与反传。
        # 打开 unused 参数检测，避免 DDP 在下一迭代报 reduction 未完成错误。
        find_unused = (int(args.cfg_enabled) <= 0) or (float(args.cfg_drop_prob) <= 0.0)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused
        )
        if rank == 0:
            print(f"[DDP] find_unused_parameters={find_unused}")
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

    start_epoch, best_ade = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device, rank
    )

    for epoch in range(start_epoch, args.num_epochs):
        # 重要：设置 epoch 以保证每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)
        fut_model = model.module if hasattr(model, "module") else model
        fut_model.setTrainProgress(epoch + 1, args.num_epochs)

        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")
            print(f"[FutModel] PosLoss alpha(epoch={epoch + 1}): {fut_model.getPosLossWeight():.4f}")

        avg_loss, avg_loss_vel, avg_loss_pos = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank
        )

        if rank == 0:
            print(
                f"Epoch [{epoch + 1}] Average Loss: total={avg_loss:.6f}, "
                f"vel={avg_loss_vel:.6f}, pos={avg_loss_pos:.6f}"
            )
            eval_loss, eval_ade, eval_fde, eval_loss_vel, eval_loss_pos = evaluate_on_testset(
                model, test_loader, device, epoch + 1, args.feature_dim,
                eval_ratio=fixed_eval_ratio, max_batches=args.eval_max_batches
            )
            print(
                f"TestSet Eval [{epoch + 1}] Loss: total={eval_loss:.6f}, "
                f"vel={eval_loss_vel:.6f}, pos={eval_loss_pos:.6f}, "
                f"ADE: {eval_ade:.4f} ft ({eval_ade * 0.3048:.4f} m), "
                f"FDE: {eval_fde:.4f} ft ({eval_fde * 0.3048:.4f} m)"
            )

            with csv_path.open("a", newline="", encoding="utf-8") as f:
                writer_csv = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "train_loss_total",
                        "train_loss_vel",
                        "train_loss_pos",
                        "eval_loss_total",
                        "eval_loss_vel",
                        "eval_loss_pos",
                        "eval_ade_ft",
                        "eval_fde_ft",
                        "eval_ade_m",
                        "eval_fde_m",
                        "lr",
                    ],
                )
                writer_csv.writerow(
                    {
                        "epoch": epoch + 1,
                        "train_loss_total": f"{avg_loss:.8f}",
                        "train_loss_vel": f"{avg_loss_vel:.8f}",
                        "train_loss_pos": f"{avg_loss_pos:.8f}",
                        "eval_loss_total": f"{eval_loss:.8f}",
                        "eval_loss_vel": f"{eval_loss_vel:.8f}",
                        "eval_loss_pos": f"{eval_loss_pos:.8f}",
                        "eval_ade_ft": f"{eval_ade:.8f}",
                        "eval_fde_ft": f"{eval_fde:.8f}",
                        "eval_ade_m": f"{eval_ade * 0.3048:.8f}",
                        "eval_fde_m": f"{eval_fde * 0.3048:.8f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.10f}",
                    }
                )

        scheduler.step()

        if dist.is_initialized():
            dist.barrier(device_ids=[local_rank])

        if rank == 0:
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            state = {
                'epoch': epoch + 1,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'train_loss_vel': avg_loss_vel,
                'train_loss_pos': avg_loss_pos,
                'eval_loss': eval_loss,
                'eval_loss_vel': eval_loss_vel,
                'eval_loss_pos': eval_loss_pos,
                'eval_ade': eval_ade,
                'eval_fde': eval_fde,
                'best_ade': best_ade,
            }

            if (epoch + 1) % args.save_interval == 0:
                save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(state, save_path)
                print(f"Saved checkpoint to {save_path}")

            if eval_ade < best_ade:
                best_ade = eval_ade
                state['best_ade'] = best_ade
                save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
                torch.save(state, save_path)
                print(f"Saved best model (ADE: {best_ade:.4f} ft) to {save_path}")

    cleanup_ddp()


if __name__ == '__main__':
    main()
