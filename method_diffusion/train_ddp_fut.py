import sys
import os
import math
import copy
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import builtins
from method_diffusion.config import get_args_parser
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.utils.fut_utils import (
    TrajectoryMetrics,
    build_ngsim_dataset,
    format_timestep_metrics,
    prepare_fut_batch,
)

def ensure_epoch_text_log(log_path: Path):
    """初始化 DDP 训练的 epoch 文本日志。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    header = (
        "epoch,train_loss_cur,train_loss_avg,train_vel_avg,train_pos_avg,train_lat_avg,train_lon_avg,"
        "train_lat_acc,train_lon_acc,"
        "eval_ratio,eval_loss,eval_ade_ft,eval_fde_ft,eval_rmse_ft,"
        "eval_ade_m,eval_fde_m,eval_rmse_m\n"
    )
    log_path.write_text(header, encoding="utf-8")


def append_epoch_text_log(
    log_path: Path,
    epoch: int,
    train_stats: dict,
    eval_ratio: float,
    eval_loss: float,
    eval_ade: float,
    eval_fde: float,
    eval_rmse: float,
):
    """向 DDP epoch 日志中追加一行结果。"""
    line = (
        f"{epoch},{train_stats['loss_last']:.6f},{train_stats['loss_avg']:.6f},"
        f"{train_stats['loss_vel_avg']:.6f},{train_stats['loss_pos_avg']:.6f},"
        f"{train_stats['loss_lat_avg']:.6f},{train_stats['loss_lon_avg']:.6f},"
        f"{train_stats['acc_lat_avg']:.6f},{train_stats['acc_lon_avg']:.6f},"
        f"{eval_ratio:.2f},{eval_loss:.6f},{eval_ade:.6f},{eval_fde:.6f},{eval_rmse:.6f},"
        f"{(eval_ade * 0.3048):.6f},{(eval_fde * 0.3048):.6f},{(eval_rmse * 0.3048):.6f}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)
        # 每个 epoch 结束后立即落盘，避免异常中断丢失最后一轮日志
        f.flush()
        os.fsync(f.fileno())


def build_epoch_text_log_path(checkpoint_dir: Path, prefix: str) -> Path:
    """按时间戳生成新的 DDP 日志文件路径。"""
    timestamp = datetime.now().strftime("%m-%d-%H:%M")
    log_dir = checkpoint_dir / "log"
    log_path = log_dir / f"{prefix}_{timestamp}.txt"
    if not log_path.exists():
        return log_path
    suffix = 1
    while True:
        candidate = log_dir / f"{prefix}_{timestamp}_{suffix}.txt"
        if not candidate.exists():
            return candidate
        suffix += 1


def print_epoch_eval_summary(
    epoch: int,
    train_stats: dict,
    eval_ratio: float,
    eval_loss: float,
    eval_ade: float,
    eval_fde: float,
    eval_rmse: float,
):
    """打印单个 epoch 的 DDP 训练与评估摘要。"""
    print(
        f"[Epoch {epoch}] "
        f"train_loss={train_stats['loss_avg']:.6f} | "
        f"lat={train_stats['loss_lat_avg']:.6f}/{train_stats['acc_lat_avg']:.4f} | "
        f"lon={train_stats['loss_lon_avg']:.6f}/{train_stats['acc_lon_avg']:.4f} | "
        f"eval_ratio={eval_ratio:.2f} | "
        f"eval_loss={eval_loss:.6f} | "
        f"ADE={eval_ade:.6f} ft ({eval_ade * 0.3048:.6f} m) | "
        f"FDE={eval_fde:.6f} ft ({eval_fde * 0.3048:.6f} m) | "
        f"RMSE={eval_rmse:.6f} ft ({eval_rmse * 0.3048:.6f} m)"
    )

TRAIN_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)
EVAL_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)
TIME_STEP_LABELS = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]


class EMAModel:
    """维护 DDP 训练模型的 EMA 影子参数。"""

    def __init__(self, model, decay=0.999):
        """初始化 EMA 状态。"""
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self._shadow_on_model = False
        for param in self.shadow.values():
            if isinstance(param, torch.Tensor):
                param.requires_grad = False

    def step(self, model):
        """在每次优化后更新 EMA 参数。"""
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                    else:
                        self.shadow[name].copy_(param.data)

    def swap_shadow(self, target_model):
        """在真实参数与 EMA 影子参数之间原地交换。"""
        # 通过参数引用交换进行 EMA 切换，避免 clone 整个 state_dict 的峰值开销。
        model_state = target_model.state_dict(keep_vars=True)
        for name, tensor in model_state.items():
            if name not in self.shadow:
                continue
            shadow_tensor = self.shadow[name]
            if tensor.device != shadow_tensor.device or tensor.dtype != shadow_tensor.dtype:
                tmp = tensor.data.detach().clone()
                tensor.data.copy_(shadow_tensor.to(device=tensor.device, dtype=tensor.dtype))
                self.shadow[name] = tmp
            else:
                tmp = tensor.data
                tensor.data = shadow_tensor
                self.shadow[name] = tmp
        self._shadow_on_model = not self._shadow_on_model


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
    """安全释放分布式进程组。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    """返回 DDP 包装后的底层模型。"""
    return model.module if hasattr(model, "module") else model


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

def compute_intent_class_weights(dataset):
    """按训练集标签频率生成横纵向意图类别权重。"""
    lat_idx = torch.as_tensor(dataset.D[:, 9].astype(int) - 1, dtype=torch.long)
    lon_idx = torch.as_tensor(dataset.D[:, 10].astype(int) - 1, dtype=torch.long)

    lat_count = torch.bincount(lat_idx.clamp(min=0, max=2), minlength=3).float()
    lon_count = torch.bincount(lon_idx.clamp(min=0, max=2), minlength=3).float()

    lat_weight = 1.0 / torch.sqrt(lat_count + 1e-6)
    lon_weight = 1.0 / torch.sqrt(lon_count + 1e-6)
    lat_weight = lat_weight / lat_weight.mean().clamp(min=1e-6)
    lon_weight = lon_weight / lon_weight.mean().clamp(min=1e-6)
    return lat_weight, lon_weight, lat_count, lon_count


def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank, ema):
    """执行一个 DDP epoch 的 future 训练。"""
    model.train()
    total_loss = 0.0
    total_vel = 0.0
    total_pos = 0.0
    total_lat = 0.0
    total_lon = 0.0
    total_acc_lat = 0.0
    total_acc_lon = 0.0
    last_loss = 0.0
    num_batches = 0

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Ep{epoch} Train",
            dynamic_ncols=True,
            bar_format=TRAIN_BAR_FORMAT,
        )
    else:
        pbar = dataloader

    for batch in pbar:
        batch_data = prepare_fut_batch(batch, feature_dim, device=device)
        loss, loss_parts = model(
            batch_data["hist"],
            batch_data["hist_nbrs"],
            batch_data["mask"],
            batch_data["temporal_mask"],
            batch_data["fut"],
            batch_data["op_mask"],
            extras=batch_data["extras"],
            device=device,
            epoch=epoch,
            return_components=True,
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.step(unwrap_model(model))

        total_loss += loss.item()
        total_vel += float(loss_parts["loss_vel"].item())
        total_pos += float(loss_parts["loss_pos"].item())
        total_lat += float(loss_parts["loss_lat"].item())
        total_lon += float(loss_parts["loss_lon"].item())
        total_acc_lat += float(loss_parts["acc_lat"].item())
        total_acc_lon += float(loss_parts["acc_lon"].item())
        last_loss = float(loss.item())
        num_batches += 1

        if rank == 0:
            avg_loss = total_loss / num_batches
            avg_vel = total_vel / num_batches
            avg_pos = total_pos / num_batches
            avg_lat = total_lat / num_batches
            avg_lon = total_lon / num_batches
            avg_acc_lat = total_acc_lat / num_batches
            avg_acc_lon = total_acc_lon / num_batches
            pbar.set_postfix_str(
                f"loss={loss.item():.6f}(avg={avg_loss:.6f}) | "
                f"vel={avg_vel:.6f} | "
                f"pos={avg_pos:.6f} | "
                f"lat={avg_lat:.6f}/{avg_acc_lat:.4f} | "
                f"lon={avg_lon:.6f}/{avg_acc_lon:.4f}"
            )

    # Aggregate metrics/count across all ranks for true global averages.
    loss_count = torch.tensor([
        total_loss,
        total_vel,
        total_pos,
        total_lat,
        total_lon,
        total_acc_lat,
        total_acc_lon,
        float(num_batches)
    ], device=device)
    last_loss_tensor = torch.tensor([last_loss], device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(last_loss_tensor, op=dist.ReduceOp.SUM)
        world_size = float(dist.get_world_size())
    else:
        world_size = 1.0

    global_total_batches = max(float(loss_count[7].item()), 1.0)
    return {
        "loss_avg": float(loss_count[0].item()) / global_total_batches,
        "loss_vel_avg": float(loss_count[1].item()) / global_total_batches,
        "loss_pos_avg": float(loss_count[2].item()) / global_total_batches,
        "loss_lat_avg": float(loss_count[3].item()) / global_total_batches,
        "loss_lon_avg": float(loss_count[4].item()) / global_total_batches,
        "acc_lat_avg": float(loss_count[5].item()) / global_total_batches,
        "acc_lon_avg": float(loss_count[6].item()) / global_total_batches,
        "loss_last": float(last_loss_tensor.item()) / world_size,
    }


@torch.no_grad()
def evaluate_on_testset(model, dataloader, device, epoch, feature_dim, eval_ratio=0.1, max_batches=0, num_samples=5):
    """在 rank0 上执行 DDP future 模型的轻量评估。"""
    # 固定验证环节的随机数种子，消除采样方差导致的指标波动
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    fut_model = model.module if hasattr(model, "module") else model
    fut_model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0
    quick_metrics = TrajectoryMetrics(pred_len=int(getattr(fut_model, "T", 25)))

    total_batches = len(dataloader)
    if total_batches == 0:
        fut_model.train()
        return 0.0, 0.0, 0.0, quick_metrics.summary()

    target_batches = total_batches
    if eval_ratio > 0:
        target_batches = max(1, int(math.ceil(total_batches * float(eval_ratio))))
    if max_batches > 0:
        target_batches = min(target_batches, int(max_batches))

    pbar = tqdm(
        enumerate(dataloader),
        total=target_batches,
        desc=f"Ep{epoch} Eval",
        dynamic_ncols=True,
        bar_format=EVAL_BAR_FORMAT,
    )
    for batch_idx, batch in pbar:
        if batch_idx >= target_batches:
            break

        batch_data = prepare_fut_batch(batch, feature_dim, device=device)
        eval_loss, pred_fut, eval_ade, eval_fde = fut_model.forwardEval_minADE(
            batch_data["hist"],
            batch_data["hist_nbrs"],
            batch_data["mask"],
            batch_data["temporal_mask"],
            batch_data["fut"],
            batch_data["op_mask"],
            batch_data["extras"],
            device,
            K=max(1, int(num_samples)),
        )

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        quick_metrics.update(pred_fut, batch_data["fut"], batch_data["op_mask"])
        num_batches += 1
        pbar.set_postfix_str(
            f"loss={eval_loss.item():.6f}(avg={(total_loss / num_batches):.6f}) | "
            f"ade={(total_ade / num_batches):.6f} | fde={(total_fde / num_batches):.6f}"
        )

    fut_model.train()
    if num_batches == 0:
        return 0.0, 0.0, 0.0, quick_metrics.summary()
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches, quick_metrics.summary()

def load_checkpoint_if_needed(args, model, optimizer, scheduler, device, rank, ema=None):
    """按配置恢复 DDP future 训练断点。"""
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
            cand = Path(args.checkpoint_dir) / f'checkpoint_epoch_{epoch_num}.pth'
            if cand.exists():
                ckpt_path = cand
        except ValueError:
            ckpt_path = None
    elif args.resume_fut not in ('none', ''):
        cand = Path(args.resume_fut)
        if cand.exists():
            ckpt_path = cand

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        base_model = unwrap_model(model)

        def strip_module_prefix(state_dict):
            return {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        # 若 checkpoint 同时包含 raw+ema，优先使用 raw 恢复训练状态。
        if 'raw_model_state_dict' in state:
            base_model.load_state_dict(strip_module_prefix(state['raw_model_state_dict']), strict=False)
        else:
            base_model.load_state_dict(strip_module_prefix(state['model_state_dict']), strict=False)

        if ema is not None:
            if 'model_state_dict' in state and 'raw_model_state_dict' in state:
                ema.shadow = copy.deepcopy(strip_module_prefix(state['model_state_dict']))
            else:
                ema.shadow = copy.deepcopy(base_model.state_dict())
            for param in ema.shadow.values():
                if isinstance(param, torch.Tensor):
                    param.requires_grad = False
            ema._shadow_on_model = False

        try:
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])
        except Exception as e:
            if rank == 0:
                print(f"Warning: Could not load optimizer state (expected due to architecture change): {e}")

        start_epoch = int(state.get('epoch', 0))
        best_ade = float(state.get('best_ade', state.get('best_loss', best_ade)))

        if rank == 0:
            print(f"Resumed from {ckpt_path} @ epoch {start_epoch}")

    return start_epoch, best_ade


def main():
    """运行 DDP future 训练入口。"""
    # 1. DDP Setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass

    args = get_args_parser().parse_args()
    if int(args.feature_dim) != 4:
        raise ValueError("train_ddp_fut currently supports feature_dim=4 only.")
    eval_ratio = float(args.eval_ratio)

    script_dir = Path(__file__).resolve().parent
    checkpoint_root = Path(args.checkpoint_dir)
    if not checkpoint_root.is_absolute():
        checkpoint_root = script_dir / checkpoint_root
    fut_ckpt_dir = checkpoint_root / "fut"
    args.checkpoint_dir = str(fut_ckpt_dir)

    if rank == 0:
        fut_ckpt_dir.mkdir(parents=True, exist_ok=True)
        epoch_text_log_path = build_epoch_text_log_path(fut_ckpt_dir, "train_ddp_fut_epoch_metrics")
        ensure_epoch_text_log(epoch_text_log_path)
        print(f"[Log] Epoch metrics file: {epoch_text_log_path}")
        print(
            f"[FutModel-DDP] self_condition_prob={args.self_condition_prob}, "
            f"eval_ratio={eval_ratio}"
        )
    else:
        epoch_text_log_path = None

    # Use args.data_root
    data_root = Path(args.data_root)
    train_path = str(data_root / 'TrainSet.mat')
    test_path = data_root / 'TestSet.mat'
    val_path = data_root / 'ValSet.mat'
    eval_path = val_path if val_path.exists() else test_path
    eval_split_name = "ValSet" if val_path.exists() else "TestSet"

    train_dataset = build_ngsim_dataset(train_path, args)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # 每个 GPU 的 batch size
        shuffle=False,  # Sampler 负责 shuffle
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        sampler=train_sampler,
        drop_last=True
    )

    eval_loader = None
    if rank == 0:
        eval_dataset = build_ngsim_dataset(str(eval_path), args)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=eval_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            drop_last=False
        )

    model = DiffusionFut(args).to(device)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"[FutModel] Parameters: {num_params / 1e6:.3f} M")

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

    ema = EMAModel(unwrap_model(model), decay=0.9999)
    start_epoch, best_ade = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device, rank, ema
    )
    lat_weight, lon_weight, lat_count, lon_count = compute_intent_class_weights(train_dataset)
    unwrap_model(model).set_intent_class_weights(lat_weight, lon_weight)
    if rank == 0:
        print(
            "[Intent] lat_count={} lon_count={} lat_weight={} lon_weight={}".format(
                lat_count.tolist(),
                lon_count.tolist(),
                [round(v, 4) for v in lat_weight.tolist()],
                [round(v, 4) for v in lon_weight.tolist()],
            )
        )
        if eval_split_name == "ValSet":
            print(f"[Eval] 使用 {eval_split_name} 做 epoch 内选模，指标为 forwardEval_minADE(K={max(1, int(args.num_samples))})")
        else:
            print(f"[Eval] 未找到 ValSet.mat，回退到 {eval_split_name} 做评估。")
    last_eval = None

    for epoch in range(start_epoch, args.num_epochs):
        # 重要：设置 epoch 以保证每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        train_stats = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, ema
        )
        avg_loss = float(train_stats["loss_avg"])

        if rank == 0:
            ema.swap_shadow(unwrap_model(model))
            eval_loss, eval_ade, eval_fde, quick_eval = evaluate_on_testset(
                model,
                eval_loader,
                device,
                epoch + 1,
                args.feature_dim,
                eval_ratio=eval_ratio,
                max_batches=args.eval_max_batches,
                num_samples=args.num_samples,
            )
            ema.swap_shadow(unwrap_model(model))
            eval_rmse = float(quick_eval["overall_rmse_ft"])
            last_eval = (eval_loss, eval_ade, eval_fde, quick_eval)
            append_epoch_text_log(
                log_path=epoch_text_log_path,
                epoch=epoch + 1,
                train_stats=train_stats,
                eval_ratio=eval_ratio,
                eval_loss=eval_loss,
                eval_ade=eval_ade,
                eval_fde=eval_fde,
                eval_rmse=eval_rmse,
            )
            print_epoch_eval_summary(
                epoch=epoch + 1,
                train_stats=train_stats,
                eval_ratio=eval_ratio,
                eval_loss=eval_loss,
                eval_ade=eval_ade,
                eval_fde=eval_fde,
                eval_rmse=eval_rmse,
            )

        scheduler.step()

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            base_model = unwrap_model(model)
            state = {
                'epoch': epoch + 1,
                'model_state_dict': ema.shadow,
                'raw_model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'eval_loss': eval_loss,
                'eval_ade': eval_ade,
                'eval_fde': eval_fde,
                'eval_rmse': eval_rmse,
                'best_ade': best_ade,
            }

            if (epoch + 1) % args.save_interval == 0:
                save_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save(state, save_path)
            if eval_ade < best_ade:
                best_ade = eval_ade
                state['best_ade'] = best_ade
                save_path = Path(args.checkpoint_dir) / "checkpoint_best.pth"
                torch.save(state, save_path)

    if rank == 0 and last_eval is not None:
        final_loss, final_ade, final_fde, final_quick = last_eval
        print(f"\n========== Final Eval (EMA, minADE@K={max(1, int(args.num_samples))}, {eval_split_name}@{eval_ratio:.2f}) ==========")
        print(
            f"Avg ADE: {final_ade:.6f} ft ({final_ade * 0.3048:.6f} m) | "
            f"Avg FDE: {final_fde:.6f} ft ({final_fde * 0.3048:.6f} m) | "
            f"Avg RMSE: {final_quick['overall_rmse_ft']:.6f} ft ({final_quick['overall_rmse_m']:.6f} m)"
        )
        print(f"RMSE per-second: {format_timestep_metrics(final_quick['rmse_per_step_ft'], time_step_labels=TIME_STEP_LABELS)}")
        print(f"DE per-second:   {format_timestep_metrics(final_quick['de_per_step_ft'], time_step_labels=TIME_STEP_LABELS)}")
        print(f"ADE per-second:  {format_timestep_metrics(final_quick['ade_prefix_ft'], time_step_labels=TIME_STEP_LABELS)}")
    cleanup_ddp()


if __name__ == '__main__':
    main()
