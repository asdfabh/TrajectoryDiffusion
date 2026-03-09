import sys
import os
import math
import copy
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_TEXT_LOG_DIR = PROJECT_ROOT / "checkpoints" / "log"


def ensure_epoch_text_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    header = (
        "epoch,train_loss,train_vel,train_vel_x,train_vel_y,train_pos,train_pos_x,train_pos_y,"
        "eval_ratio,eval_loss,eval_ade_ft,eval_fde_ft,eval_ade_m,eval_fde_m,lr\n"
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
    lr: float,
):
    line = (
        f"{epoch},"
        f"{train_stats['loss']:.6f},{train_stats['loss_vel']:.6f},{train_stats['loss_vel_x']:.6f},"
        f"{train_stats['loss_vel_y']:.6f},{train_stats['loss_pos']:.6f},{train_stats['loss_pos_x']:.6f},"
        f"{train_stats['loss_pos_y']:.6f},{eval_ratio:.2f},{eval_loss:.6f},{eval_ade:.6f},{eval_fde:.6f},"
        f"{(eval_ade * 0.3048):.6f},{(eval_fde * 0.3048):.6f},{lr:.10f}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


TRAIN_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} {postfix}"
EVAL_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} {postfix}"


class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        for param in self.shadow.values():
            if isinstance(param, torch.Tensor):
                param.requires_grad = False

    def step(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                    else:
                        self.shadow[name].copy_(param.data)

    def apply_shadow(self, target_model):
        target_model.load_state_dict(self.shadow)


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

def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, rank, ema=None):
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_vel_x_loss = 0.0
    total_vel_y_loss = 0.0
    total_pos_loss = 0.0
    total_pos_x_loss = 0.0
    total_pos_y_loss = 0.0
    num_batches = 0

    # 只有 Rank 0 显示进度条
    if rank == 0:
        pbar = tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc=f"Ep{epoch} Train",
            dynamic_ncols=True,
            bar_format=TRAIN_BAR_FORMAT,
        )
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_parts = model(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, return_components=True)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            unwrapped_model = model.module if hasattr(model, "module") else model
            ema.step(unwrapped_model)

        total_loss += loss.item()
        total_vel_loss += float(loss_parts["loss_vel"].item())
        total_vel_x_loss += float(loss_parts["loss_vel_x"].item())
        total_vel_y_loss += float(loss_parts["loss_vel_y"].item())
        total_pos_loss += float(loss_parts["loss_pos"].item())
        total_pos_x_loss += float(loss_parts["loss_pos_x"].item())
        total_pos_y_loss += float(loss_parts["loss_pos_y"].item())
        num_batches += 1

        if rank == 0:
            pbar.set_postfix_str(
                f"loss/loss_avg={loss.item():.6f}/{(total_loss / num_batches):.6f} | "
                f"vel/vel_x/vel_y={(total_vel_loss / num_batches):.6f}/{(total_vel_x_loss / num_batches):.6f}/{(total_vel_y_loss / num_batches):.6f} | "
                f"pos/pos_x/pos_y={(total_pos_loss / num_batches):.6f}/{(total_pos_x_loss / num_batches):.6f}/{(total_pos_y_loss / num_batches):.6f}"
            )

    # Aggregate metrics/count across all ranks for true global averages.
    loss_count = torch.tensor([
        total_loss,
        total_vel_loss,
        total_vel_x_loss,
        total_vel_y_loss,
        total_pos_loss,
        total_pos_x_loss,
        total_pos_y_loss,
        float(num_batches)
    ], device=device)
    if dist.is_initialized():
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
    global_total_batches = max(float(loss_count[7].item()), 1.0)
    return {
        "loss": float(loss_count[0].item()) / global_total_batches,
        "loss_vel": float(loss_count[1].item()) / global_total_batches,
        "loss_vel_x": float(loss_count[2].item()) / global_total_batches,
        "loss_vel_y": float(loss_count[3].item()) / global_total_batches,
        "loss_pos": float(loss_count[4].item()) / global_total_batches,
        "loss_pos_x": float(loss_count[5].item()) / global_total_batches,
        "loss_pos_y": float(loss_count[6].item()) / global_total_batches,
    }


@torch.no_grad()
def evaluate_on_testset(model, dataloader, device, epoch, feature_dim, eval_ratio=0.1, max_batches=0):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    fut_model = model.module if hasattr(model, "module") else model
    fut_model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    total_batches = len(dataloader)
    if total_batches == 0:
        fut_model.train()
        return 0.0, 0.0, 0.0

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

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        eval_loss, _, eval_ade, eval_fde = fut_model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        num_batches += 1
        pbar.set_postfix_str(
            f"loss/loss_avg={eval_loss.item():.6f}/{(total_loss / num_batches):.6f} | "
            f"ade/fde={(total_ade / num_batches):.6f}/{(total_fde / num_batches):.6f}"
        )

    fut_model.train()
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches

def load_checkpoint_if_needed(args, model, optimizer, scheduler, device, ema=None):
    start_epoch = 0
    best_ade = float("inf")
    ckpt_path = None

    if args.resume_fut == "latest":
        ckpts = sorted(Path(args.checkpoint_dir).glob("checkpoint_epoch_*.pth"))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif args.resume_fut == "best":
        best_candidate = Path(args.checkpoint_dir) / "checkpoint_best.pth"
        if best_candidate.exists():
            ckpt_path = best_candidate
    elif args.resume_fut.startswith("epoch"):
        try:
            epoch_num = int(args.resume_fut.replace("epoch", ""))
            cand = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch_num}.pth"
            if cand.exists():
                ckpt_path = cand
        except ValueError:
            ckpt_path = None
    elif args.resume_fut not in ("none", ""):
        cand = Path(args.resume_fut)
        if cand.exists():
            ckpt_path = cand

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)

        if "raw_model_state_dict" in state:
            model.load_state_dict(state["raw_model_state_dict"], strict=False)
        else:
            model.load_state_dict(state["model_state_dict"], strict=False)

        if ema is not None:
            if "model_state_dict" in state and "raw_model_state_dict" in state:
                ema.shadow = copy.deepcopy(state["model_state_dict"])
            else:
                ema.shadow = copy.deepcopy(model.state_dict())

        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception:
            pass

        start_epoch = int(state.get("epoch", 0))
        best_ade = float(state.get("best_ade", state.get("best_loss", best_ade)))
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
    epoch_text_log_path = UNIFIED_TEXT_LOG_DIR / "train_ddp_fut_epoch_metrics.txt"

    # Keep the same checkpoint layout as train_fut.py.
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / 'fut')

    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        ensure_epoch_text_log(epoch_text_log_path)
        fixed_eval_ratio = 0.03
        print(
            f"[FutModel] Inference sampler: steps={args.num_inference_steps}, "
            f"spacing={args.inference_timestep_spacing}, eta={args.ddim_eta}, x0_clip={args.x0_clip}"
        )
        print(
            f"[FutModel] Train strategy: self_condition_prob={args.self_condition_prob}, "
            f"loss=vel_huber+pos_huber_physical_time_discount, y_weight={args.fut_y_loss_weight}, "
            f"huber_delta={args.fut_huber_delta}, pos_weight={args.fut_pos_loss_weight}"
        )
        print(
            f"[FutModel] Architecture: hidden_dim_fut={args.hidden_dim_fut}, depth_fut={args.depth_fut}"
        )
        print(
            f"[FutModel] TestSet eval sampling: eval_ratio={fixed_eval_ratio}, "
            f"eval_max_batches={args.eval_max_batches}"
        )
    else:
        fixed_eval_ratio = 0.03

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
    actual_batch_size = max(1, args.batch_size // world_size)
    if rank == 0:
        print(
            f"[FutModel] BatchSize(per-gpu/global): {actual_batch_size}/{actual_batch_size * world_size} "
            f"(requested global target={args.batch_size})"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
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
            batch_size=actual_batch_size,
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    ema = EMAModel(model, decay=0.9999)
    start_epoch, best_ade = load_checkpoint_if_needed(
        args, model, optimizer, scheduler, device, ema
    )

    # 仅在分布式环境下使用 DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    else:
        print("Running in non-distributed mode (Single GPU/CPU)")

    for epoch in range(start_epoch, args.num_epochs):
        # 重要：设置 epoch 以保证每个 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        train_stats = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, ema=ema
        )
        avg_loss = float(train_stats["loss"])

        if rank == 0:
            unwrapped_model = model.module if hasattr(model, "module") else model
            original_state = {k: v.clone() for k, v in unwrapped_model.state_dict().items()}
            ema.apply_shadow(unwrapped_model)

            print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.6f}")
            print(
                f"Train Detail [{epoch + 1}] Vel: {train_stats['loss_vel']:.6f}, "
                f"VelXY: {train_stats['loss_vel_x']:.6f}/{train_stats['loss_vel_y']:.6f}, "
                f"Pos: {train_stats['loss_pos']:.6f}, "
                f"PosXY: {train_stats['loss_pos_x']:.6f}/{train_stats['loss_pos_y']:.6f}"
            )
            eval_loss, eval_ade, eval_fde = evaluate_on_testset(
                model, test_loader, device, epoch + 1, args.feature_dim,
                eval_ratio=fixed_eval_ratio, max_batches=args.eval_max_batches
            )
            print(
                f"TestSet Eval@0.03 [{epoch + 1}] Loss: {eval_loss:.6f}, "
                f"ADE: {eval_ade:.6f} ft ({eval_ade * 0.3048:.6f} m), "
                f"FDE: {eval_fde:.6f} ft ({eval_fde * 0.3048:.6f} m)"
            )
            unwrapped_model.load_state_dict(original_state)

            current_lr = optimizer.param_groups[0]["lr"]
            append_epoch_text_log(
                log_path=epoch_text_log_path,
                epoch=epoch + 1,
                train_stats=train_stats,
                eval_ratio=fixed_eval_ratio,
                eval_loss=eval_loss,
                eval_ade=eval_ade,
                eval_fde=eval_fde,
                lr=current_lr,
            )

        scheduler.step()

        if dist.is_initialized():
            dist.barrier()

        if rank == 0:
            unwrapped_model = model.module if hasattr(model, "module") else model
            state = {
                'epoch': epoch + 1,
                'model_state_dict': ema.shadow,
                'raw_model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'eval_loss': eval_loss,
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
