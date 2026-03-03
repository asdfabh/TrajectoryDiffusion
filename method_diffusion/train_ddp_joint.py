import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import builtins
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm

from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return rank, local_rank, world_size
    print("Not using distributed mode")
    return 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_ddp(model):
    return model.module if isinstance(model, DDP) else model


def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    hist = batch['hist']
    va = batch['va']
    lane = batch['lane']
    cclass = batch['cclass']
    fut = batch['fut']
    op_mask = batch['op_mask']
    hist_nbrs = batch['nbrs']
    va_nbrs = batch['nbrs_va']
    lane_nbrs = batch['nbrs_lane']
    cclass_nbrs = batch['nbrs_class']
    mask = batch['mask']
    temporal_mask = batch['temporal_mask']

    if feature_dim == 6:
        hist = torch.cat((hist, va, lane, cclass), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs, cclass_nbrs), dim=-1).to(device)
    elif feature_dim == 5:
        hist = torch.cat((hist, va, lane), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs, lane_nbrs), dim=-1).to(device)
    elif feature_dim == 4:
        hist = torch.cat((hist, va), dim=-1).to(device)
        hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)
    else:
        hist = hist.to(device)
        hist_nbrs = hist_nbrs.to(device)

    fut = fut.to(device)
    op_mask = op_mask.to(device)

    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob).to(device)
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob).to(device)
    else:
        hist_mask = random_mask(hist, p=mask_prob).to(device)

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1).to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)
    return hist, hist_masked, fut, op_mask, hist_nbrs, mask, temporal_mask


def build_hist_outputs(model_hist, hist, hist_masked, device, freeze_hist, detach_hist_for_fut):
    hist_raw = unwrap_ddp(model_hist)

    if freeze_hist:
        with torch.no_grad():
            _, pred_hist_eval, _, _ = hist_raw.forward_eval(hist, hist_masked, device)
        loss_h = torch.zeros((), device=device)
        return loss_h, pred_hist_eval

    # Hist 主干训练信号
    loss_h, pred_hist_train, _, _ = model_hist(hist, hist_masked, device)
    if not detach_hist_for_fut:
        return loss_h, pred_hist_train

    # Fut 条件默认切换为 eval 输出并断开 Fut->Hist 梯度。
    with torch.no_grad():
        _, pred_hist_eval, _, _ = hist_raw.forward_eval(hist, hist_masked, device)
    return loss_h, pred_hist_eval.detach()


def train_epoch(
    model_fut,
    model_hist,
    dataloader,
    optimizer,
    device,
    epoch,
    feature_dim,
    rank,
    mask_type='random',
    mask_prob=0.4,
    freeze_hist=True,
    hist_loss_weight=0.0,
    detach_hist_for_fut=True,
):
    model_fut.train()
    if freeze_hist:
        model_hist.eval()
    else:
        model_hist.train()

    total_loss = 0.0
    total_loss_hist = 0.0
    total_loss_hist_weighted = 0.0
    total_loss_fut = 0.0
    total_loss_fut_vel = 0.0
    total_loss_fut_pos = 0.0
    total_loss_fut_pos_x = 0.0
    total_loss_fut_pos_y = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", ncols=280)
    else:
        pbar = enumerate(dataloader)

    for _, batch in pbar:
        hist, hist_masked, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        loss_h, hist_for_fut = build_hist_outputs(
            model_hist=model_hist,
            hist=hist,
            hist_masked=hist_masked,
            device=device,
            freeze_hist=freeze_hist,
            detach_hist_for_fut=detach_hist_for_fut,
        )

        loss_f, fut_parts = model_fut(
            hist_for_fut, hist_nbrs, mask, temporal_mask, fut, op_mask, device, return_components=True
        )

        loss_h_weighted = hist_loss_weight * loss_h
        loss = loss_f + loss_h_weighted

        optimizer.zero_grad()
        loss.backward()

        params = list(model_fut.parameters())
        if not freeze_hist:
            params += list(model_hist.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_loss_hist += float(loss_h.item())
        total_loss_hist_weighted += float(loss_h_weighted.item())
        total_loss_fut += float(loss_f.item())
        total_loss_fut_vel += float(fut_parts["loss_vel"].item())
        total_loss_fut_pos += float(fut_parts["loss_pos"].item())
        total_loss_fut_pos_x += float(fut_parts["loss_pos_x"].item())
        total_loss_fut_pos_y += float(fut_parts["loss_pos_y"].item())
        num_batches += 1

        if rank == 0:
            pbar.set_postfix({
                'L_all': f'{total_loss / num_batches:.6f}',
                'L_hist': f'{total_loss_hist / num_batches:.6f}',
                'L_hist_w': f'{total_loss_hist_weighted / num_batches:.6f}',
                'L_fut': f'{total_loss_fut / num_batches:.6f}',
                'L_fut_vel': f'{total_loss_fut_vel / num_batches:.6f}',
                'L_fut_pos': f'{total_loss_fut_pos / num_batches:.6f}',
                'L_fut_pos_xy': f'{total_loss_fut_pos_x / num_batches:.6f}/{total_loss_fut_pos_y / num_batches:.6f}',
            })

    stats = torch.tensor([
        total_loss,
        total_loss_hist,
        total_loss_hist_weighted,
        total_loss_fut,
        total_loss_fut_vel,
        total_loss_fut_pos,
        total_loss_fut_pos_x,
        total_loss_fut_pos_y,
        float(num_batches),
    ], device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    global_num_batches = max(float(stats[8].item()), 1.0)
    return {
        "loss_all": float(stats[0].item()) / global_num_batches,
        "loss_hist": float(stats[1].item()) / global_num_batches,
        "loss_hist_weighted": float(stats[2].item()) / global_num_batches,
        "loss_fut": float(stats[3].item()) / global_num_batches,
        "loss_fut_vel": float(stats[4].item()) / global_num_batches,
        "loss_fut_pos": float(stats[5].item()) / global_num_batches,
        "loss_fut_pos_x": float(stats[6].item()) / global_num_batches,
        "loss_fut_pos_y": float(stats[7].item()) / global_num_batches,
    }


def load_checkpoint_ddp(resume_arg, default_dir, model, device, rank, model_name="Model"):
    ckpt_path = None
    default_dir = Path(default_dir)

    if resume_arg == 'latest':
        ckpts = sorted(default_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts:
            ckpt_path = ckpts[-1]
    elif resume_arg == 'best':
        best = default_dir / 'checkpoint_best.pth'
        if best.exists():
            ckpt_path = best
    elif resume_arg.startswith('epoch'):
        try:
            ep = int(resume_arg.replace('epoch', ''))
            ckpt_path = default_dir / f'checkpoint_epoch_{ep}.pth'
        except Exception:
            pass
    elif resume_arg not in ('none', '', None):
        ckpt_path = Path(resume_arg)

    start_epoch = 0
    best_loss = float('inf')

    if ckpt_path and ckpt_path.exists():
        if rank == 0:
            print(f"Loading {model_name} from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        state_dict = state['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)
    else:
        if rank == 0:
            print(f"Initialized {model_name} randomly (No checkpoint found at {ckpt_path})")

    return start_epoch, best_loss


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass

    args = get_args_parser().parse_args()

    current_script_dir = Path(__file__).resolve().parent
    dir_name = Path(args.checkpoint_dir).name
    base_ckpt_dir = current_script_dir / dir_name

    hist_ckpt_dir = base_ckpt_dir / 'hist'
    fut_ckpt_dir = base_ckpt_dir / 'fut'
    if rank == 0:
        print(f"\n[Info] Script Location: {current_script_dir}")
        print(f"[Info] Checkpoint Base Directory forced to: {base_ckpt_dir}")
        hist_ckpt_dir.mkdir(parents=True, exist_ok=True)
        fut_ckpt_dir.mkdir(parents=True, exist_ok=True)

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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    print("\n[Hist Model] Initializing...")
    model_hist = DiffusionPast(args).to(device)
    load_checkpoint_ddp(args.resume_hist, hist_ckpt_dir, model_hist, device, rank, "Hist")

    freeze_hist = int(args.joint_freeze_hist) > 0
    hist_loss_weight = float(args.joint_hist_loss_weight)
    detach_hist_for_fut = int(args.joint_detach_hist_for_fut) > 0
    hist_lr_scale = max(0.0, float(args.joint_hist_lr_scale))

    if freeze_hist:
        model_hist.eval()
        for param in model_hist.parameters():
            param.requires_grad = False
        hist_loss_weight = 0.0
    else:
        model_hist.train()
        for param in model_hist.parameters():
            param.requires_grad = True
        model_hist = DDP(model_hist, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    print("\n[Fut Model] Initializing...")
    model_fut = DiffusionFut(args).to(device)
    start_epoch, best_loss = load_checkpoint_ddp(args.resume_fut, fut_ckpt_dir, model_fut, device, rank, "Fut")
    model_fut = DDP(model_fut, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if rank == 0:
        print(
            "[Joint] mode=train_ddp_joint | "
            f"freeze_hist={freeze_hist}, hist_loss_weight={hist_loss_weight}, "
            f"detach_hist_for_fut={detach_hist_for_fut}, hist_lr_scale={hist_lr_scale}"
        )

    param_groups = [{"params": model_fut.parameters(), "lr": args.learning_rate}]
    if not freeze_hist:
        hist_lr = args.learning_rate * hist_lr_scale
        param_groups.append({"params": model_hist.parameters(), "lr": hist_lr})
        if rank == 0:
            print(f"[Joint] optimizer lr: fut={args.learning_rate:.3e}, hist={hist_lr:.3e}")
    else:
        if rank == 0:
            print(f"[Joint] optimizer lr: fut={args.learning_rate:.3e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0:
            print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        train_stats = train_epoch(
            model_fut=model_fut,
            model_hist=model_hist,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            feature_dim=args.feature_dim,
            rank=rank,
            mask_type='random',
            mask_prob=args.mask_prob,
            freeze_hist=freeze_hist,
            hist_loss_weight=hist_loss_weight,
            detach_hist_for_fut=detach_hist_for_fut,
        )
        avg_loss = train_stats["loss_all"]

        if rank == 0:
            print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.6f}")
            print(
                f"Joint Detail [{epoch + 1}] Hist: {train_stats['loss_hist']:.6f}, "
                f"HistW: {train_stats['loss_hist_weighted']:.6f}, "
                f"Fut: {train_stats['loss_fut']:.6f}"
            )
            print(
                f"Fut Detail [{epoch + 1}] Vel: {train_stats['loss_fut_vel']:.6f}, "
                f"Pos: {train_stats['loss_fut_pos']:.6f}, "
                f"PosXY: {train_stats['loss_fut_pos_x']:.6f}/{train_stats['loss_fut_pos_y']:.6f}"
            )

            state_fut = {
                'epoch': epoch + 1,
                'model_state_dict': model_fut.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'joint_freeze_hist': freeze_hist,
                'joint_hist_loss_weight': hist_loss_weight,
                'joint_detach_hist_for_fut': detach_hist_for_fut,
                'joint_hist_lr_scale': hist_lr_scale,
            }
            if (epoch + 1) % args.save_interval == 0:
                torch.save(state_fut, fut_ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth")
                if not freeze_hist:
                    hist_state = unwrap_ddp(model_hist).state_dict()
                    torch.save(
                        {'model_state_dict': hist_state, 'epoch': epoch + 1, 'loss_hist': train_stats['loss_hist']},
                        hist_ckpt_dir / f"joint_checkpoint_epoch_{epoch + 1}.pth"
                    )

            if avg_loss < best_loss:
                best_loss = avg_loss
                state_fut['best_loss'] = best_loss
                torch.save(state_fut, fut_ckpt_dir / "checkpoint_best.pth")
                if not freeze_hist:
                    hist_state = unwrap_ddp(model_hist).state_dict()
                    torch.save(
                        {'model_state_dict': hist_state, 'loss_hist': train_stats['loss_hist']},
                        hist_ckpt_dir / "joint_checkpoint_best.pth"
                    )

        scheduler.step()

    cleanup_ddp()


if __name__ == '__main__':
    main()
