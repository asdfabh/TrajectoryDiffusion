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
    else:
        print("Not using distributed mode")
        return 0, 0, 1


def cleanup_ddp():
    dist.destroy_process_group()


def reduce_value(value, average=True):
    if not dist.is_initialized(): return value
    world_size = dist.get_world_size()
    if world_size < 2: return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average: value /= world_size
    return value


def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    hist = batch['hist']
    va = batch['va']
    lane = batch['lane']
    cclass = batch['cclass']
    fut = batch['fut']
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

    if mask_type == 'random':
        hist_mask = random_mask(hist, p=mask_prob).to(device)
    elif mask_type == 'block':
        hist_mask = continuous_mask(hist, p=mask_prob).to(device)
    else:
        hist_mask = random_mask(hist, p=mask_prob).to(device)

    hist_masked_val = hist_mask * hist
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1)

    hist_masked = hist_masked.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)
    return hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask


def train_epoch(model_fut, model_hist, dataloader, optimizer, device, epoch, feature_dim, rank,
                mask_type='random', mask_prob=0.4, freeze_hist=True):
    model_fut.train()
    if freeze_hist:
        model_hist.eval()
    else:
        model_hist.train()

    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", dynamic_ncols=True)
    else:
        pbar = enumerate(dataloader)

    for batch_idx, batch in pbar:
        hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
            batch, feature_dim, mask_type=mask_type, mask_prob=mask_prob, device=device
        )

        if freeze_hist:
            with torch.no_grad():
                m_hist = model_hist.module if isinstance(model_hist, DDP) else model_hist
                loss_h, pred_hist_clean, _, _ = m_hist.forward_eval(hist, hist_masked, device)
                loss_h = 0.0
        else:
            loss_h, pred_hist_clean, _, _ = model_hist(hist, hist_masked, device)

        loss_f, pred, ade, fde = model_fut(pred_hist_clean, hist_nbrs, mask, temporal_mask, fut, device)

        loss = loss_f + loss_h

        optimizer.zero_grad()
        loss.backward()

        params = list(model_fut.parameters())
        if not freeze_hist: params += list(model_hist.parameters())
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            curr_loss = torch.tensor(loss.item(), device=device)
            dist.all_reduce(curr_loss)
            total_loss += curr_loss.item() / dist.get_world_size()

        num_batches += 1

        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.8f}',
                'avg_loss': f'{total_loss / num_batches:.8f}',
                'ade': f'{ade.mean().item():.4f}',
                'fde': f'{fde.mean().item():.4f}',
            })

    return total_loss / num_batches


def load_checkpoint_ddp(resume_arg, default_dir, model, device, rank, model_name="Model"):
    ckpt_path = None
    default_dir = Path(default_dir)

    if resume_arg == 'latest':
        ckpts = sorted(default_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts: ckpt_path = ckpts[-1]
    elif resume_arg == 'best':
        best = default_dir / 'checkpoint_best.pth'
        if best.exists(): ckpt_path = best
    elif resume_arg.startswith('epoch'):
        try:
            ep = int(resume_arg.replace('epoch', ''))
            ckpt_path = default_dir / f'checkpoint_epoch_{ep}.pth'
        except:
            pass
    elif resume_arg not in ('none', '', None):
        ckpt_path = Path(resume_arg)

    start_epoch = 0
    best_loss = float('inf')

    if ckpt_path and ckpt_path.exists():
        if rank == 0: print(f"Loading {model_name} from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)

        state_dict = state['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict, strict=False)
        start_epoch = state.get('epoch', 0)
        best_loss = state.get('best_loss', best_loss)
    else:
        if rank == 0: print(f"Initialized {model_name} randomly (No checkpoint found at {ckpt_path})")

    return start_epoch, best_loss


def main():
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    if rank != 0:
        def print_pass(*args, **kwargs): pass

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

    data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
    train_path = str(data_root / 'TrainSet.mat')
    train_dataset = NgsimDataset(train_path, t_h=30, t_f=50, d_s=2)

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

    freeze_hist = True
    print(f"Training Mode: Joint Training | Freeze Hist: {freeze_hist}")

    if freeze_hist:
        model_hist.eval()
        for param in model_hist.parameters():
            param.requires_grad = False
    else:
        model_hist = DDP(model_hist, device_ids=[local_rank], output_device=local_rank)

    print("\n[Fut Model] Initializing...")
    model_fut = DiffusionFut(args).to(device)
    start_epoch, best_loss = load_checkpoint_ddp(args.resume_fut, fut_ckpt_dir, model_fut, device, rank, "Fut")

    model_fut = DDP(model_fut, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    params = list(model_fut.parameters())
    if not freeze_hist: params += list(model_hist.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        if rank == 0: print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        avg_loss = train_epoch(
            model_fut, model_hist, train_loader, optimizer, device, epoch + 1,
            args.feature_dim, rank, freeze_hist=freeze_hist
        )

        if rank == 0:
            print(f"Epoch [{epoch + 1}] Average Loss: {avg_loss:.4f}")

            state_fut = {
                'epoch': epoch + 1,
                'model_state_dict': model_fut.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }
            if (epoch + 1) % args.save_interval == 0:
                torch.save(state_fut, fut_ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth")
                if not freeze_hist:
                    hist_state = model_hist.module.state_dict() if isinstance(model_hist,
                                                                              DDP) else model_hist.state_dict()
                    torch.save({'model_state_dict': hist_state},
                               hist_ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth")

            if avg_loss < best_loss:
                best_loss = avg_loss
                state_fut['best_loss'] = best_loss
                torch.save(state_fut, fut_ckpt_dir / "checkpoint_best.pth")
                if not freeze_hist:
                    hist_state = model_hist.module.state_dict() if isinstance(model_hist,
                                                                              DDP) else model_hist.state_dict()
                    torch.save({'model_state_dict': hist_state}, hist_ckpt_dir / "joint_checkpoint_best.pth")

        scheduler.step()

    cleanup_ddp()


if __name__ == '__main__':
    main()