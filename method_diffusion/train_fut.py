import sys
import os
import math
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut


# 指数移动平均 (Exponential Moving Average) 包装器类
# 用于平滑扩散模型在训练后期的参数高频震荡
class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        # 影子参数仅做记录，不参与反向传播和梯度计算
        for param in self.shadow.values():
            if isinstance(param, torch.Tensor):
                param.requires_grad = False

    def step(self, model):
        # 每次优化器步进后调用此方法，平滑更新影子权重
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    if param.dtype.is_floating_point:
                        self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
                    else:
                        self.shadow[name].copy_(param.data)

    def apply_shadow(self, target_model):
        # 将当前的影子权重加载到目标模型中，用于评估或推理
        target_model.load_state_dict(self.shadow)


def prepare_input_data(batch, feature_dim, device="cuda"):
    hist = batch["hist"]
    va = batch["va"]
    lane = batch["lane"]
    cclass = batch["cclass"]
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    hist_nbrs = batch["nbrs"]
    va_nbrs = batch["nbrs_va"]
    lane_nbrs = batch["nbrs_lane"]
    cclass_nbrs = batch["nbrs_class"]
    mask = batch["mask"]
    temporal_mask = batch["temporal_mask"]

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
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)
    return hist, hist_nbrs, mask, temporal_mask, fut, op_mask


# 训练函数新增 ema 参数，在反向传播后进行滑动平均更新
def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, ema):
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_vel_x_loss = 0.0
    total_vel_y_loss = 0.0
    total_pos_loss = 0.0
    total_pos_x_loss = 0.0
    total_pos_y_loss = 0.0
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        ncols=220,
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_parts = model.forwardTrain(
            hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, return_components=True
        )
        # _, pred_fut, _, _ = model.forwardEval_minADE(
        #     hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device, K=6
        # )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 每次真实权重梯度更新后，让影子模型平滑跟进
        if ema is not None:
            ema.step(model)

        total_loss += float(loss.item())
        total_vel_loss += float(loss_parts["loss_vel"].item())
        total_vel_x_loss += float(loss_parts["loss_vel_x"].item())
        total_vel_y_loss += float(loss_parts["loss_vel_y"].item())
        total_pos_loss += float(loss_parts["loss_pos"].item())
        total_pos_x_loss += float(loss_parts["loss_pos_x"].item())
        total_pos_y_loss += float(loss_parts["loss_pos_y"].item())
        num_batches += 1
        pbar.set_postfix({
            "loss": f"{loss.item():.6f}",
            "avg_loss": f"{(total_loss / num_batches):.6f}",
            "vel": f"{(total_vel_loss / num_batches):.6f}",
            "vel_xy": f"{(total_vel_x_loss / num_batches):.6f}/{(total_vel_y_loss / num_batches):.6f}",
            "pos": f"{(total_pos_loss / num_batches):.6f}",
            "pos_xy": f"{(total_pos_x_loss / num_batches):.6f}/{(total_pos_y_loss / num_batches):.6f}",
        })

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "loss_vel": total_vel_loss / denom,
        "loss_vel_x": total_vel_x_loss / denom,
        "loss_vel_y": total_vel_y_loss / denom,
        "loss_pos": total_pos_loss / denom,
        "loss_pos_x": total_pos_x_loss / denom,
        "loss_pos_y": total_pos_y_loss / denom,
    }


@torch.no_grad()
def evaluate_on_testset(model, dataloader, device, epoch, feature_dim, eval_ratio=0.1, max_batches=0):
    # 固定验证环节的随机数种子，消除采样方差导致的指标波动
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model.eval()
    total_loss = 0.0
    total_ade = 0.0
    total_fde = 0.0
    num_batches = 0

    total_batches = len(dataloader)
    if total_batches == 0:
        model.train()
        return 0.0, 0.0, 0.0

    target_batches = total_batches
    if eval_ratio > 0:
        target_batches = max(1, int(math.ceil(total_batches * float(eval_ratio))))
    if max_batches > 0:
        target_batches = min(target_batches, int(max_batches))

    pbar = tqdm(enumerate(dataloader), total=target_batches, desc=f"Epoch {epoch} Eval", ncols=120)
    for batch_idx, batch in pbar:
        if batch_idx >= target_batches:
            break

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        eval_loss, _, eval_ade, eval_fde = model.forwardEval(hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device)

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        num_batches += 1
        pbar.set_postfix({
            "avg_ade_ft": f"{(total_ade / num_batches):.4f}",
            "avg_fde_ft": f"{(total_fde / num_batches):.4f}",
        })

    model.train()
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

        # 兼容性设计：如果存在 raw_model_state_dict，说明这是带 EMA 的断点，优先用 raw 恢复训练状态
        if "raw_model_state_dict" in state:
            model.load_state_dict(state["raw_model_state_dict"], strict=False)
        else:
            model.load_state_dict(state["model_state_dict"], strict=False)

        # 同步恢复 EMA 的影子权重
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
    args = get_args_parser().parse_args()
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / "fut")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.checkpoint_dir) / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_ratio = max(0.0, min(1.0, float(args.eval_ratio)))
    if eval_ratio <= 0.0:
        eval_ratio = 0.1

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
        f"[FutModel] TestSet eval sampling: eval_ratio={eval_ratio}, eval_max_batches={args.eval_max_batches}"
    )

    data_root = Path(args.data_root)
    train_path = str(data_root / "TrainSet.mat")
    test_path = str(data_root / "TestSet.mat")

    train_dataset = NgsimDataset(
        train_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    test_dataset = NgsimDataset(
        test_path,
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = DiffusionFut(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # 初始化 EMA 实例，扩散模型黄金衰减率 0.999
    ema = EMAModel(model, decay=0.9999)

    start_epoch, best_ade = load_checkpoint_if_needed(args, model, optimizer, scheduler, device, ema)

    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n========== Epoch {epoch + 1}/{args.num_epochs} ==========")

        # 训练阶段：传入 ema 参与权重影子步进
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, ema)
        avg_loss = float(train_stats["loss"])

        # 评估阶段：为了不破坏正在训练的原始权重，先深拷贝备份，再套用 EMA 权重
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow(model)

        eval_loss, eval_ade, eval_fde = evaluate_on_testset(
            model,
            test_loader,
            device,
            epoch + 1,
            args.feature_dim,
            eval_ratio=eval_ratio,
            max_batches=args.eval_max_batches,
        )

        # 评估结束，将原始训练权重覆盖回来，继续下一轮训练
        model.load_state_dict(original_state)

        print(f"Epoch [{epoch + 1}] Train Loss: {avg_loss:.6f}")
        print(
            f"Train Detail [{epoch + 1}] Vel: {train_stats['loss_vel']:.6f}, "
            f"VelXY: {train_stats['loss_vel_x']:.6f}/{train_stats['loss_vel_y']:.6f}, "
            f"Pos: {train_stats['loss_pos']:.6f}, "
            f"PosXY: {train_stats['loss_pos_x']:.6f}/{train_stats['loss_pos_y']:.6f}"
        )
        print(
            f"TestSet Eval [{epoch + 1}] Loss: {eval_loss:.6f}, "
            f"ADE: {eval_ade:.4f} ft ({eval_ade * 0.3048:.4f} m), "
            f"FDE: {eval_fde:.4f} ft ({eval_fde * 0.3048:.4f} m)"
        )

        writer.add_scalar("Train/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Train/Loss_vel", train_stats["loss_vel"], epoch + 1)
        writer.add_scalar("Train/Loss_vel_x", train_stats["loss_vel_x"], epoch + 1)
        writer.add_scalar("Train/Loss_vel_y", train_stats["loss_vel_y"], epoch + 1)
        writer.add_scalar("Train/Loss_pos", train_stats["loss_pos"], epoch + 1)
        writer.add_scalar("Train/Loss_pos_x", train_stats["loss_pos_x"], epoch + 1)
        writer.add_scalar("Train/Loss_pos_y", train_stats["loss_pos_y"], epoch + 1)
        writer.add_scalar("Eval/Loss", eval_loss, epoch + 1)
        writer.add_scalar("Eval/ADE_ft", eval_ade, epoch + 1)
        writer.add_scalar("Eval/FDE_ft", eval_fde, epoch + 1)
        writer.add_scalar("Eval/ADE_m", eval_ade * 0.3048, epoch + 1)
        writer.add_scalar("Eval/FDE_m", eval_fde * 0.3048, epoch + 1)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch + 1)

        scheduler.step()

        # 模型保存阶段核心逻辑：
        # 将极其平滑的 EMA 权重存入默认的 model_state_dict 键中。
        # 这样当你直接用 evaluate_fut.py 跑测试时，无需改任何代码，加载的就是最佳参数。
        # 同时保存 raw_model_state_dict 用于断点恢复真实的训练状态。
        state = {
            "epoch": epoch + 1,
            "model_state_dict": ema.shadow,
            "raw_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "eval_loss": eval_loss,
            "eval_ade": eval_ade,
            "eval_fde": eval_fde,
            "best_ade": best_ade,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if eval_ade < best_ade:
            best_ade = eval_ade
            state["best_ade"] = best_ade
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

    writer.close()


if __name__ == "__main__":
    main()
