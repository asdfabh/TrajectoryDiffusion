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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIFIED_TEXT_LOG_DIR = PROJECT_ROOT / "checkpoints" / "log"


def ensure_epoch_text_log(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        return
    header = (
        "epoch,train_loss,train_vel,train_vel_x,train_vel_y,train_pos,train_pos_x,train_pos_y,"
        "train_end,train_intent,train_end_over_pos,train_intent_over_vel,"
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
    end_over_pos = train_stats["loss_end"] / max(train_stats["loss_pos"], 1e-8)
    intent_over_vel = train_stats["loss_intent"] / max(train_stats["loss_vel"], 1e-8)
    line = (
        f"{epoch},"
        f"{train_stats['loss']:.6f},{train_stats['loss_vel']:.6f},{train_stats['loss_vel_x']:.6f},"
        f"{train_stats['loss_vel_y']:.6f},{train_stats['loss_pos']:.6f},{train_stats['loss_pos_x']:.6f},"
        f"{train_stats['loss_pos_y']:.6f},{train_stats['loss_end']:.6f},{train_stats['loss_intent']:.6f},"
        f"{end_over_pos:.6f},{intent_over_vel:.6f},"
        f"{eval_ratio:.2f},{eval_loss:.6f},{eval_ade:.6f},{eval_fde:.6f},"
        f"{(eval_ade * 0.3048):.6f},{(eval_fde * 0.3048):.6f},{lr:.10f}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


TRAIN_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)
EVAL_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:6}| {n_fmt}/{total_fmt} "
    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
)
TIME_STEP_LABELS = [("1s", 4), ("2s", 9), ("3s", 14), ("4s", 19), ("5s", 24)]


class QuickMetrics:
    """训练内轻量评估统计器：单样本采样 + TAME-style 分时段指标。"""

    def __init__(self, pred_len, meter_per_unit=0.3048):
        self.pred_len = int(pred_len)
        self.meter_per_unit = float(meter_per_unit)
        self.total_se = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_de = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_counts = torch.zeros(self.pred_len, dtype=torch.float64)
        self.total_dist_sum = 0.0
        self.total_valid_points = 0.0
        self.total_fde_sum = 0.0
        self.total_fde_count = 0.0

    def update(self, pred, target, op_mask):
        pred = pred[:, :self.pred_len, :2]
        target = target[:, :self.pred_len, :2]
        valid_mask = op_mask[:, :self.pred_len, 0] if op_mask.dim() == 3 else op_mask[:, :self.pred_len]
        valid_mask = (valid_mask > 0).float().to(pred.device)

        diff = pred - target
        se = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(se)

        self.total_se += torch.sum(se * valid_mask, dim=0).double().cpu()
        self.total_de += torch.sum(dist * valid_mask, dim=0).double().cpu()
        self.total_counts += torch.sum(valid_mask, dim=0).double().cpu()

        self.total_dist_sum += float(torch.sum(dist * valid_mask).item())
        self.total_valid_points += float(torch.sum(valid_mask).item())

        t_idx = torch.arange(dist.size(1), device=dist.device).unsqueeze(0).expand_as(dist)
        masked_idx = torch.where(valid_mask > 0, t_idx, t_idx.new_full(t_idx.shape, -1))
        last_idx = masked_idx.max(dim=1).values
        has_valid = last_idx >= 0
        final_dist = dist.gather(1, last_idx.clamp(min=0).unsqueeze(1)).squeeze(1)
        self.total_fde_sum += float(torch.sum(final_dist * has_valid.float()).item())
        self.total_fde_count += float(torch.sum(has_valid.float()).item())

    def summary(self):
        counts = self.total_counts.clamp(min=1.0)
        rmse_per_step_ft = torch.sqrt(self.total_se / counts)
        fde_per_step_ft = self.total_de / counts

        cumsum_de = torch.cumsum(self.total_de, dim=0)
        cumsum_counts = torch.cumsum(self.total_counts, dim=0).clamp(min=1.0)
        ade_prefix_ft = cumsum_de / cumsum_counts
        if ade_prefix_ft.numel() > 1:
            ade_prefix_ft[1:] = cumsum_de[:-1] / cumsum_counts[:-1]

        overall_ade_ft = 0.0 if self.total_valid_points == 0 else self.total_dist_sum / self.total_valid_points
        overall_fde_ft = 0.0 if self.total_fde_count == 0 else self.total_fde_sum / self.total_fde_count

        return {
            "rmse_per_step_ft": rmse_per_step_ft,
            "rmse_per_step_m": rmse_per_step_ft * self.meter_per_unit,
            "fde_per_step_ft": fde_per_step_ft,
            "fde_per_step_m": fde_per_step_ft * self.meter_per_unit,
            "ade_prefix_ft": ade_prefix_ft,
            "ade_prefix_m": ade_prefix_ft * self.meter_per_unit,
            "overall_ade_ft": overall_ade_ft,
            "overall_fde_ft": overall_fde_ft,
            "overall_ade_m": overall_ade_ft * self.meter_per_unit,
            "overall_fde_m": overall_fde_ft * self.meter_per_unit,
        }


def format_timestep_metrics(metric_tensor, meter_per_unit=0.3048):
    values = []
    for label, t_idx in TIME_STEP_LABELS:
        if t_idx < metric_tensor.numel():
            val_ft = float(metric_tensor[t_idx].item())
            values.append(f"{label}: {val_ft:.3f} ft ({val_ft * meter_per_unit:.3f} m)")
    return " | ".join(values) if values else "no valid timestep"


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
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    hist_nbrs = batch["nbrs"]
    mask = batch["mask"]
    temporal_mask = batch["temporal_mask"]

    if int(feature_dim) != 4:
        raise ValueError("train_fut unified future branch currently requires feature_dim=4.")
    # unified state 在模型内部重建为 [rel_x, rel_y, delta_x, delta_y]，
    # 这里仅传原始位置轨迹，避免 feature_dim=4 的双重语义混淆。
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
    total_end_loss = 0.0
    total_intent_loss = 0.0
    num_batches = 0
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        dynamic_ncols=True,
        bar_format=TRAIN_BAR_FORMAT,
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
        total_end_loss += float(loss_parts["loss_end"].item())
        total_intent_loss += float(loss_parts["loss_intent"].item())
        num_batches += 1
        avg_loss = total_loss / num_batches
        avg_vel = total_vel_loss / num_batches
        avg_pos = total_pos_loss / num_batches
        avg_end = total_end_loss / num_batches
        avg_intent = total_intent_loss / num_batches
        pbar.set_postfix_str(
            f"loss={loss.item():.6f}({avg_loss:.6f}) | "
            f"v={avg_vel:.6f} p={avg_pos:.6f} e={avg_end:.6f} i={avg_intent:.6f}"
        )

    denom = max(num_batches, 1)
    return {
        "loss": total_loss / denom,
        "loss_vel": total_vel_loss / denom,
        "loss_vel_x": total_vel_x_loss / denom,
        "loss_vel_y": total_vel_y_loss / denom,
        "loss_pos": total_pos_loss / denom,
        "loss_pos_x": total_pos_x_loss / denom,
        "loss_pos_y": total_pos_y_loss / denom,
        "loss_end": total_end_loss / denom,
        "loss_intent": total_intent_loss / denom,
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
    quick_metrics = QuickMetrics(pred_len=int(getattr(model, "T", 25)))

    total_batches = len(dataloader)
    if total_batches == 0:
        model.train()
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

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_input_data(batch, feature_dim, device=device)
        eval_loss, pred_fut, eval_ade, eval_fde = model.forwardEval(
            hist, hist_nbrs, mask, temporal_mask, fut, op_mask, device
        )

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        quick_metrics.update(pred_fut, fut, op_mask)
        num_batches += 1
        pbar.set_postfix_str(
            f"loss/loss_avg={eval_loss.item():.6f}/{(total_loss / num_batches):.6f} | "
            f"ade/fde={(total_ade / num_batches):.6f}/{(total_fde / num_batches):.6f}"
        )

    model.train()
    if num_batches == 0:
        return 0.0, 0.0, 0.0, quick_metrics.summary()
    return total_loss / num_batches, total_ade / num_batches, total_fde / num_batches, quick_metrics.summary()


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
    if int(args.feature_dim) != 4:
        raise ValueError("train_fut currently supports feature_dim=4 only.")
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / "fut")
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.checkpoint_dir) / "logs"
    writer = SummaryWriter(log_dir=str(log_dir))
    epoch_text_log_path = UNIFIED_TEXT_LOG_DIR / "train_fut_epoch_metrics.txt"
    ensure_epoch_text_log(epoch_text_log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_ratio = 0.03

    print(
        f"[FutModel] Inference sampler: steps={args.num_inference_steps}, "
        f"spacing={args.inference_timestep_spacing}, eta={args.ddim_eta}, x0_clip={args.x0_clip}"
    )
    print(
        f"[FutModel] Train strategy: self_condition_prob={args.self_condition_prob}, "
        f"loss=vel_huber+integrated_pos_huber(no_time_decay,no_xy_weight), "
        f"huber_delta={args.fut_huber_delta}"
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
        train_peak_mem_mb = None
        eval_ema_peak_mem_mb = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        # 训练阶段：传入 ema 参与权重影子步进
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, ema)
        if torch.cuda.is_available():
            train_peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        avg_loss = float(train_stats["loss"])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        # 评估阶段：为了不破坏正在训练的原始权重，先深拷贝备份，再套用 EMA 权重
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.apply_shadow(model)

        eval_loss, eval_ade, eval_fde, quick_eval = evaluate_on_testset(
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
        if torch.cuda.is_available():
            eval_ema_peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        end_over_pos = train_stats["loss_end"] / max(train_stats["loss_pos"], 1e-8)
        intent_over_vel = train_stats["loss_intent"] / max(train_stats["loss_vel"], 1e-8)

        print(f"Epoch [{epoch + 1}] Train Loss: {avg_loss:.6f}")
        print(
            f"Train Detail [{epoch + 1}] Vel: {train_stats['loss_vel']:.6f}, "
            f"VelXY: {train_stats['loss_vel_x']:.6f}/{train_stats['loss_vel_y']:.6f}, "
            f"Pos: {train_stats['loss_pos']:.6f}, "
            f"PosXY: {train_stats['loss_pos_x']:.6f}/{train_stats['loss_pos_y']:.6f}, "
            f"End: {train_stats['loss_end']:.6f}, Intent: {train_stats['loss_intent']:.6f}, "
            f"End/Pos: {end_over_pos:.4f}, Intent/Vel: {intent_over_vel:.4f}"
        )
        print(
            f"TestSet Eval@0.03 [{epoch + 1}] Loss: {eval_loss:.6f}, "
            f"ADE: {eval_ade:.6f} ft ({eval_ade * 0.3048:.6f} m), "
            f"FDE: {eval_fde:.6f} ft ({eval_fde * 0.3048:.6f} m)"
        )
        print(
            f"[QuickEval TAME-style][Ep{epoch + 1}] RMSE: "
            f"{format_timestep_metrics(quick_eval['rmse_per_step_ft'])}"
        )
        print(
            f"[QuickEval TAME-style][Ep{epoch + 1}] FDE: "
            f"{format_timestep_metrics(quick_eval['fde_per_step_ft'])}"
        )
        print(
            f"[QuickEval TAME-style][Ep{epoch + 1}] ADE: "
            f"{format_timestep_metrics(quick_eval['ade_prefix_ft'])}"
        )
        if train_peak_mem_mb is not None and eval_ema_peak_mem_mb is not None:
            print(
                f"[GPU PeakMem][Ep{epoch + 1}] train_step={train_peak_mem_mb:.2f} MB, "
                f"eval_ema={eval_ema_peak_mem_mb:.2f} MB"
            )

        current_lr = optimizer.param_groups[0]["lr"]
        append_epoch_text_log(
            log_path=epoch_text_log_path,
            epoch=epoch + 1,
            train_stats=train_stats,
            eval_ratio=eval_ratio,
            eval_loss=eval_loss,
            eval_ade=eval_ade,
            eval_fde=eval_fde,
            lr=current_lr,
        )

        writer.add_scalar("Train/Loss", avg_loss, epoch + 1)
        writer.add_scalar("Train/Loss_vel", train_stats["loss_vel"], epoch + 1)
        writer.add_scalar("Train/Loss_vel_x", train_stats["loss_vel_x"], epoch + 1)
        writer.add_scalar("Train/Loss_vel_y", train_stats["loss_vel_y"], epoch + 1)
        writer.add_scalar("Train/Loss_pos", train_stats["loss_pos"], epoch + 1)
        writer.add_scalar("Train/Loss_pos_x", train_stats["loss_pos_x"], epoch + 1)
        writer.add_scalar("Train/Loss_pos_y", train_stats["loss_pos_y"], epoch + 1)
        writer.add_scalar("Train/Loss_end", train_stats["loss_end"], epoch + 1)
        writer.add_scalar("Train/Loss_intent", train_stats["loss_intent"], epoch + 1)
        writer.add_scalar("Train/Loss_end_over_pos", end_over_pos, epoch + 1)
        writer.add_scalar("Train/Loss_intent_over_vel", intent_over_vel, epoch + 1)
        writer.add_scalar("Eval/Loss", eval_loss, epoch + 1)
        writer.add_scalar("Eval/ADE_ft", eval_ade, epoch + 1)
        writer.add_scalar("Eval/FDE_ft", eval_fde, epoch + 1)
        writer.add_scalar("Eval/ADE_m", eval_ade * 0.3048, epoch + 1)
        writer.add_scalar("Eval/FDE_m", eval_fde * 0.3048, epoch + 1)
        for label, idx in TIME_STEP_LABELS:
            if idx < quick_eval["rmse_per_step_ft"].numel():
                writer.add_scalar(f"EvalQuick/RMSE_ft_{label}", quick_eval["rmse_per_step_ft"][idx].item(), epoch + 1)
                writer.add_scalar(f"EvalQuick/FDE_ft_{label}", quick_eval["fde_per_step_ft"][idx].item(), epoch + 1)
                writer.add_scalar(f"EvalQuick/ADE_ft_{label}", quick_eval["ade_prefix_ft"][idx].item(), epoch + 1)
        if train_peak_mem_mb is not None and eval_ema_peak_mem_mb is not None:
            writer.add_scalar("Resource/PeakMemTrain_MB", train_peak_mem_mb, epoch + 1)
            writer.add_scalar("Resource/PeakMemEvalEMA_MB", eval_ema_peak_mem_mb, epoch + 1)
        writer.add_scalar("Train/LR", current_lr, epoch + 1)

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
