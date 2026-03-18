import sys
import os
import math
import copy
import inspect
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut


def ensure_epoch_text_log(log_path: Path):
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
    line = (
        f"{epoch},"
        f"{train_stats['loss_last']:.6f},{train_stats['loss_avg']:.6f},"
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
        overall_rmse_ft = 0.0 if self.total_valid_points == 0 else float(torch.sqrt(self.total_se.sum() / self.total_counts.sum().clamp(min=1.0)).item())

        return {
            "rmse_per_step_ft": rmse_per_step_ft,
            "rmse_per_step_m": rmse_per_step_ft * self.meter_per_unit,
            "fde_per_step_ft": fde_per_step_ft,
            "fde_per_step_m": fde_per_step_ft * self.meter_per_unit,
            "ade_prefix_ft": ade_prefix_ft,
            "ade_prefix_m": ade_prefix_ft * self.meter_per_unit,
            "overall_ade_ft": overall_ade_ft,
            "overall_fde_ft": overall_fde_ft,
            "overall_rmse_ft": overall_rmse_ft,
            "overall_ade_m": overall_ade_ft * self.meter_per_unit,
            "overall_fde_m": overall_fde_ft * self.meter_per_unit,
            "overall_rmse_m": overall_rmse_ft * self.meter_per_unit,
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
        self._shadow_on_model = False
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

    def swap_shadow(self, target_model):
        # 通过参数引用交换进行 EMA 切换，避免 clone 整个 state_dict 造成显存峰值。
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


def prepare_input_data(batch, feature_dim, device="cuda"):
    hist = batch["hist"]
    va = batch["va"]
    fut = batch["fut"]
    op_mask = batch["op_mask"]
    hist_nbrs = batch["nbrs"]
    va_nbrs = batch["nbrs_va"]
    mask = batch["mask"]
    temporal_mask = batch["temporal_mask"]

    if int(feature_dim) != 4:
        raise ValueError("train_fut currently supports feature_dim=4: [rel_x, rel_y, v, a].")
    # future 分支输入恢复为 [rel_x, rel_y, v, a]
    hist = torch.cat((hist, va), dim=-1).to(device)
    hist_nbrs = torch.cat((hist_nbrs, va_nbrs), dim=-1).to(device)

    fut = fut.to(device)
    op_mask = op_mask.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    extras = {
        "ego_lane": batch["lane"].to(device),
        "nbr_lane": batch["nbrs_lane"].to(device),
        "nbr_dist": batch["nbrs_distance"].to(device),
        "lat_gt": batch["lat_enc"].argmax(dim=-1).long().to(device),
        "lon_gt": batch["lon_enc"].argmax(dim=-1).long().to(device),
    }
    return hist, hist_nbrs, mask, temporal_mask, fut, op_mask, extras


def build_ngsim_dataset(mat_path, args):
    # 兼容不同 NgsimDataset 版本，按签名过滤可用参数，避免 unexpected keyword 异常。
    dataset_kwargs = {
        "t_h": 30,
        "t_f": 50,
        "d_s": 2,
        "enc_size": args.encoder_input_dim,
        "feature_dim": args.feature_dim,
    }
    sig = inspect.signature(NgsimDataset.__init__)
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kw:
        return NgsimDataset(mat_path, **dataset_kwargs)
    filtered = {k: v for k, v in dataset_kwargs.items() if k in sig.parameters}
    return NgsimDataset(mat_path, **filtered)


def compute_intent_class_weights(dataset):
    lat_idx = torch.as_tensor(dataset.D[:, 9].astype(int) - 1, dtype=torch.long)
    lon_idx = torch.as_tensor(dataset.D[:, 10].astype(int) - 1, dtype=torch.long)

    lat_count = torch.bincount(lat_idx.clamp(min=0, max=2), minlength=3).float()
    lon_count = torch.bincount(lon_idx.clamp(min=0, max=2), minlength=3).float()

    lat_weight = 1.0 / torch.sqrt(lat_count + 1e-6)
    lon_weight = 1.0 / torch.sqrt(lon_count + 1e-6)
    lat_weight = lat_weight / lat_weight.mean().clamp(min=1e-6)
    lon_weight = lon_weight / lon_weight.mean().clamp(min=1e-6)
    return lat_weight, lon_weight, lat_count, lon_count


# 训练函数新增 ema 参数，在反向传播后进行滑动平均更新
def train_epoch(model, dataloader, optimizer, device, epoch, feature_dim, ema):
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
    pbar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Ep{epoch} Train",
        dynamic_ncols=True,
        bar_format=TRAIN_BAR_FORMAT,
    )

    for batch in pbar:
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask, extras = prepare_input_data(batch, feature_dim, device=device)
        loss, loss_parts = model.forwardTrain(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            extras,
            device,
            epoch=epoch,
            return_components=True,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 每次真实权重梯度更新后，让影子模型平滑跟进
        if ema is not None:
            ema.step(model)

        total_loss += float(loss.item())
        total_vel += float(loss_parts["loss_vel"].item())
        total_pos += float(loss_parts["loss_pos"].item())
        total_lat += float(loss_parts["loss_lat"].item())
        total_lon += float(loss_parts["loss_lon"].item())
        total_acc_lat += float(loss_parts["acc_lat"].item())
        total_acc_lon += float(loss_parts["acc_lon"].item())
        last_loss = float(loss.item())
        num_batches += 1
        avg_loss = total_loss / num_batches
        avg_vel = total_vel / num_batches
        avg_pos = total_pos / num_batches
        avg_lat = total_lat / num_batches
        avg_lon = total_lon / num_batches
        avg_acc_lat = total_acc_lat / num_batches
        avg_acc_lon = total_acc_lon / num_batches
        pbar.set_postfix_str(
            f"loss={last_loss:.6f}(avg={avg_loss:.6f}) | "
            f"vel={avg_vel:.6f} | "
            f"pos={avg_pos:.6f} | "
            f"lat={avg_lat:.6f}/{avg_acc_lat:.4f} | "
            f"lon={avg_lon:.6f}/{avg_acc_lon:.4f}"
        )

    denom = max(num_batches, 1)
    return {
        "loss_avg": total_loss / denom,
        "loss_last": last_loss,
        "loss_vel_avg": total_vel / denom,
        "loss_pos_avg": total_pos / denom,
        "loss_lat_avg": total_lat / denom,
        "loss_lon_avg": total_lon / denom,
        "acc_lat_avg": total_acc_lat / denom,
        "acc_lon_avg": total_acc_lon / denom,
    }


@torch.no_grad()
def evaluate_on_testset(model, dataloader, device, epoch, feature_dim, eval_ratio=0.1, max_batches=0, num_samples=5):
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

        hist, hist_nbrs, mask, temporal_mask, fut, op_mask, extras = prepare_input_data(batch, feature_dim, device=device)
        eval_loss, pred_fut, eval_ade, eval_fde = model.forwardEval_minADE(
            hist,
            hist_nbrs,
            mask,
            temporal_mask,
            fut,
            op_mask,
            extras,
            device,
            K=max(1, int(num_samples)),
        )

        total_loss += float(eval_loss.item())
        total_ade += float(eval_ade.item())
        total_fde += float(eval_fde.item())
        quick_metrics.update(pred_fut, fut, op_mask)
        num_batches += 1
        pbar.set_postfix_str(
            f"loss={eval_loss.item():.6f}(avg={(total_loss / num_batches):.6f}) | "
            f"ade={(total_ade / num_batches):.6f} | fde={(total_fde / num_batches):.6f}"
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
            ema._shadow_on_model = False

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
    script_dir = Path(__file__).resolve().parent
    checkpoint_root = Path(args.checkpoint_dir)
    if not checkpoint_root.is_absolute():
        checkpoint_root = script_dir / checkpoint_root
    fut_ckpt_dir = checkpoint_root / "fut"
    fut_ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir = str(fut_ckpt_dir)

    epoch_text_log_path = build_epoch_text_log_path(fut_ckpt_dir, "train_fut_epoch_metrics")
    ensure_epoch_text_log(epoch_text_log_path)
    print(f"[Log] Epoch metrics file: {epoch_text_log_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_ratio = float(args.eval_ratio)

    print(f"[FutModel] self_condition_prob={args.self_condition_prob}, eval_ratio={eval_ratio}")

    data_root = Path(args.data_root)
    train_path = str(data_root / "TrainSet.mat")
    test_path = str(data_root / "TestSet.mat")
    val_path = data_root / "ValSet.mat"
    eval_path = val_path if val_path.exists() else Path(test_path)
    eval_split_name = "ValSet" if val_path.exists() else "TestSet"

    train_dataset = build_ngsim_dataset(train_path, args)
    eval_dataset = build_ngsim_dataset(str(eval_path), args)

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
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_dataset.collate_fn,
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
    lat_weight, lon_weight, lat_count, lon_count = compute_intent_class_weights(train_dataset)
    model.set_intent_class_weights(lat_weight, lon_weight)
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
        train_stats = train_epoch(model, train_loader, optimizer, device, epoch + 1, args.feature_dim, ema)
        avg_loss = float(train_stats["loss_avg"])

        ema.swap_shadow(model)
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
        ema.swap_shadow(model)
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
            "eval_rmse": eval_rmse,
            "best_ade": best_ade,
        }

        if (epoch + 1) % args.save_interval == 0:
            torch.save(state, Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch + 1}.pth")
        if eval_ade < best_ade:
            best_ade = eval_ade
            state["best_ade"] = best_ade
            torch.save(state, Path(args.checkpoint_dir) / "checkpoint_best.pth")

    # 训练结束后输出一次最终评估汇总
    if last_eval is None:
        ema.swap_shadow(model)
        last_eval = evaluate_on_testset(
            model,
            eval_loader,
            device,
            args.num_epochs,
            args.feature_dim,
            eval_ratio=eval_ratio,
            max_batches=args.eval_max_batches,
            num_samples=args.num_samples,
        )
        ema.swap_shadow(model)

    final_loss, final_ade, final_fde, final_quick = last_eval
    print(f"\n========== Final Eval (EMA, minADE@K={max(1, int(args.num_samples))}, {eval_split_name}@{eval_ratio:.2f}) ==========")
    print(
        f"Avg ADE: {final_ade:.6f} ft ({final_ade * 0.3048:.6f} m) | "
        f"Avg FDE: {final_fde:.6f} ft ({final_fde * 0.3048:.6f} m) | "
        f"Avg RMSE: {final_quick['overall_rmse_ft']:.6f} ft ({final_quick['overall_rmse_m']:.6f} m)"
    )
    print(f"RMSE per-second: {format_timestep_metrics(final_quick['rmse_per_step_ft'])}")
    print(f"FDE per-second:  {format_timestep_metrics(final_quick['fde_per_step_ft'])}")
    print(f"ADE per-second:  {format_timestep_metrics(final_quick['ade_prefix_ft'])}")


if __name__ == "__main__":
    main()
