import torch

from method_diffusion.utils.mask_util import random_mask


def build_hist_mask(hist, mask_ratio=0.4):
    """为 future / joint 工具构造历史观测掩码。"""
    return random_mask(hist, p=mask_ratio)


def wrap_angle(angle):
    """将角度差包裹到 [-pi, pi]。"""
    return torch.atan2(torch.sin(angle), torch.cos(angle))

def normalize_traj_valid_mask(valid_mask, pred):
    """将不同形状的 future 有效位掩码统一成 `[B, T]` 浮点张量。"""
    if valid_mask is None:
        return torch.ones(pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype)
    if valid_mask.dim() == 3:
        valid_mask = valid_mask[..., 0]
    return (valid_mask > 0.5).to(pred.device).float()


def compute_batch_metric(pred, target, valid_mask=None):
    """计算单个 batch 的 RMSE / ADE / FDE。"""
    pred_xy = pred[..., :2]
    target_xy = target[..., :2]
    valid_mask = normalize_traj_valid_mask(valid_mask, pred_xy)

    diff = pred_xy - target_xy
    dist_sq = torch.sum(diff ** 2, dim=-1)
    dist = torch.norm(diff, dim=-1)
    rmse = torch.sqrt((dist_sq * valid_mask).sum() / (valid_mask.sum() + 1e-6))
    ade = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    valid_counts = valid_mask.sum(dim=1).long()
    has_valid = valid_counts > 0
    last_idx = torch.clamp(valid_counts - 1, min=0)
    final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    fde = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)
    return rmse, ade, fde


def compute_batch_kinematic_metrics(pred, target, valid_mask=None):
    """计算单个 batch 的 theta MAE(度) 与 v MAE(m/s)。"""
    valid_mask = normalize_traj_valid_mask(valid_mask, pred)

    theta_mae_deg = pred.new_tensor(0.0)
    v_mae_mps = pred.new_tensor(0.0)

    if pred.size(-1) >= 3 and target.size(-1) >= 3:
        theta_diff = wrap_angle(pred[..., 2] - target[..., 2]).abs()
        theta_mae_deg = theta_diff.mul(180.0 / torch.pi)
        theta_mae_deg = (theta_mae_deg * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    if pred.size(-1) >= 4 and target.size(-1) >= 4:
        v_diff = (pred[..., 3] - target[..., 3]).abs()
        v_mae_mps = (v_diff * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    return theta_mae_deg, v_mae_mps


def select_closest_prediction(all_preds, target, valid_mask=None):
    """从多模态预测中选择整段 future RMSE 最小的轨迹。"""
    _, dist_sq, _, valid = candidate_xy_errors(all_preds, target, valid_mask)
    mse_k = dist_sq.sum(dim=2) / valid.sum(dim=1).clamp(min=1).unsqueeze(1)
    best_idx = torch.argmin(mse_k, dim=1)

    bsz, _, t_len, feat_dim = all_preds.shape
    gather_idx = best_idx.view(bsz, 1, 1, 1).expand(bsz, 1, t_len, feat_dim)
    best_pred = all_preds.gather(1, gather_idx).squeeze(1)
    return best_pred, best_idx, torch.sqrt(mse_k)


def candidate_xy_errors(pred, target, valid_mask=None):
    """返回所有完整候选的逐点误差；无效真值不参与计算。"""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.dim() != 4 or pred.size(1) == 0:
        raise ValueError("Predictions must have shape [B,K,T,D] with K >= 1.")
    valid = normalize_traj_valid_mask(valid_mask, pred[:, 0]).bool()
    diff = pred[..., :2] - target[:, None, :, :2]
    diff = torch.where(valid[:, None, :, None], diff, 0.0).double()
    dist_sq = diff.square().sum(dim=-1)
    return pred, dist_sq, dist_sq.sqrt(), valid


def sample_xy_errors(pred, target, valid_mask=None):
    """逐样本返回整段最优 RMSE/ADE/FDE；三项分别选择完整候选。"""
    _, dist_sq, dist, valid = candidate_xy_errors(pred, target, valid_mask)
    counts = valid.sum(dim=1)
    has_valid = counts > 0
    denom = counts.clamp(min=1).unsqueeze(1)
    rmse = (dist_sq.sum(dim=2) / denom).sqrt().min(dim=1).values
    ade = (dist.sum(dim=2) / denom).min(dim=1).values
    positions = torch.arange(valid.size(1), device=valid.device)
    last_idx = torch.where(valid, positions, -1).max(dim=1).values.clamp(min=0)
    endpoint = dist.gather(2, last_idx[:, None, None].expand(-1, dist.size(1), 1))
    fde = endpoint.squeeze(2).min(dim=1).values
    return rmse, ade, fde, has_valid


class SampleImprovementStats:
    """累计 refined 相对 baseline 的逐样本 XY 改善统计。"""

    def __init__(self, eps=1e-6):
        self.eps = float(eps)
        self.count = 0
        self.rmse_improved = 0
        self.rmse_worse = 0
        self.ade_improved = 0
        self.ade_worse = 0
        self.fde_improved = 0
        self.fde_worse = 0
        self.total_delta_rmse = 0.0
        self.total_delta_ade = 0.0
        self.total_delta_fde = 0.0

    def update(self, baseline, refined, target, valid_mask=None):
        base_rmse, base_ade, base_fde, has_valid = sample_xy_errors(baseline, target, valid_mask)
        ref_rmse, ref_ade, ref_fde, _ = sample_xy_errors(refined, target, valid_mask)
        if not torch.any(has_valid):
            return

        base_rmse = base_rmse[has_valid]
        base_ade = base_ade[has_valid]
        base_fde = base_fde[has_valid]
        ref_rmse = ref_rmse[has_valid]
        ref_ade = ref_ade[has_valid]
        ref_fde = ref_fde[has_valid]

        delta_rmse = ref_rmse - base_rmse
        delta_ade = ref_ade - base_ade
        delta_fde = ref_fde - base_fde
        self.count += int(has_valid.sum().item())
        self.rmse_improved += int((delta_rmse < -self.eps).sum().item())
        self.rmse_worse += int((delta_rmse > self.eps).sum().item())
        self.ade_improved += int((delta_ade < -self.eps).sum().item())
        self.ade_worse += int((delta_ade > self.eps).sum().item())
        self.fde_improved += int((delta_fde < -self.eps).sum().item())
        self.fde_worse += int((delta_fde > self.eps).sum().item())
        self.total_delta_rmse += float(delta_rmse.sum().item())
        self.total_delta_ade += float(delta_ade.sum().item())
        self.total_delta_fde += float(delta_fde.sum().item())

    def summary(self):
        denom = max(self.count, 1)
        return {
            "count": self.count,
            "rmse_improved_rate": self.rmse_improved / denom,
            "rmse_worse_rate": self.rmse_worse / denom,
            "ade_improved_rate": self.ade_improved / denom,
            "ade_worse_rate": self.ade_worse / denom,
            "fde_improved_rate": self.fde_improved / denom,
            "fde_worse_rate": self.fde_worse / denom,
            "mean_delta_rmse": self.total_delta_rmse / denom,
            "mean_delta_ade": self.total_delta_ade / denom,
            "mean_delta_fde": self.total_delta_fde / denom,
        }


def print_sample_improvement(summary, title):
    print("\n" + "=" * 30 + f" {title} " + "=" * 30)
    print(f"Samples: {summary['count']}")
    print(
        f"RMSE improved/worse: {summary['rmse_improved_rate']:.6f} / "
        f"{summary['rmse_worse_rate']:.6f}, mean_delta={summary['mean_delta_rmse']:.8f} m"
    )
    print(
        f"minADE improved/worse: {summary['ade_improved_rate']:.6f} / "
        f"{summary['ade_worse_rate']:.6f}, mean_delta={summary['mean_delta_ade']:.8f} m"
    )
    print(
        f"minFDE improved/worse: {summary['fde_improved_rate']:.6f} / "
        f"{summary['fde_worse_rate']:.6f}, mean_delta={summary['mean_delta_fde']:.8f} m"
    )
    print("=" * 75)


def build_eval_timestep_pairs(train_timestep_max, num_inference_steps, inference_trunc_timestep):
    """构造显式 DDIM 推理步，例如 30->24->18->12->6->0。"""
    if inference_trunc_timestep % num_inference_steps != 0:
        raise ValueError("Inference trunc timestep must be divisible by num_inference_steps.")

    step_stride = inference_trunc_timestep // num_inference_steps
    current_timesteps = []
    next_timesteps = []
    current_t = int(inference_trunc_timestep)

    for _ in range(int(num_inference_steps)):
        next_t = max(current_t - int(step_stride), 0)
        current_timesteps.append(int(current_t))
        next_timesteps.append(int(next_t))
        current_t = next_t

    if current_timesteps[0] >= train_timestep_max or next_timesteps[-1] != 0:
        raise ValueError("Invalid fut inference timestep schedule.")
    return current_timesteps, next_timesteps


def ddim_step(scheduler, pred_x0, sample, current_timestep, next_timestep):
    """在同一条 alpha_bar 曲线上执行显式 t_cur -> t_next 的 DDIM 确定性更新。"""
    alpha_prod_t = scheduler.alphas_cumprod[current_timestep].to(device=sample.device, dtype=sample.dtype)
    alpha_prod_next = scheduler.alphas_cumprod[next_timestep].to(device=sample.device, dtype=sample.dtype)
    beta_prod_t = (1 - alpha_prod_t).clamp(min=1e-6)
    beta_prod_next = (1 - alpha_prod_next).clamp(min=0.0)
    pred_epsilon = (sample - alpha_prod_t.sqrt() * pred_x0) / beta_prod_t.sqrt()
    return alpha_prod_next.sqrt() * pred_x0 + beta_prod_next.sqrt() * pred_epsilon


class TrajectoryMetrics:
    """累计整段 RMSE（checkpoint 选择）及逐时刻 RMSE/minADE@K/minFDE@K。

    RMSE 沿用整段 RMSE 最优候选的逐时刻误差。
    ADE 在每个 horizon 上先对每条候选计算前缀均值，再选择完整候选。
    FDE 在每个 horizon 上按终点误差选择完整候选。绝不逐点拼接候选。
    同一 horizon 的三项指标均只统计该时刻有有效真值的样本。
    """

    stat_names = (
        "total_coord_se", "total_min_ade", "total_min_fde",
        "total_theta_abs_deg", "total_v_abs", "total_counts",
    )

    def __init__(self, pred_len, num_candidates=1):
        self.pred_len = int(pred_len)
        self.num_candidates = int(num_candidates)
        for name in self.stat_names:
            setattr(self, name, torch.zeros(self.pred_len, dtype=torch.float64))

    @staticmethod
    def normalize_valid_mask(valid_mask, pred):
        return normalize_traj_valid_mask(valid_mask, pred)

    @torch.no_grad()
    def update(self, pred, target, valid_mask=None):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        pred = pred[:, :, :self.pred_len]
        target = target[:, :self.pred_len]
        if valid_mask is not None:
            valid_mask = valid_mask[:, :self.pred_len]
        pred, dist_sq, dist, valid = candidate_xy_errors(pred, target, valid_mask)
        self.num_candidates = pred.size(1)
        t_len = pred.size(2)
        region = slice(0, t_len)

        # 每条样本先选一条整段 RMSE 最优轨迹，逐秒 RMSE 均来自这条轨迹。
        best_rmse_idx = dist_sq.sum(dim=2).argmin(dim=1)
        batch_idx = torch.arange(pred.size(0), device=pred.device)
        best_pred = pred[batch_idx, best_rmse_idx]
        best_dist_sq = dist_sq[batch_idx, best_rmse_idx]
        self.total_coord_se[region] += best_dist_sq.sum(dim=0).cpu()

        # 先对同一条候选的完整前缀求 ADE，再在候选维选择；禁止先逐点取 min。
        prefix_counts = valid.cumsum(dim=1).clamp(min=1)
        ade_candidates = dist.cumsum(dim=2) / prefix_counts[:, None, :]
        best_ade_idx = ade_candidates.argmin(dim=1, keepdim=True)
        best_ade = ade_candidates.gather(1, best_ade_idx).squeeze(1)
        best_fde_idx = dist.argmin(dim=1, keepdim=True)
        best_fde = dist.gather(1, best_fde_idx).squeeze(1)
        self.total_min_ade[region] += torch.where(valid, best_ade, 0.0).sum(dim=0).cpu()
        self.total_min_fde[region] += torch.where(valid, best_fde, 0.0).sum(dim=0).cpu()
        self.total_counts[region] += valid.sum(dim=0).double().cpu()

        # 其他状态指标继续来自 RMSE 最优候选。
        if pred.size(-1) >= 3 and target.size(-1) >= 3:
            theta_diff = wrap_angle(best_pred[..., 2] - target[..., 2]).abs().double()
            self.total_theta_abs_deg[region] += torch.where(valid, theta_diff, 0.0).sum(dim=0).cpu() * (180.0 / torch.pi)
        if pred.size(-1) >= 4 and target.size(-1) >= 4:
            v_diff = (best_pred[..., 3] - target[..., 3]).abs().double()
            self.total_v_abs[region] += torch.where(valid, v_diff, 0.0).sum(dim=0).cpu()

    def summary(self):
        # 没有该 horizon 真值时返回 NaN，避免把未评估的 horizon 显示成零误差。
        counts = self.total_counts
        ade = self.total_min_ade / counts
        fde = self.total_min_fde / counts
        return {
            "rmse_full_m": (self.total_coord_se.sum() / counts.sum()).sqrt(),
            "rmse_per_step_m": (self.total_coord_se / counts).sqrt(),
            "min_ade_per_step_m": ade,
            "min_fde_per_step_m": fde,
            "theta_mae_per_step_deg": self.total_theta_abs_deg / counts,
            "v_mae_per_step_mps": self.total_v_abs / counts,
            "valid_counts": counts.clone(),
            "num_candidates": self.num_candidates,
        }


def reduce_trajectory_metrics(metrics, device):
    """DDP 合并原始累计量，合并后再开方/求均值。"""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        for name in metrics.stat_names:
            value = getattr(metrics, name).to(device)
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
            setattr(metrics, name, value.cpu())
    return metrics


class DistributedEvalSampler(torch.utils.data.Sampler):
    """验证集按 rank 分片，不补齐重复样本。"""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.num_replicas - 1) // self.num_replicas)


def get_horizon_pairs(pred_len, dt=0.2):
    """返回能精确对应采样时间步的逐秒 horizon。"""
    import math
    if dt <= 0:
        raise ValueError("Prediction step duration must be positive.")
    pairs = []
    for second in range(1, int(math.floor(pred_len * dt + 1e-6)) + 1):
        step = int(round(second / dt))
        if step <= pred_len and math.isclose(step * dt, second, abs_tol=1e-6):
            pairs.append((f"{second}s", step - 1))
    return pairs


def horizon_metric_rows(summary, dt=0.2):
    """逐秒三项指标与 AVG；AVG 是所报告逐秒数值的算术平均。"""
    pairs = get_horizon_pairs(len(summary["rmse_per_step_m"]), dt)
    rows = []
    for label, idx in pairs:
        rows.append({
            "horizon": label,
            "rmse_m": float(summary["rmse_per_step_m"][idx]),
            "min_ade_m": float(summary["min_ade_per_step_m"][idx]),
            "min_fde_m": float(summary["min_fde_per_step_m"][idx]),
            "valid_samples": int(summary["valid_counts"][idx]),
        })
    if rows:
        avg = {"horizon": "AVG", "valid_samples": ""}
        for key in ("rmse_m", "min_ade_m", "min_fde_m"):
            values = [row[key] for row in rows if row["valid_samples"] > 0]
            avg[key] = sum(values) / len(values) if values else float("nan")
        rows.append(avg)
    return rows


def print_trajectory_metrics(summary, title, dt=0.2):
    k = summary["num_candidates"]
    print(f"\n{title} | K={k}")
    print(f"{'Horizon':<8} | {'RMSE (m)':<12} | {f'minADE@{k} (m)':<16} | {f'minFDE@{k} (m)':<16} | Samples")
    for row in horizon_metric_rows(summary, dt):
        print(f"{row['horizon']:<8} | {row['rmse_m']:<12.6f} | {row['min_ade_m']:<16.6f} | {row['min_fde_m']:<16.6f} | {row['valid_samples']}")


def write_horizon_metrics(csv_path, epoch, summary, dt=0.2, stage="Eval", writer=None):
    """逐秒指标写入独立 CSV/TensorBoard，整段 RMSE 单独标明选择用途。"""
    import csv
    from pathlib import Path
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = horizon_metric_rows(summary, dt)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    fieldnames = ("epoch", "stage", "k", "horizon", "rmse_m", "min_ade_m", "min_fde_m", "valid_samples")
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({"epoch": epoch, "stage": stage, "k": summary["num_candidates"], **row})
            if writer is not None:
                for key, metric in (("rmse_m", "RMSE"), ("min_ade_m", "minADE"), ("min_fde_m", "minFDE")):
                    writer.add_scalar(f"{stage}/{metric}/{row['horizon']}", row[key], epoch)
    if writer is not None:
        writer.add_scalar(f"{stage}/RMSE_full_selection_m", summary["rmse_full_m"], epoch)


def validation_metric_values(summary, horizon_idx=None):
    """兼容训练摘要的六项标量；第一项为整段 RMSE，其余为指定 horizon。"""
    idx = len(summary["rmse_per_step_m"]) - 1 if horizon_idx is None else int(horizon_idx)
    return (
        float(summary["rmse_full_m"]),
        float(summary["min_ade_per_step_m"][idx]),
        float(summary["min_fde_per_step_m"][idx]),
        float(summary["rmse_per_step_m"][idx]),
        float(summary["theta_mae_per_step_deg"][idx]),
        float(summary["v_mae_per_step_mps"][idx]),
    )
