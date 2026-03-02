import sys
import os
import torch
import math
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask, continuous_mask
from method_diffusion.utils.visualization import visualize_batch_trajectories

def prepare_input_data(batch, feature_dim, mask_type='random', mask_prob=0.4, device='cuda'):
    """数据准备函数，与训练代码保持一致"""
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
    hist_masked = torch.cat([hist_masked_val, hist_mask], dim=-1)

    hist_masked = hist_masked.to(device)
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, hist_masked, hist_mask, fut, op_mask, hist_nbrs, mask, temporal_mask


class MetricsCalculator:
    """辅助类：用于累积和计算评估指标 (ADE, FDE, RMSE at timesteps)"""

    def __init__(self, t_max, device, meter_per_unit=0.3048):
        self.t_max = t_max
        self.device = device
        self.meter_per_unit = meter_per_unit

        self.total_se = torch.zeros(t_max).to(device)
        self.total_de = torch.zeros(t_max).to(device)
        self.total_counts = torch.zeros(t_max).to(device)
        self.total_dist_sum = 0.0
        self.total_valid_points = 0.0
        self.total_fde_sum = 0.0
        self.total_fde_count = 0.0

    @staticmethod
    def _normalize_valid_mask(valid_mask, pred):
        if valid_mask is None:
            return torch.ones(pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask[..., 0]
        return (valid_mask > 0.5).to(pred.device).float()

    def update(self, pred, target, valid_mask=None):
        pred = pred[:, :self.t_max, :2]
        target = target[:, :self.t_max, :2]
        _, T, _ = pred.shape
        valid_mask = self._normalize_valid_mask(valid_mask, pred)[:, :T]

        diff = pred - target
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq)
        dist_sq = dist_sq * valid_mask
        dist = dist * valid_mask

        self.total_se[:T] += torch.sum(dist_sq, dim=0)
        self.total_de[:T] += torch.sum(dist, dim=0)
        self.total_counts[:T] += torch.sum(valid_mask, dim=0)

        self.total_dist_sum += torch.sum(dist).item()
        self.total_valid_points += torch.sum(valid_mask).item()

        valid_counts = torch.sum(valid_mask, dim=1).long()
        has_valid = valid_counts > 0
        last_idx = torch.clamp(valid_counts - 1, min=0)
        final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)

        self.total_fde_sum += torch.sum(final_dist * has_valid.float()).item()
        self.total_fde_count += torch.sum(has_valid.float()).item()

    def get_summary(self):
        counts = self.total_counts.clamp(min=1)
        rmse_per_step_ft = torch.sqrt(self.total_se / counts)
        de_avg_per_step_ft = self.total_de / counts

        overall_ade_ft = 0.0 if self.total_valid_points == 0 else self.total_dist_sum / self.total_valid_points
        overall_fde_ft = 0.0 if self.total_fde_count == 0 else self.total_fde_sum / self.total_fde_count

        return {
            'rmse_per_step_ft': rmse_per_step_ft,
            'rmse_per_step_m': rmse_per_step_ft * self.meter_per_unit,
            'de_per_step_ft': de_avg_per_step_ft,
            'de_per_step_m': de_avg_per_step_ft * self.meter_per_unit,
            'overall_ade_ft': overall_ade_ft,
            'overall_ade_m': overall_ade_ft * self.meter_per_unit,
            'overall_fde_ft': overall_fde_ft,
            'overall_fde_m': overall_fde_ft * self.meter_per_unit,
            'num_valid_points': self.total_valid_points,
            'num_valid_final_points': self.total_fde_count,
        }


def load_checkpoint(model, resume_arg, default_dir, device, model_name="Model"):
    """
    鲁棒的模型加载函数
    :param model: 模型实例
    :param resume_arg: 参数传入的字符串 (none, best, latest, epochX, or path)
    :param default_dir: 默认查找目录 (Path对象)
    :param device: 设备
    :param model_name: 打印日志用的名称
    """
    ckpt_path = None
    default_dir = Path(default_dir)

    if resume_arg in ('none', '', None):
        print(f"[{model_name}] No checkpoint specified (arg='{resume_arg}'). Initializing randomly.")
        return model

    if Path(resume_arg).exists():
        ckpt_path = Path(resume_arg)
    elif (default_dir / resume_arg).exists():
        ckpt_path = default_dir / resume_arg
    elif resume_arg == 'latest':
        ckpts = sorted(default_dir.glob('checkpoint_epoch_*.pth'))
        if ckpts: ckpt_path = ckpts[-1]
    elif resume_arg == 'best':
        best_cand = default_dir / 'checkpoint_best.pth'
        if best_cand.exists(): ckpt_path = best_cand
    elif resume_arg.startswith('epoch'):
        try:
            ep = int(resume_arg.replace('epoch', ''))
            ckpt_path = default_dir / f'checkpoint_epoch_{ep}.pth'
        except:
            pass

    if ckpt_path and ckpt_path.exists():
        print(f"[{model_name}] Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[{model_name}] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[{model_name}] Unexpected keys: {len(unexpected)}")

    else:
        print(f"[{model_name}] [Warning] Checkpoint '{resume_arg}' NOT FOUND in {default_dir}. Model remains random.")

    model.eval()
    return model


def get_test_loader(args):
    """获取测试集 DataLoader"""
    if hasattr(args, 'test_path') and args.test_path:
        test_path = args.test_path
    elif os.path.exists(os.path.join(args.data_root, 'TestSet.mat')):
        test_path = os.path.join(args.data_root, 'TestSet.mat')
    else:
        data_root = Path(args.data_root)
        test_path = str(data_root / 'TestSet.mat')

    print(f"Loading test data from: {test_path}")

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
        persistent_workers=args.num_workers > 0,
    )
    return test_loader


def resolve_checkpoint_base_dir(checkpoint_dir):
    ckpt_path = Path(checkpoint_dir)
    if ckpt_path.is_absolute():
        return ckpt_path
    return (Path.cwd() / ckpt_path).resolve()


def print_metrics_table(metrics, name="Model", time_indices=[4, 9, 14, 19, 24],
                        time_labels=['1s', '2s', '3s', '4s', '5s']):
    print(f'\n{"=" * 30} Test Results: {name} {"=" * 30}')
    rmse_m = metrics['rmse_per_step_m']
    rmse_ft = metrics['rmse_per_step_ft']
    de_m = metrics['de_per_step_m']
    de_ft = metrics['de_per_step_ft']

    print(f'Overall ADE: {metrics["overall_ade_m"]:.4f} m | {metrics["overall_ade_ft"]:.4f} ft')
    print(f'Overall FDE: {metrics["overall_fde_m"]:.4f} m | {metrics["overall_fde_ft"]:.4f} ft')
    print('-' * 74)

    valid_steps = [(time_labels[i], t) for i, t in enumerate(time_indices) if t < len(rmse_m)]

    # 1. RMSE at specific timesteps
    print('RMSE at specific timesteps:')
    rmse_str = " | ".join([f"{lbl}: {rmse_m[t].item():.2f} m / {rmse_ft[t].item():.2f} ft" for lbl, t in valid_steps])
    print(rmse_str if rmse_str else "No valid timestep.")

    # 2. Displacement Error at specific timestep
    print('Displacement Error at specific timesteps:')
    de_str = " | ".join([f"{lbl}: {de_m[t].item():.2f} m / {de_ft[t].item():.2f} ft" for lbl, t in valid_steps])
    print(de_str if de_str else "No valid timestep.")
    print('=' * 80)


def compute_batch_metrics(pred, target, op_mask, meter_per_unit=0.3048):
    pred = pred[..., :2]
    target = target[..., :2]
    valid_mask = op_mask[..., 0] if op_mask.dim() == 3 else op_mask
    valid_mask = (valid_mask > 0.5).float().to(pred.device)

    diff = pred - target
    dist_sq = torch.sum(diff ** 2, dim=-1)
    dist = torch.sqrt(dist_sq)

    ade_ft = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)
    rmse_ft = torch.sqrt((dist_sq * valid_mask).sum() / (valid_mask.sum() + 1e-6))

    valid_counts = torch.sum(valid_mask, dim=1).long()
    has_valid = valid_counts > 0
    last_idx = torch.clamp(valid_counts - 1, min=0)
    final_dist = dist.gather(1, last_idx.unsqueeze(1)).squeeze(1)
    fde_ft = (final_dist * has_valid.float()).sum() / (has_valid.float().sum() + 1e-6)

    return {
        "ADE (batch, residual-anchor)": {"m": (ade_ft * meter_per_unit).item(), "ft": ade_ft.item()},
        "FDE (batch, residual-anchor)": {"m": (fde_ft * meter_per_unit).item(), "ft": fde_ft.item()},
        "RMSE (batch, residual-anchor)": {"m": (rmse_ft * meter_per_unit).item(), "ft": rmse_ft.item()},
    }


def compute_single_vis_metrics(pred, target, op_mask, batch_idx=0, meter_per_unit=0.3048):
    """
    针对单条可视化轨迹计算 ADE/FDE，并输出 GT/Pred End Point（坐标+index）。
    """
    b = max(0, min(int(batch_idx), pred.size(0) - 1))
    pred_xy = pred[b, :, :2]
    target_xy = target[b, :, :2]

    valid_mask = op_mask[b]
    if valid_mask.dim() == 2:
        valid_mask = valid_mask[:, 0]
    valid_mask = (valid_mask > 0.5).float().to(pred_xy.device)

    dist = torch.norm(pred_xy - target_xy, dim=-1)
    ade_ft = (dist * valid_mask).sum() / (valid_mask.sum() + 1e-6)

    valid_ids = torch.where(valid_mask > 0.5)[0]
    last_idx = int(valid_ids[-1].item()) if valid_ids.numel() > 0 else int(pred_xy.shape[0] - 1)
    fde_ft = dist[last_idx]

    gt_end = target_xy[last_idx]
    pred_end = pred_xy[last_idx]

    return {
        "ade_ft": float(ade_ft.item()),
        "fde_ft": float(fde_ft.item()),
        "ade_m": float(ade_ft.item() * meter_per_unit),
        "fde_m": float(fde_ft.item() * meter_per_unit),
        "end_index": last_idx,
        "gt_end_ft": (float(gt_end[1].item()), float(gt_end[0].item())),
        "pred_end_ft": (float(pred_end[1].item()), float(pred_end[0].item())),
        "gt_end_m": (float(gt_end[1].item() * meter_per_unit), float(gt_end[0].item() * meter_per_unit)),
        "pred_end_m": (float(pred_end[1].item() * meter_per_unit), float(pred_end[0].item() * meter_per_unit)),
    }


def run_evaluation(args, device):
    # 强制固定测试时的噪声种子！
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    test_loader = get_test_loader(args)
    total_test_batches = len(test_loader)
    test_ratio = max(0.0, min(1.0, float(args.test_ratio)))
    target_test_batches = max(1, int(math.ceil(total_test_batches * test_ratio))) if total_test_batches > 0 else 0
    print(f"[Eval] Test ratio: {test_ratio:.2f}, evaluating {target_test_batches}/{total_test_batches} batches")

    model_hist = None
    base_ckpt_dir = resolve_checkpoint_base_dir(args.checkpoint_dir)

    hist_ckpt_dir = base_ckpt_dir / 'hist'
    fut_ckpt_dir = base_ckpt_dir / 'fut'

    print("\n[Init] Initializing Fut Model...")
    model_fut = DiffusionFut(args).to(device)
    num_params = sum(p.numel() for p in model_fut.parameters())
    print(f"[FutModel] Parameters: {num_params / 1e6:.3f} M")
    print(f"[FutModel] Architecture: hidden_dim_fut={args.hidden_dim_fut}, depth_fut={args.depth_fut}")
    load_checkpoint(model_fut, args.resume_fut, fut_ckpt_dir, device, model_name="FutModel")
    if hasattr(model_fut, "is_main_process"):
        model_fut.is_main_process = False
    print("[FutModel] Eval formulation: residual_anchor_rollout")
    print(
        f"[FutModel] Inference sampler: steps={args.num_inference_steps}, "
        f"spacing={args.inference_timestep_spacing}, eta={args.ddim_eta}, x0_clip={args.x0_clip}, mode=residual_anchor_rollout"
    )

    if args.eval_mode == 'joint':
        print("\n[Init] Initializing Hist Model for Joint Evaluation...")
        model_hist = DiffusionPast(args).to(device)
        load_checkpoint(model_hist, args.resume_hist, hist_ckpt_dir, device, model_name="HistModel")

    calc_fut = MetricsCalculator(args.T_f, device)
    calc_hist = MetricsCalculator(args.T, device) if args.eval_mode == 'joint' else None
    visualize_dir = None
    if args.visualize_samples > 0:
        if args.visualize_dir:
            visualize_dir = Path(args.visualize_dir)
        elif not args.show_plots:
            visualize_dir = base_ckpt_dir / 'visualizations'
    if visualize_dir is not None:
        visualize_dir.mkdir(parents=True, exist_ok=True)
    visualized_count = 0

    with torch.no_grad():
        pbar = tqdm(
            enumerate(test_loader),
            total=target_test_batches,
            desc=f"Testing ({args.eval_mode})",
            ncols=120
        )

        for batch_idx, batch in pbar:
            if batch_idx >= target_test_batches:
                break
            hist, hist_masked, hist_mask, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
                batch, args.feature_dim, mask_type='random', mask_prob=args.mask_prob, device=device
            )

            current_hist_input = hist
            if model_hist is not None:
                _, pred_hist, _, _ = model_hist.forward_eval(hist, hist_masked, device)
                calc_hist.update(pred_hist[..., :2], hist[..., :2], valid_mask=torch.ones_like(hist[..., 0]))
                current_hist_input = pred_hist

            k_samples = max(1, int(args.num_samples))
            _, pred_fut, _, _ = model_fut.forwardEval_minADE(
                current_hist_input, hist_nbrs, mask, temporal_mask, fut, op_mask, device, K=k_samples
            )
            calc_fut.update(pred_fut, fut, valid_mask=op_mask)

            if visualized_count < args.visualize_samples:
                pred_all_vis = getattr(model_fut, "last_minade_all_preds", None)
                pred_best_idx_vis = getattr(model_fut, "last_minade_best_idx", None)
                batch_metrics = compute_batch_metrics(pred_fut, fut, op_mask)
                vis_metrics = compute_single_vis_metrics(pred_fut, fut, op_mask, batch_idx=0)
                running_summary = calc_fut.get_summary()
                running_metrics = {
                    "ADE (running, residual-anchor)": {"m": running_summary["overall_ade_m"], "ft": running_summary["overall_ade_ft"]},
                    "FDE (running, residual-anchor)": {"m": running_summary["overall_fde_m"], "ft": running_summary["overall_fde_ft"]},
                }
                traj_metrics = {
                    "ADE (vis traj, residual-anchor)": {"m": vis_metrics["ade_m"], "ft": vis_metrics["ade_ft"]},
                    "FDE (vis traj, residual-anchor)": {"m": vis_metrics["fde_m"], "ft": vis_metrics["fde_ft"]},
                }
                metrics_for_plot = {**traj_metrics, **batch_metrics, **running_metrics}
                save_path = None
                if visualize_dir is not None:
                    save_path = visualize_dir / f"eval_{args.eval_mode}_batch{batch_idx}_sample0.png"

                vis_info = visualize_batch_trajectories(
                    hist=current_hist_input,
                    hist_nbrs=hist_nbrs,
                    temporal_mask=temporal_mask,
                    future=fut,
                    pred=pred_fut,
                    pred_all=pred_all_vis,
                    pred_best_idx=pred_best_idx_vis,
                    hist_masked=hist_mask,
                    batch_idx=0,
                    save_path=str(save_path) if save_path else None,
                    metrics=metrics_for_plot,
                    input_unit='ft',
                    show_plot=args.show_plots
                )
                fut_end = vis_info.get("fut_last") if isinstance(vis_info, dict) else None
                pred_end = vis_info.get("pred_last") if isinstance(vis_info, dict) else None

                print(
                    f"[Vis][ResidualAnchor][batch={batch_idx}, sample=0] "
                    f"ADE={vis_metrics['ade_ft']:.4f} ft ({vis_metrics['ade_m']:.4f} m), "
                    f"FDE={vis_metrics['fde_ft']:.4f} ft ({vis_metrics['fde_m']:.4f} m)"
                )
                if fut_end and pred_end:
                    print(
                        f"[Vis][ResidualAnchor][EndPoint idx={vis_metrics['end_index']}] "
                        f"GT(ft)={fut_end['coord']} | Pred(ft)={pred_end['coord']} | "
                        f"GT(m)=({vis_metrics['gt_end_m'][0]:.3f}, {vis_metrics['gt_end_m'][1]:.3f}) | "
                        f"Pred(m)=({vis_metrics['pred_end_m'][0]:.3f}, {vis_metrics['pred_end_m'][1]:.3f})"
                    )
                visualized_count += 1

    if args.eval_mode == 'joint' and calc_hist:
        hist_metrics = calc_hist.get_summary()
        print_metrics_table(hist_metrics, name="History Reconstruction",
                            time_indices=[4, 9, 14], time_labels=['1s', '2s', '3s'])

    fut_metrics = calc_fut.get_summary()
    print_metrics_table(fut_metrics, name="Future Prediction (Residual Anchor Rollout)")


def main():
    parser = get_args_parser()
    parser.add_argument('--eval_mode', type=str, default='fut_only', choices=['fut_only', 'joint'],
                        help="评估模式: 'fut_only' (使用GT历史) 或 'joint' (使用Hist模型输出)")
    parser.add_argument('--test_path', type=str, default=None, help="测试集路径 (可选，覆盖默认)")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="测试集评估比例，0~1，默认0.1表示评估10% TestSet")
    parser.add_argument('--visualize_samples', type=int, default=0, help="可视化样本数，0表示不绘制")
    parser.add_argument('--visualize_dir', type=str, default=None, help="可视化图片保存目录")
    parser.add_argument('--show_plots', action='store_true', help="是否弹窗显示可视化")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluation Mode: {args.eval_mode}")

    run_evaluation(args, device)


if __name__ == '__main__':
    main()
