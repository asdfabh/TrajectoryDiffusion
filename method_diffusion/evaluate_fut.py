import sys
import os
import torch
import csv
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import argparse

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask
from method_diffusion.utils.traj_vis_metrics import visualize_hist_nbrs_fut_pred
from method_diffusion.utils.traj_metrics import (
    TrajectoryMetricsAccumulator,
    compute_batch_metrics,
)

def prepare_input_data(batch, feature_dim, device='cuda'):
    """数据准备函数，与训练代码保持一致"""
    hist = batch['hist']
    va = batch['va']
    lane = batch['lane']
    cclass = batch['cclass']
    fut = batch['fut']
    op_mask = batch['op_mask']
    hist_nbrs = batch['nbrs']  # [B, T, N, 2] or [B, N, T, 2]
    va_nbrs = batch['nbrs_va']  # [B, T, N, 2] or [B, N, T, 2]
    lane_nbrs = batch['nbrs_lane']  # [B, T, N, 1] or [B, N, T, 1]
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
    mask = mask.to(device)
    temporal_mask = temporal_mask.to(device)

    return hist, fut, op_mask, hist_nbrs, mask, temporal_mask


def compute_batch_ade_fde(pred, target, op_mask=None, unit_conversion=0.3048):
    metrics = compute_batch_metrics(
        pred=pred,
        target=target,
        op_mask=op_mask,
        unit_conversion=unit_conversion,
    )
    return float(metrics["ade"].item()), float(metrics["fde"].item())


class MetricsCalculator:
    """统一指标累积器（TAME evaluate 同口径）"""

    def __init__(self, t_max, device, unit_conversion=0.3048):
        self.t_max = t_max
        self.device = device
        self.unit_conversion = unit_conversion
        self.acc = TrajectoryMetricsAccumulator(
            t_max=t_max,
            device=device,
            unit_conversion=unit_conversion,
        )

    def update(self, pred, target, op_mask=None):
        self.acc.update(pred, target, op_mask=op_mask)

    def get_summary(self):
        return self.acc.get_summary()

    def running_ade(self) -> float:
        return self.acc.running_ade()

    def running_fde(self) -> float:
        return self.acc.running_fde()


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
        data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
        test_path = str(data_root / 'TestSet.mat')

    print(f"Loading test data from: {test_path}")

    base_dataset = NgsimDataset(test_path, t_h=30, t_f=50, d_s=2, feature_dim=args.feature_dim)
    eval_dataset = base_dataset
    total_samples = len(base_dataset)

    if args.test_ratio < 1.0:
        subset_size = max(1, int(total_samples * args.test_ratio))
        generator = torch.Generator()
        generator.manual_seed(int(args.test_ratio_seed))
        subset_indices = torch.randperm(total_samples, generator=generator)[:subset_size].tolist()
        eval_dataset = Subset(base_dataset, subset_indices)
        print(
            f"Using test subset: {subset_size}/{total_samples} samples "
            f"({args.test_ratio * 100:.2f}%), seed={args.test_ratio_seed}"
        )
    else:
        print(f"Using full test set: {total_samples} samples (100.00%)")

    test_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=base_dataset.collate_fn,
        pin_memory=True
    )
    return test_loader, total_samples, len(eval_dataset)


def print_metrics_table(metrics, name="Model", time_indices=[4, 9, 14, 19, 24],
                        time_labels=['1s', '2s', '3s', '4s', '5s']):
    print(f'\n{"=" * 30} Test Results: {name} {"=" * 30}')
    rmse = metrics['rmse_per_step']
    de = metrics['de_per_step']

    if 'overall_mse' in metrics:
        print(f'Overall MSE: {metrics["overall_mse"]:.4f} m^2')
    print(f'Overall ADE: {metrics["overall_ade"]:.4f} m')
    print(f'Overall FDE: {metrics["overall_fde"]:.4f} m')
    print('-' * 74)

    # 1. RMSE at specific timesteps
    print('RMSE (m):')
    valid_indices = [t for t in time_indices if t < len(rmse)]
    rmse_str = " | ".join([f"{time_labels[i]}: {rmse[t].item():.2f}" for i, t in enumerate(valid_indices)])
    print(rmse_str)

    # 2. FDE (Displacement Error at specific timestep)
    print('Displacement Error (m) at specific time:')
    de_str = " | ".join([f"{time_labels[i]}: {de[t].item():.2f}" for i, t in enumerate(valid_indices)])
    print(de_str)
    print('=' * 80)


def run_evaluation(args, device):
    model_hist = None
    model_fut = None

    script_dir = Path(__file__).resolve().parent

    arg_ckpt_path = Path(args.checkpoint_dir)
    if arg_ckpt_path.is_absolute():
        base_ckpt_dir = arg_ckpt_path
    else:
        base_ckpt_dir = script_dir / arg_ckpt_path.name

    test_loader, total_samples, selected_samples = get_test_loader(args)

    hist_ckpt_dir = base_ckpt_dir / 'hist'
    fut_ckpt_dir = base_ckpt_dir / 'fut'

    if args.save_batch_vis:
        if args.vis_dir:
            vis_dir = Path(args.vis_dir)
            if not vis_dir.is_absolute():
                vis_dir = script_dir / vis_dir
        else:
            vis_dir = base_ckpt_dir / "eval_batch_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv_path = vis_dir / "batch_metrics.csv"
        print(f"[VIS] Batch visualization dir: {vis_dir}")
    else:
        vis_dir = None
        metrics_csv_path = None

    print(f"[Eval] Sample coverage: {selected_samples}/{total_samples}")

    print("\n[Init] Initializing Fut Model...")
    model_fut = DiffusionFut(args).to(device)
    load_checkpoint(model_fut, args.resume_fut, fut_ckpt_dir, device, model_name="FutModel")

    if args.eval_mode == 'joint':
        print("\n[Init] Initializing Hist Model for Joint Evaluation...")
        model_hist = DiffusionPast(args).to(device)
        load_checkpoint(model_hist, args.resume_hist, hist_ckpt_dir, device, model_name="HistModel")

    calc_fut = MetricsCalculator(args.T_f, device)
    calc_hist = MetricsCalculator(args.T, device) if args.eval_mode == 'joint' else None

    metrics_file = None
    metrics_writer = None
    if metrics_csv_path is not None:
        metrics_file = open(metrics_csv_path, "w", newline="")
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow([
            "batch_idx",
            "batch_size",
            "batch_ade_m",
            "batch_fde_m",
            "running_ade_m",
            "running_fde_m",
            "sample_ade_m",
            "sample_fde_m",
            "image_path",
        ])

    try:
        with torch.no_grad():
            pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing ({args.eval_mode})", ncols=120)

            for batch_idx, batch in pbar:
                hist, fut, op_mask, hist_nbrs, mask, temporal_mask = prepare_input_data(
                    batch, args.feature_dim, device=device
                )

                current_hist_input = hist
                pred_hist_traj = None
                if model_hist is not None:
                    hist_mask = random_mask(hist, p=0.5).to(device)
                    hist_masked = torch.cat([hist * hist_mask, hist_mask], dim=-1)
                    _, pred_hist, _, _ = model_hist.forward_eval(hist, hist_masked, device)
                    calc_hist.update(pred_hist[..., :2], hist[..., :2])

                    current_hist_input = pred_hist
                    pred_hist_traj = pred_hist  # 保存下来用于可视化

                _, pred_fut, _, _ = model_fut.forward_eval(
                    current_hist_input, hist_nbrs, mask, temporal_mask, fut, device, op_mask=op_mask
                )
                calc_fut.update(pred_fut, fut, op_mask=op_mask)

                batch_ade_m, batch_fde_m = compute_batch_ade_fde(
                    pred_fut, fut, op_mask=op_mask, unit_conversion=calc_fut.unit_conversion
                )
                running_ade_m = calc_fut.running_ade()
                running_fde_m = calc_fut.running_fde()

                pbar.set_postfix({
                    "batch_ADE(m)": f"{batch_ade_m:.4f}",
                    "batch_FDE(m)": f"{batch_fde_m:.4f}",
                    "running_ADE(m)": f"{running_ade_m:.4f}",
                    "running_FDE(m)": f"{running_fde_m:.4f}",
                })

                if args.save_batch_vis and vis_dir is not None:
                    vis_sample_idx = int(args.vis_sample_idx)
                    if vis_sample_idx < 0 or vis_sample_idx >= pred_fut.shape[0]:
                        vis_sample_idx = 0

                    vis_title = (
                        f"Batch {batch_idx:05d} | "
                        f"ADE {batch_ade_m:.4f}m | FDE {batch_fde_m:.4f}m | "
                        f"Running ADE {running_ade_m:.4f}m | Running FDE {running_fde_m:.4f}m"
                    )
                    vis_file = vis_dir / (
                        f"batch_{batch_idx:05d}_ade_{batch_ade_m:.4f}_fde_{batch_fde_m:.4f}.png"
                    )
                    sample_metrics, saved_file = visualize_hist_nbrs_fut_pred(
                        hist=pred_hist_traj if pred_hist_traj is not None else current_hist_input,
                        nbrs=hist_nbrs,
                        fut=fut,
                        pred=pred_fut,
                        op_mask=op_mask,
                        sample_index=vis_sample_idx,
                        save_path=str(vis_file),
                        title_prefix=vis_title,
                        temporal_mask=temporal_mask,
                    )

                    if metrics_writer is not None:
                        metrics_writer.writerow([
                            batch_idx,
                            int(pred_fut.shape[0]),
                            f"{batch_ade_m:.8f}",
                            f"{batch_fde_m:.8f}",
                            f"{running_ade_m:.8f}",
                            f"{running_fde_m:.8f}",
                            f"{sample_metrics['ade_m']:.8f}",
                            f"{sample_metrics['fde_m']:.8f}",
                            str(saved_file) if saved_file is not None else str(vis_file),
                        ])
                        metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()

    if args.eval_mode == 'joint' and calc_hist:
        hist_metrics = calc_hist.get_summary()
        print_metrics_table(hist_metrics, name="History Reconstruction",
                            time_indices=[4, 9, 14], time_labels=['1s', '2s', '3s'])

    fut_metrics = calc_fut.get_summary()
    print_metrics_table(fut_metrics, name="Future Prediction")
    if metrics_csv_path is not None:
        print(f"[VIS] Batch metrics saved to: {metrics_csv_path}")


def main():
    parser = get_args_parser()
    parser.add_argument('--eval_mode', type=str, default='fut_only', choices=['fut_only', 'joint'],
                        help="评估模式: 'fut_only' (使用GT历史) 或 'joint' (使用Hist模型输出)")
    parser.add_argument('--test_path', type=str, default=None, help="测试集路径 (可选，覆盖默认)")
    parser.add_argument('--test_ratio', type=float, default=0.01,
                        help="测试集采样比例，范围 (0, 1]，例如 0.1 表示使用 10% 测试样本")
    parser.add_argument('--test_ratio_seed', type=int, default=2026,
                        help="测试集按比例抽样时的随机种子")
    parser.add_argument('--save_batch_vis', action=argparse.BooleanOptionalAction, default=True,
                        help="是否为每个 batch 保存可视化图")
    parser.add_argument('--vis_dir', type=str, default='',
                        help="可视化输出目录；为空时默认写入 checkpoint_dir/eval_batch_vis")
    parser.add_argument('--vis_sample_idx', type=int, default=0,
                        help="每个 batch 可视化的样本索引")

    args = parser.parse_args()

    if args.test_ratio <= 0.0 or args.test_ratio > 1.0:
        raise ValueError(f"--test_ratio must be in (0, 1], got {args.test_ratio}")

    args.batch_size = 512
    args.num_workers = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluation Mode: {args.eval_mode}")
    print(f"Test Ratio: {args.test_ratio * 100:.2f}%")
    print(f"Save Batch Visualization: {args.save_batch_vis}")

    run_evaluation(args, device)


if __name__ == '__main__':
    main()
