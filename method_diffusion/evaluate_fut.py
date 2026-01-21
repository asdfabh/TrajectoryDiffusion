import sys
import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import argparse

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


class MetricsCalculator:
    """辅助类：用于累积和计算评估指标 (ADE, FDE, RMSE at timesteps)"""

    def __init__(self, t_max, device, unit_conversion=0.3048):
        self.t_max = t_max
        self.device = device
        self.unit_conversion = unit_conversion

        self.total_se = torch.zeros(t_max).to(device)
        self.total_de = torch.zeros(t_max).to(device)
        self.total_counts = torch.zeros(t_max).to(device)
        self.total_samples = 0
        self.total_ade_sum = 0.0
        self.total_fde_sum = 0.0

    def update(self, pred, target):
        pred = pred[:, :self.t_max, :2]
        target = target[:, :self.t_max, :2]
        B, T, _ = pred.shape

        diff = pred - target
        dist_sq = torch.sum(diff ** 2, dim=-1)
        dist = torch.sqrt(dist_sq)

        self.total_se[:T] += torch.sum(dist_sq, dim=0)
        self.total_de[:T] += torch.sum(dist, dim=0)
        self.total_counts[:T] += B

        self.total_ade_sum += torch.sum(dist).item()
        self.total_fde_sum += torch.sum(dist[:, -1]).item()
        self.total_samples += B

    def get_summary(self):
        counts = self.total_counts.clamp(min=1)
        rmse_per_step = torch.sqrt(self.total_se / counts) * self.unit_conversion
        de_avg_per_step = (self.total_de / counts) * self.unit_conversion

        overall_ade = 0.0
        overall_fde = 0.0
        if self.total_samples > 0:
            overall_ade = (self.total_ade_sum / (self.total_samples * self.t_max)) * self.unit_conversion
            overall_fde = (self.total_fde_sum / self.total_samples) * self.unit_conversion

        return {
            'rmse_per_step': rmse_per_step,
            'de_per_step': de_avg_per_step,
            'overall_ade': overall_ade,
            'overall_fde': overall_fde
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
        data_root = Path(__file__).resolve().parent.parent / '/mnt/datasets/ngsimdata'
        test_path = str(data_root / 'TestSet.mat')

    print(f"Loading test data from: {test_path}")

    test_dataset = NgsimDataset(test_path, t_h=30, t_f=50, d_s=2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True
    )
    return test_loader


def print_metrics_table(metrics, name="Model", time_indices=[4, 9, 14, 19, 24],
                        time_labels=['1s', '2s', '3s', '4s', '5s']):
    print(f'\n{"=" * 30} Test Results: {name} {"=" * 30}')
    rmse = metrics['rmse_per_step']
    de = metrics['de_per_step']

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
    test_loader = get_test_loader(args)

    model_hist = None
    model_fut = None

    script_dir = Path(__file__).resolve().parent

    arg_ckpt_path = Path(args.checkpoint_dir)
    if arg_ckpt_path.is_absolute():
        base_ckpt_dir = arg_ckpt_path
    else:
        base_ckpt_dir = script_dir / arg_ckpt_path.name

    hist_ckpt_dir = base_ckpt_dir / 'hist'
    fut_ckpt_dir = base_ckpt_dir / 'fut'

    print("\n[Init] Initializing Fut Model...")
    model_fut = DiffusionFut(args).to(device)
    load_checkpoint(model_fut, args.resume_fut, fut_ckpt_dir, device, model_name="FutModel")

    if args.eval_mode == 'joint':
        print("\n[Init] Initializing Hist Model for Joint Evaluation...")
        model_hist = DiffusionPast(args).to(device)
        load_checkpoint(model_hist, args.resume_hist, hist_ckpt_dir, device, model_name="HistModel")

    calc_fut = MetricsCalculator(args.T_f, device)
    calc_hist = MetricsCalculator(args.T, device) if args.eval_mode == 'joint' else None

    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Testing ({args.eval_mode})", ncols=120)

        for batch_idx, batch in pbar:
            hist, hist_masked, hist_mask, fut, hist_nbrs, mask, temporal_mask = prepare_input_data(
                batch, args.feature_dim, mask_type='random', mask_prob=0.5, device=device
            )

            current_hist_input = hist
            pred_hist_traj = None
            if model_hist is not None:
                _, pred_hist, _, _ = model_hist.forward_eval(hist, hist_masked, device)
                calc_hist.update(pred_hist[..., :2], hist[..., :2])

                current_hist_input = pred_hist
                pred_hist_traj = pred_hist  # 保存下来用于可视化

            _, pred_fut, _, _ = model_fut.forward_eval(current_hist_input, hist_nbrs, mask, temporal_mask, fut, device)
            calc_fut.update(pred_fut, fut)

            # hist_ego_raw = hist.unsqueeze(2)  # [B, T, 1, D]
            # mask_flat = temporal_mask.view(temporal_mask.size(0), -1, temporal_mask.size(-1))
            # mask_N_first = mask_flat.unsqueeze(2).expand(-1, -1, hist_ego_raw.size(1), -1)
            # hist_nbrs_grid = torch.zeros_like(mask_N_first, dtype=hist_nbrs.dtype)
            # hist_nbrs_grid = hist_nbrs_grid.masked_scatter_(mask_N_first.bool(), hist_nbrs)
            # hist_nbrs_aligned = hist_nbrs_grid.permute(0, 2, 1, 3).contiguous()  # [B, T, N, D]
            # full_gt_hist = torch.cat([hist_ego_raw, hist_nbrs_aligned], dim=2)  # [B, T, 1+N, D]
            #
            # visualize_batch_trajectories( hist=pred_hist_traj, hist_nbrs=full_gt_hist, future=fut, pred=pred_fut, hist_masked=hist_mask, batch_idx=0)

    if args.eval_mode == 'joint' and calc_hist:
        hist_metrics = calc_hist.get_summary()
        print_metrics_table(hist_metrics, name="History Reconstruction",
                            time_indices=[4, 9, 14], time_labels=['1s', '2s', '3s'])

    fut_metrics = calc_fut.get_summary()
    print_metrics_table(fut_metrics, name="Future Prediction")


def main():
    parser = get_args_parser()
    parser.add_argument('--eval_mode', type=str, default='joint', choices=['fut_only', 'joint'],
                        help="评估模式: 'fut_only' (使用GT历史) 或 'joint' (使用Hist模型输出)")
    parser.add_argument('--test_path', type=str, default=None, help="测试集路径 (可选，覆盖默认)")

    args = parser.parse_args()

    args.batch_size = 512
    args.num_workers = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Evaluation Mode: {args.eval_mode}")

    run_evaluation(args, device)


if __name__ == '__main__':
    main()