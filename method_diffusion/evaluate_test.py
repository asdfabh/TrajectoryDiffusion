import math
import torch
from pathlib import Path
import numpy as np

from method_diffusion.models.net import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.train import prepare_input_data
from method_diffusion.utils.visualization import plot_traj, plot_traj_with_mask
from method_diffusion.evaluate import load_checkpoint


def _resolve_sample_indices(args, dataset_len):
    """根据 --sample_ids / --num_samples 生成样本索引列表"""
    if args.sample_ids:
        indices = []
        for token in args.sample_ids.split(','):
            token = token.strip()
            if not token:
                continue
            idx = int(token)
            idx = int(np.clip(idx, 0, dataset_len - 1))
            indices.append(idx)
        if not indices:
            raise ValueError("sample_ids 解析结果为空，请检查输入。")
        return indices
    num_samples = max(1, min(args.num_samples, dataset_len))
    rng = np.random.default_rng(args.sample_seed)
    return rng.choice(dataset_len, size=num_samples, replace=False).tolist()


def evaluate():
    """多样本推理: 指定或随机样本索引, 输出GT/掩码/预测并可视化"""
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'
    dataset = NgsimDataset(str(data_root / 'TestSet.mat'), t_h=30, t_f=50, d_s=2)
    sample_indices = _resolve_sample_indices(args, len(dataset))
    print(f"Dataset size: {len(dataset)} | Selected samples ({len(sample_indices)}): {sample_indices}")

    model = DiffusionPast(args).to(device)
    load_checkpoint(model, Path(args.checkpoint_dir), device)
    model.eval()

    ego_rmse_values, nbr_rmse_values = [], []

    for rank, sample_index in enumerate(sample_indices, start=1):
        print(f"\nProcessing sample {rank}/{len(sample_indices)} (idx={sample_index})")
        batch = dataset.collate_fn([dataset[sample_index]])
        hist_masked, nbrs_masked, nbrs_num, src, nbrs_src = prepare_input_data(
            batch,
            args.feature_dim,
            mask_type='random',
            mask_prob=args.mask_prob,
            device=device,
        )

        with torch.no_grad():
            pred_ego, pred_nbrs, _ = model.forward_eval(
                hist_masked,
                nbrs_masked,
                nbrs_num,
                num_inference_steps=args.num_inference_steps,
            )

        nbrs_count = int(nbrs_num.view(-1)[0].item()) if nbrs_num.numel() else 0
        gt_hist = batch['hist'][0, :, :2].cpu().numpy()
        gt_nbrs = (
            batch['nbrs'][:nbrs_count, :, :2].cpu().numpy()
            if nbrs_count > 0 else np.empty((0, gt_hist.shape[0], 2))
        )

        hist_masked_cpu = hist_masked.detach().cpu()
        nbrs_masked_cpu = nbrs_masked.detach().cpu()
        masked_hist_xy = hist_masked_cpu[0, :, 0, :-1][:, :2].numpy()
        hist_obs_flag = hist_masked_cpu[0, :, 0, -1:].numpy()
        hist_masked_with_flag = np.concatenate([masked_hist_xy, hist_obs_flag], axis=-1)
        masked_nbrs_xy = (
            nbrs_masked_cpu[0, :, :nbrs_count, :-1][:, :, :2]
            .permute(1, 0, 2)
            .numpy()
            if nbrs_count > 0 else np.empty((0, gt_hist.shape[0], 2))
        )

        pred_hist = pred_ego[0, :, 0, :2].detach().cpu().numpy()
        pred_nbrs_np = (
            pred_nbrs[:nbrs_count, :, :2].detach().cpu().numpy()
            if nbrs_count > 0 and pred_nbrs.numel() > 0
            else np.empty((0, pred_hist.shape[0], 2))
        )

        ego_rmse = float(np.sqrt(np.mean((pred_hist - gt_hist) ** 2)))
        nbr_rmse = (
            float(np.sqrt(np.mean((pred_nbrs_np - gt_nbrs) ** 2)))
            if nbrs_count > 0 else float('nan')
        )
        ego_rmse_values.append(ego_rmse)
        if not math.isnan(nbr_rmse):
            nbr_rmse_values.append(nbr_rmse)
        print(f"Ego RMSE: {ego_rmse:.3f} | Neighbor RMSE: {nbr_rmse:.3f}")

        hist_list = [gt_hist]
        nbrs_list = [gt_nbrs]
        masked_hist_list = [masked_hist_xy]
        masked_nbrs_list = [masked_nbrs_xy]
        pred_hist_list = [pred_hist]
        pred_nbrs_list = [pred_nbrs_np]

        plot_traj_with_mask(
            hist_original=hist_list,
            hist_masked=[hist_masked_with_flag],
            hist_pred=pred_hist_list,
            nbrs_original=nbrs_list,
            nbrs_masked=masked_nbrs_list,
            # nbrs_pred=pred_nbrs_list,
            fig_num1=1,
            fig_num2=1,
        )

        # 读取归一化参数
        norm_config_path = str(Path(__file__).resolve().parent / 'dataset/ngsim_stats.npz')
        norm_config = np.load(norm_config_path)
        pos_mean = norm_config['pos_mean']
        pos_std = norm_config['pos_std']

        def norm_data(x, mean, std):
            if isinstance(x, list):
                return [norm_data(item, mean, std) for item in x]

            if isinstance(x, torch.Tensor):
                x_np = x.cpu().numpy()
            else:
                x_np = np.array(x)

            if isinstance(mean, torch.Tensor):
                mean_np = mean.cpu().numpy()
            else:
                mean_np = np.array(mean)

            if isinstance(std, torch.Tensor):
                std_np = std.cpu().numpy()
            else:
                std_np = np.array(std)

            x_norm = x_np.copy()

            x_norm[..., :2] = (x_np[..., :2] - mean_np) / std_np
            x_norm[..., :2] = np.clip(x_norm[..., :2], -5.0, 5.0)

            return x_norm

        # 可视化归一化坐标系下的轨迹
        plot_traj_with_mask(
            hist_original=norm_data(hist_list, pos_mean, pos_std),
            hist_masked=[norm_data(hist_masked_with_flag, pos_mean, pos_std)],
            hist_pred=norm_data(pred_hist_list, pos_mean, pos_std),
            nbrs_original=norm_data(nbrs_list, pos_mean, pos_std) if nbrs_list is not None else None,
            nbrs_masked=norm_data(masked_nbrs_list, pos_mean, pos_std) if masked_nbrs_list is not None else None,
            nbrs_pred=None,
            fig_num1=3,
            fig_num2=3
        )

    if ego_rmse_values:
        mean_ego = float(np.mean(ego_rmse_values))
        mean_nbr = float(np.mean(nbr_rmse_values)) if nbr_rmse_values else float('nan')
        print("\nSummary:")
        print(f"Mean Ego RMSE over {len(ego_rmse_values)} samples: {mean_ego:.3f}")
        print(f"Mean Neighbor RMSE: {mean_nbr:.3f}")
    print("Inference finished.")


if __name__ == '__main__':
    evaluate()
