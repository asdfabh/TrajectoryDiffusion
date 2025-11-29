import torch
from torch.utils.data import DataLoader
from pathlib import Path
from method_diffusion.models.net import DiffusionPast
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.config import get_args_parser
from method_diffusion.utils.mask_util import random_mask_traj, block_mask_traj
import numpy as np


def prepare_input_data(batch, input_dim, mask_type='random', mask_prob=0.4):
    hist = batch['hist']
    nbrs = batch['nbrs']
    va = batch['va']
    nbrs_va = batch['nbrs_va']
    lane = batch['lane']
    nbrs_lane = batch['nbrs_lane']
    cclass = batch['cclass']
    nbrs_class = batch['nbrs_class']
    nbrs_num = batch['nbrs_num'].squeeze(-1)

    # 拼接特征
    if input_dim == 2:
        src = hist
        nbrs_src = nbrs
    elif input_dim == 5:
        src = torch.cat((hist, cclass, va), dim=-1)
        nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va), dim=-1)
    else:
        src = torch.cat((hist, cclass, va, lane), dim=-1)
        nbrs_src = torch.cat((nbrs, nbrs_class, nbrs_va, nbrs_lane), dim=-1)

    # 生成掩码
    B, T, _ = hist.shape
    N_total = nbrs.shape[0]

    hist_mask = torch.zeros(B, T, dtype=torch.float32)
    for b in range(B):
        traj = hist[b].cpu().numpy()
        if mask_type == 'random':
            mask = random_mask_traj(traj, p=mask_prob)
        else:
            mask = block_mask_traj(traj, missing_ratio=mask_prob)
        hist_mask[b] = torch.from_numpy(mask).float()

    nbrs_mask = torch.zeros(N_total, T, dtype=torch.float32)
    for n in range(N_total):
        traj = nbrs[n].cpu().numpy()
        if np.all(traj == 0):
            nbrs_mask[n] = 0.0
        else:
            if mask_type == 'random':
                mask = random_mask_traj(traj, p=mask_prob)
            else:
                mask = block_mask_traj(traj, missing_ratio=mask_prob)
            nbrs_mask[n] = torch.from_numpy(mask).float()

    return src, nbrs_src, hist_mask, nbrs_mask, nbrs_num


def evaluate(model, dataloader, device, input_dim, mask_type='random', mask_prob=0.4, num_inference_steps=50):
    """评估函数 - 分别计算 ego 和 nbrs 的指标"""
    model.eval()
    total_loss = 0.0

    # 分别统计 ego 和 nbrs
    ego_total_ade = 0.0
    ego_total_fde = 0.0
    ego_num_samples = 0

    nbrs_total_ade = 0.0
    nbrs_total_fde = 0.0
    nbrs_num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            hist, nbrs, hist_mask, nbrs_mask, nbrs_num = prepare_input_data(
                batch, input_dim, mask_type=mask_type, mask_prob=mask_prob
            )

            hist = hist.to(device)
            nbrs = nbrs.to(device)
            hist_mask = hist_mask.to(device)
            nbrs_mask = nbrs_mask.to(device)
            nbrs_num = nbrs_num.to(device)

            # 前向推理
            loss, pred = model.forward_test(
                hist, hist_mask, nbrs, nbrs_mask, nbrs_num,
                num_inference_steps=num_inference_steps
            )

            B = hist.shape[0]
            T = hist.shape[1]
            N_total = nbrs.shape[0]

            # pred: [B, N*T, 2],其中 N=40 (1 ego + 39 nbrs)
            pred_reshape = pred.view(B, 40, T, 2)  # [B, 40, T, 2]

            # ========== 1. 评估 Ego ==========
            pred_ego = pred_reshape[:, 0, :, :]  # [B, T, 2]
            gt_ego = batch['hist'].to(device)  # [B, T, 2]

            # 反归一化 ego
            pred_ego_denorm = model.denorm(
                torch.cat([pred_ego, torch.zeros(B, T, 4, device=device)], dim=-1)
            )[..., 0:2]

            # Ego 的有效掩码
            ego_mask_valid = hist_mask.bool()  # [B, T]

            # Ego ADE
            ego_disp = torch.norm(pred_ego_denorm - gt_ego, dim=-1)  # [B, T]
            ego_ade = (ego_disp * ego_mask_valid).sum() / ego_mask_valid.sum()

            # Ego FDE
            ego_final_idx = ego_mask_valid.sum(dim=1) - 1
            ego_fde_list = []
            for b in range(B):
                if ego_final_idx[b] >= 0:
                    ego_fde_list.append(ego_disp[b, ego_final_idx[b]])
            ego_fde = torch.stack(ego_fde_list).mean() if ego_fde_list else torch.tensor(0.0)

            ego_total_ade += ego_ade.item() * B
            ego_total_fde += ego_fde.item() * B
            ego_num_samples += B

            # ========== 2. 评估 Nbrs ==========
            if N_total > 0:
                # 重建 nbrs 的稠密预测
                pred_nbrs_dense = []  # 按批次拆分
                offset = 0
                for b in range(B):
                    count = int(nbrs_num[b])
                    if count > 0:
                        # 从 pred_reshape 中取出该批次的 nbrs 预测
                        pred_nbrs_b = pred_reshape[b, 1:count + 1, :, :]  # [count, T, 2]
                        pred_nbrs_dense.append(pred_nbrs_b)
                        offset += count

                if pred_nbrs_dense:
                    pred_nbrs_all = torch.cat(pred_nbrs_dense, dim=0)  # [N_total, T, 2]
                    gt_nbrs = batch['nbrs'].to(device)  # [N_total, T, 2]

                    # 反归一化 nbrs
                    pred_nbrs_denorm = model.denorm(
                        torch.cat([pred_nbrs_all, torch.zeros(N_total, T, 4, device=device)], dim=-1)
                    )[..., 0:2]

                    # Nbrs 的有效掩码
                    nbrs_mask_valid = nbrs_mask.bool()  # [N_total, T]

                    # Nbrs ADE
                    nbrs_disp = torch.norm(pred_nbrs_denorm - gt_nbrs, dim=-1)  # [N_total, T]
                    valid_count = nbrs_mask_valid.sum()
                    if valid_count > 0:
                        nbrs_ade = (nbrs_disp * nbrs_mask_valid).sum() / valid_count
                        nbrs_total_ade += nbrs_ade.item() * N_total

                    # Nbrs FDE
                    nbrs_final_idx = nbrs_mask_valid.sum(dim=1) - 1
                    nbrs_fde_list = []
                    for n in range(N_total):
                        if nbrs_final_idx[n] >= 0:
                            nbrs_fde_list.append(nbrs_disp[n, nbrs_final_idx[n]])
                    if nbrs_fde_list:
                        nbrs_fde = torch.stack(nbrs_fde_list).mean()
                        nbrs_total_fde += nbrs_fde.item() * N_total

                    nbrs_num_samples += N_total

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f} | Ego ADE: {ego_ade.item():.4f} | Ego FDE: {ego_fde.item():.4f}")

    # 平均指标
    avg_loss = total_loss / len(dataloader)
    avg_ego_ade = ego_total_ade / ego_num_samples
    avg_ego_fde = ego_total_fde / ego_num_samples
    avg_nbrs_ade = nbrs_total_ade / nbrs_num_samples if nbrs_num_samples > 0 else 0.0
    avg_nbrs_fde = nbrs_total_fde / nbrs_num_samples if nbrs_num_samples > 0 else 0.0

    return avg_loss, avg_ego_ade, avg_ego_fde, avg_nbrs_ade, avg_nbrs_fde


def main():
    args = get_args_parser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试集路径
    data_root = Path(__file__).resolve().parent.parent / 'data/ngsimdata'
    test_path = str(data_root / 'TestSet.mat')

    # 创建数据集
    test_dataset = NgsimDataset(test_path, t_h=30, t_f=50, d_s=2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn
    )

    # 加载模型
    model = DiffusionPast(args).to(device)

    # 加载检查点
    checkpoint_path = Path(args.checkpoint_dir) / 'checkpoint_epoch_1.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载检查点: {checkpoint_path}")
    else:
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return

    # 评估
    print("\n========== 开始评估 ==========")
    avg_loss, avg_ego_ade, avg_ego_fde, avg_nbrs_ade, avg_nbrs_fde = evaluate(
        model, test_loader, device,
        args.feature_dim,
        mask_type='random',
        mask_prob=0.4,
        num_inference_steps=50
    )

    print(f"\n========== 评估结果 ==========")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Ego ADE: {avg_ego_ade:.4f} m | Ego FDE: {avg_ego_fde:.4f} m")
    print(f"Nbrs ADE: {avg_nbrs_ade:.4f} m | Nbrs FDE: {avg_nbrs_fde:.4f} m")


if __name__ == '__main__':
    main()
