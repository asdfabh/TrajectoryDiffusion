import sys
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from method_diffusion.config import get_args_parser
from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.models.fut_model import DiffusionFut
from method_diffusion.models.hist_model import DiffusionPast
from method_diffusion.run.evaluate import (
    HIST_CHECKPOINT_DIR,
    HistReconstructionMetrics,
    load_checkpoint as load_hist_checkpoint,
    print_metrics as print_hist_metrics,
)
from method_diffusion.run.evaluate_fut import (
    load_checkpoint as load_fut_checkpoint,
    print_metrics as print_fut_metrics,
)
from method_diffusion.run.train_joint import JOINT_FUT_CHECKPOINT_DIR, JOINT_HIST_CHECKPOINT_DIR
from method_diffusion.utils.fut_utils import TrajectoryMetrics
from method_diffusion.utils.mask_util import mixed_mask


def get_eval_args():
    return get_args_parser().parse_args()


def prepare_joint_batch(batch, feature_dim, device="cuda"):
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


def filter_valid_batch(batch):
    sample_valid = batch.get("sample_valid", None)
    if sample_valid is None:
        return batch
    sample_valid = sample_valid.bool()
    if sample_valid.numel() == 0 or bool(sample_valid.all()):
        return batch

    filtered = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == sample_valid.shape[0]:
            filtered[key] = value[sample_valid]
        else:
            filtered[key] = value
    return filtered


def build_hist_mask(hist, mask_ratio, random_mask_ratio, block_mask_start):
    hist_mask = mixed_mask(
        hist,
        p=mask_ratio,
        random_ratio=random_mask_ratio,
        block_start=block_mask_start,
    )
    hist_mask = hist_mask.to(hist.device)
    hist_masked = torch.cat([hist_mask * hist, hist_mask], dim=-1)
    return hist_masked, hist_mask


def resolve_hist_checkpoint_dir(resume_hist):
    resume_path = Path(str(resume_hist))
    if resume_path.exists():
        return resume_path.parent

    for checkpoint_dir in [JOINT_HIST_CHECKPOINT_DIR, HIST_CHECKPOINT_DIR]:
        if (checkpoint_dir / "checkpoint_best.pth").exists():
            return checkpoint_dir
    return HIST_CHECKPOINT_DIR


def build_test_loader(args):
    dataset_name = str(args.dataset).lower()
    data_root = Path(args.data_root_highd if dataset_name == "highd" else args.data_root_ngsim)
    test_path = data_root / "TestSet.mat"
    if not test_path.exists():
        test_path = data_root / "ValSet.mat"
    print(f"[JointEval] Dataset: {dataset_name}")
    print(f"[JointEval] Test path: {test_path}")
    test_dataset = NgsimDataset(
        str(test_path),
        t_h=30,
        t_f=50,
        d_s=2,
        enc_size=args.encoder_input_dim,
        feature_dim=args.feature_dim,
    )
    return DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(model_hist, model_fut, dataloader, device, feature_dim, num_samples, mask_ratio, random_mask_ratio, block_mask_start):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    model_hist.eval()
    model_fut.eval()
    hist_metrics = HistReconstructionMetrics()
    fut_metrics = TrajectoryMetrics(model_fut.T)
    k_samples = max(1, int(num_samples))

    pbar = tqdm(enumerate(dataloader, start=1), total=len(dataloader), desc="Eval Joint", ncols=140)
    for batch_idx, batch in pbar:
        batch = filter_valid_batch(batch)
        hist, hist_nbrs, mask, temporal_mask, fut, op_mask = prepare_joint_batch(
            batch,
            feature_dim,
            device=device,
        )

        hist_masked, hist_mask = build_hist_mask(hist, mask_ratio, random_mask_ratio, block_mask_start)
        _, pred_hist = model_hist.forward_eval(hist, hist_masked, device)
        hist_metrics.update(pred_hist, hist, hist_mask)

        if k_samples > 1:
            pred_fut, _, _ = model_fut.forwardEval_minADE(
                pred_hist,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                op_mask,
                device,
                K=k_samples,
            )
        else:
            pred_fut, _, _ = model_fut.forwardEval(
                pred_hist,
                hist_nbrs,
                mask,
                temporal_mask,
                fut,
                op_mask,
                device,
            )
        fut_metrics.update(pred_fut, fut, op_mask)

        hist_summary = hist_metrics.summary()
        fut_summary = fut_metrics.summary()
        pbar.set_postfix({
            "hist_ade_m": f"{hist_summary['xy_ade_m']['masked']:.4f}",
            "hist_rmse_m": f"{hist_summary['xy_rmse_m']['masked']:.4f}",
            "fut_ade_m": f"{fut_summary['overall_ade_m']:.4f}",
            "fut_fde_m": f"{fut_summary['overall_fde_m']:.4f}",
        })

        if batch_idx % 100 == 0:
            print_hist_metrics(hist_summary, f"Joint Hist Iteration {batch_idx}", mask_ratio, random_mask_ratio, block_mask_start)
            print_fut_metrics(fut_summary, f"Joint Fut Iteration {batch_idx}")

    final_hist = hist_metrics.summary()
    final_fut = fut_metrics.summary()
    print_hist_metrics(final_hist, "Joint Hist Reconstruction Result", mask_ratio, random_mask_ratio, block_mask_start)
    print_fut_metrics(final_fut, "Joint Fut Prediction Result")
    return final_hist, final_fut


def main():
    args = get_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist_checkpoint_dir = resolve_hist_checkpoint_dir(args.resume_hist)
    fut_checkpoint_dir = JOINT_FUT_CHECKPOINT_DIR

    print(f"[JointEval] Device: {device}")
    print(f"[JointEval] Hist checkpoint dir: {hist_checkpoint_dir}")
    print(f"[JointEval] Fut checkpoint dir: {fut_checkpoint_dir}")
    print(f"[JointEval] num_samples={args.num_samples}, num_inference_steps={args.num_inference_steps}")

    dataloader = build_test_loader(args)
    model_hist = DiffusionPast(args).to(device)
    model_fut = DiffusionFut(args).to(device)
    load_hist_checkpoint(model_hist, args.resume_hist, hist_checkpoint_dir, device, model_name="JointHistEval")
    load_fut_checkpoint(model_fut, args.resume_fut, fut_checkpoint_dir, device)
    evaluate(
        model_hist=model_hist,
        model_fut=model_fut,
        dataloader=dataloader,
        device=device,
        feature_dim=args.feature_dim,
        num_samples=args.num_samples,
        mask_ratio=max(0.0, min(1.0, float(args.mask_prob))),
        random_mask_ratio=max(0.0, min(1.0, float(args.random_mask_ratio))),
        block_mask_start=int(args.block_mask_start) > 0,
    )


if __name__ == "__main__":
    main()
