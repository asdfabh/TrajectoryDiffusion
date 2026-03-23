import argparse

def get_args_parser():
    parser = argparse.ArgumentParser("Set diffusion predicter", add_help=False)

    # Data
    parser.add_argument("--dataset", default="ngsim", type=str)
    parser.add_argument("--data_root", default="/mnt/datasets/ngsimdata", type=str)

    # Train runtime
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_epochs", default=35, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--save_interval", default=5, type=int)
    parser.add_argument("--mask_prob", default=0.4, type=float)
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str)
    parser.add_argument("--resume_fut", default="none", type=str)
    parser.add_argument("--resume_hist", default="best", type=str)

    # Joint train strategy
    parser.add_argument("--joint_freeze_hist", default=1, type=int)
    parser.add_argument("--joint_hist_loss_weight", default=0.0, type=float)
    parser.add_argument("--joint_detach_hist_for_fut", default=1, type=int)
    parser.add_argument("--joint_hist_lr_scale", default=0.2, type=float)

    # Hist diffusion model
    parser.add_argument("--input_dim", default=128, type=int)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--output_dim", default=4, type=int)
    parser.add_argument("--hist_length", default=16, type=int)
    parser.add_argument("--pred_length", default=25, type=int)
    parser.add_argument("--heads", default=4, type=int)
    parser.add_argument("--depth", default=2, type=int)
    parser.add_argument("--dec_layers", default=2, type=int)
    parser.add_argument("--pre_norm", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--mlp_ratio", default=4, type=int)
    parser.add_argument("--num_train_timesteps", default=500, type=int)
    parser.add_argument("--time_embedding_size", default=256, type=int)
    parser.add_argument("--T", default=16, type=int)

    # Social/history encoder
    parser.add_argument("--feature_dim", default=4, type=int)
    parser.add_argument("--attn_nhead", default=4, type=int)
    parser.add_argument("--attn_out", default=16, type=int)
    parser.add_argument("--encoder_input_dim", default=64, type=int)
    parser.add_argument("--enc_layers", default=2, type=int)
    parser.add_argument("--dim_feedforward", default=2048, type=int)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--network", default="highwaynet", type=str)

    # Fut diffusion model
    parser.add_argument("--hidden_dim_fut", default=128, type=int)
    parser.add_argument("--output_dim_fut", default=2, type=int)
    parser.add_argument("--heads_fut", default=4, type=int)
    parser.add_argument("--depth_fut", default=3, type=int)
    parser.add_argument("--dropout_fut", default=0.1, type=float)
    parser.add_argument("--mlp_ratio_fut", default=4, type=int)
    parser.add_argument("--num_train_timesteps_fut", default=500, type=int)
    parser.add_argument("--time_embedding_size_fut", default=256, type=int)
    parser.add_argument("--T_f", default=25, type=int)

    # Fut inference sampler
    parser.add_argument("--num_inference_steps", default=15, type=int)
    parser.add_argument("--ddim_eta", default=0.0, type=float)
    parser.add_argument("--x0_clip", default=5.0, type=float)

    # Fut train strategy
    parser.add_argument("--self_condition_prob", default=0.5, type=float)

    # Fut loss
    parser.add_argument("--fut_huber_delta", default=1.0, type=float)
    parser.add_argument("--fut_pos_loss_weight", default=1.5, type=float)
    parser.add_argument("--intent_lat_loss_weight", default=0.3, type=float)
    parser.add_argument("--intent_lon_loss_weight", default=0.3, type=float)
    parser.add_argument("--intent_use_recent_bias", default=1, type=int)
    parser.add_argument("--intent_num_prototypes", default=9, type=int)

    # Fut visualization
    parser.add_argument("--fut_enable_train_vis", default=0, type=int)
    parser.add_argument("--fut_enable_eval_vis", default=0, type=int)
    parser.add_argument("--hist_enable_train_vis", default=0, type=int)
    parser.add_argument("--hist_enable_eval_vis", default=0, type=int)

    # Eval
    parser.add_argument("--eval_ratio", default=0.5, type=float)
    parser.add_argument("--num_samples", default=5, type=int)

    return parser
