import argparse


def get_args_parser():
    parser = argparse.ArgumentParser("Set diffusion predicter", add_help=False)

    # Data
    parser.add_argument("--dataset", default="ngsim", type=str, help="训练/评估所用数据集，可选ngsim或highd")
    parser.add_argument("--data_root", default="/mnt/datasets/ngsimdata", type=str, help="数据集根目录")

    # Feature
    parser.add_argument("--feature_dim", default=4, type=int, help="单个轨迹特征维度，例如x,y,v,a,laneID,class")
    parser.add_argument("--feature_dim_fut", default=2, type=int, help="单个轨迹特征维度，例如x,y,v,a")
    parser.add_argument("--cross_feature_dim_fut", default=4, type=int, help="单个轨迹特征维度，例如x,y,v,a")

    # Train Runtime
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--save_interval", default=5, type=int)
    parser.add_argument("--mask_prob", default=0.4, type=float, help="历史轨迹随机掩码概率")
    parser.add_argument("--checkpoint_dir", default="./checkpoints", type=str, help="checkpoint根目录")
    parser.add_argument("--resume_fut", default="none", type=str, help="none/latest/best/或直接给定checkpoint路径")
    parser.add_argument("--resume_hist", default="best", type=str, help="none/latest/best/或直接给定checkpoint路径")

    # Hist Diffusion Model
    parser.add_argument("--input_dim", default=128, type=int, help="模型输入嵌入维度")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Transformer隐藏层维度")
    parser.add_argument("--output_dim", default=4, type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument("--heads", default=4, type=int, help="Transformer多头注意力头数")
    parser.add_argument("--depth", default=2, type=int, help="Transformer层数")
    parser.add_argument("--dropout", default=0.1, type=float, help="")
    parser.add_argument("--mlp_ratio", default=4, type=int, help="")
    parser.add_argument("--num_train_timesteps", default=500, type=int, help="")
    parser.add_argument("--time_embedding_size", default=256, type=int)
    parser.add_argument("--T", default=16, type=int)

    # Social/History Encoder
    parser.add_argument("--attn_nhead", default=4, type=int, help="Number of attention heads inside Social Attention Mechanism")
    parser.add_argument("--attn_out", default=16, type=int, help="Output dimension of the Social Attention Mechanism")
    parser.add_argument("--encoder_input_dim", default=64, type=int, help="Dimension of the encoder input features")
    parser.add_argument("--enc_layers", default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument("--dim_feedforward", default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--nheads", default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument("--activation", default="relu", type=str, help="Activation function in the transformer encoder: relu or None")
    parser.add_argument("--network", default="highwaynet", type=str, help="Activation function in the transformer encoder: relu or None")

    # Fut Diffusion Model
    parser.add_argument("--input_dim_fut", default=128, type=int, help="模型输入嵌入维度")
    parser.add_argument("--hidden_dim_fut", default=128, type=int, help="Transformer隐藏层维度")
    parser.add_argument("--output_dim_fut", default=2, type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument("--encoder_depth", default=2, type=int, help="Encoder层数")
    parser.add_argument("--heads_fut", default=4, type=int, help="Transformer多头注意力头数")
    parser.add_argument("--depth_fut", default=3, type=int, help="Transformer层数")
    parser.add_argument("--dropout_fut", default=0.1, type=float, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument("--mlp_ratio_fut", default=4, type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument("--num_train_timesteps_fut", default=500, type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument("--time_embedding_size_fut", default=256, type=int)
    parser.add_argument("--T_f", default=25, type=int)

    # Fut Inference Sampler
    parser.add_argument("--num_inference_steps", default=20, type=int, help="DDIM推理步数")
    parser.add_argument("--inference_timestep_spacing", default="trailing", type=str, choices=["leading", "trailing"], help="DDIM推理时间步采样策略")
    parser.add_argument("--ddim_eta", default=0.0, type=float, help="DDIM采样随机性参数eta")
    parser.add_argument("--x0_clip", default=5.0, type=float, help="推理时对归一化pred_x0进行裁剪，<=0表示不裁剪")

    # Fut Train Strategy
    parser.add_argument("--self_condition_prob", default=0.5, type=float, help="自条件训练概率，推荐0.5")
    parser.add_argument("--train_unroll_weight", default=0.4, type=float, help="2-step一致性训练损失权重，<=0表示关闭")
    parser.add_argument("--train_timestep_align_ratio", default=0.7, type=float, help="训练t采样对齐推理步点的比例，0~1")
    parser.add_argument("--train_unroll_detach_x0", default=1, type=int, help="展开去噪时构造x_{t-1}是否detach第一步x0预测，1是0否")

    # Fut Loss
    parser.add_argument("--fut_loss_mode", default="l1_time_vel", type=str, choices=["l1_time_vel", "legacy"], help="fut损失模式")
    parser.add_argument("--fut_time_weight_min", default=1.0, type=float, help="时间权重最小值")
    parser.add_argument("--fut_time_weight_max", default=2.0, type=float, help="时间权重最大值")
    parser.add_argument("--fut_pos_loss_type", default="huber", type=str, choices=["l1", "huber"], help="位置损失类型")
    parser.add_argument("--fut_huber_delta", default=1.0, type=float, help="Huber位置损失的delta")
    parser.add_argument("--fut_loss_pos_weight", default=1.0, type=float, help="位置损失权重")
    parser.add_argument("--fut_loss_vel_weight", default=0.6, type=float, help="速度损失权重")
    parser.add_argument("--fut_loss_acc_weight", default=0.15, type=float, help="加速度损失权重(legacy)")
    parser.add_argument("--fut_loss_endpoint_weight", default=0.8, type=float, help="终点损失权重(legacy)")
    parser.add_argument("--fut_high_noise_threshold", default=0.6, type=float, help="高噪声阈值（t/T）")
    parser.add_argument("--fut_high_noise_weight", default=1.5, type=float, help="高噪声样本损失权重倍率")

    # Fut Train/Eval Visualization
    parser.add_argument("--fut_enable_train_vis", default=0, type=int, help="Fut训练前向是否启用可视化show，默认关闭")
    parser.add_argument("--fut_enable_eval_vis", default=0, type=int, help="Fut评估前向是否启用可视化show，默认关闭")
    parser.add_argument("--fut_vis_every_n", default=200, type=int, help="开启可视化时每N个step可视化一次")

    # Eval
    parser.add_argument("--eval_ratio", default=0.1, type=float, help="训练过程中的推理评估比例，默认0.1表示评估10% TestSet")
    parser.add_argument("--eval_max_batches", default=0, type=int, help="训练过程中的推理评估最多使用的batch数，0表示不额外限制")
    parser.add_argument("--num_samples", default=5, type=int, help="evaluate_test 随机抽样的样本数量")
    parser.add_argument("--sample_ids", default="", type=str, help="逗号分隔的样本索引列表，设置后覆盖 num_samples")
    parser.add_argument("--sample_seed", default=None, type=int, help="随机抽样时使用的随机种子")
    return parser
