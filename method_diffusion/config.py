import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set diffusion predicter', add_help=False)
    # 模型参数
    parser.add_argument('--dataset', default='ngsim', type=str, help="训练/评估所用数据集，可选ngsim或highd")
    parser.add_argument('--feature_dim', default='4', type=int, help="单个轨迹特征维度，例如x,y,v,a,laneID,class")
    parser.add_argument('--input_dim', default='128', type=int, help="模型输入嵌入维度")
    parser.add_argument('--hidden_dim', default='128', type=int, help="Transformer隐藏层维度")
    parser.add_argument('--output_dim', default='4', type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument('--heads', default='4', type=int, help="Transformer多头注意力头数")
    parser.add_argument('--depth', default='2', type=int, help="Transformer层数")
    parser.add_argument('--dropout', default='0.1', type=float, help="")
    parser.add_argument('--mlp_ratio', default='4', type=int, help="")
    parser.add_argument('--num_train_timesteps', default='500', type=int, help="")
    parser.add_argument('--time_embedding_size', type=int, default=256)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--feature_dim_fut', default='2', type=int, help="单个轨迹特征维度，例如x,y,v,a")
    parser.add_argument('--cross_feature_dim_fut', default='4', type=int, help="单个轨迹特征维度，例如x,y,v,a")
    parser.add_argument('--input_dim_fut', default='128', type=int, help="模型输入嵌入维度")
    parser.add_argument('--hidden_dim_fut', default='128', type=int, help="Transformer隐藏层维度")
    parser.add_argument('--output_dim_fut', default='2', type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument('--encoder_depth', default='2', type=int, help="Encoder层数")
    parser.add_argument('--heads_fut', default='4', type=int, help="Transformer多头注意力头数")
    parser.add_argument('--depth_fut', default='3', type=int, help="Transformer层数")
    parser.add_argument('--dropout_fut', default='0.1', type=float, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--mlp_ratio_fut', default='4', type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--num_train_timesteps_fut', default=500, type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--time_embedding_size_fut', type=int, default=256)
    parser.add_argument('--T_f', type=int, default=25)

    # Encoder 参数
    parser.add_argument('--attn_nhead', default=4, type=int, help="Number of attention heads inside Social Attention Mechanism")
    parser.add_argument('--attn_out', default=16, type=int, help="Output dimension of the Social Attention Mechanism")
    parser.add_argument('--encoder_input_dim', default=64, type=int, help="Dimension of the encoder input features")
    parser.add_argument('--enc_layers', default=2, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--activation', default='relu', type=str, help="Activation function in the transformer encoder: relu or None")
    parser.add_argument('--network', default='highwaynet', type=str, help="Activation function in the transformer encoder: relu or None")

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints') #/home/lq/MaDiff/method_diffusion/checkpoints
    parser.add_argument('--resume_fut', default='epoch10', type=str, help="none/latest/best/或直接给定checkpoint路径")
    parser.add_argument('--resume_hist', default='best', type=str, help="none/latest/best/或直接给定checkpoint路径")
    parser.add_argument('--mask_prob', type=float, default=0.4, help="历史轨迹随机掩码概率")
    parser.add_argument('--data_root', type=str, default='/mnt/datasets/ngsimdata', help="数据集根目录")
    parser.add_argument('--num_inference_steps', type=int, default=20, help="DDIM推理步数")
    parser.add_argument('--inference_timestep_spacing', type=str, default='trailing',
                        choices=['leading', 'trailing'], help="DDIM推理时间步采样策略")
    parser.add_argument('--ddim_eta', type=float, default=0.0, help="DDIM采样随机性参数eta")
    parser.add_argument('--x0_clip', type=float, default=5.0,
                        help="推理时对归一化pred_x0进行裁剪，<=0表示不裁剪")
    parser.add_argument('--train_unroll_weight', type=float, default=0.4,
                        help="2-step一致性训练损失权重，<=0表示关闭")
    parser.add_argument('--train_timestep_align_ratio', type=float, default=0.7,
                        help="训练t采样对齐推理步点的比例，0~1")
    parser.add_argument('--eval_ratio', type=float, default=0.1,
                        help="训练过程中的推理评估比例，默认0.1表示评估10% TestSet")
    parser.add_argument('--eval_max_batches', type=int, default=0,
                        help="训练过程中的推理评估最多使用的batch数，0表示不额外限制")
    parser.add_argument('--num_samples', type=int, default=5, help="evaluate_test 随机抽样的样本数量")
    parser.add_argument('--sample_ids', type=str, default='', help="逗号分隔的样本索引列表，设置后覆盖 num_samples")
    parser.add_argument('--sample_seed', type=int, default=None, help="随机抽样时使用的随机种子")

    return parser
