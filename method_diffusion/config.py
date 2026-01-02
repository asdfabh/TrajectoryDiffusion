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
    parser.add_argument('--dropout', default='0.1', type=float, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--mlp_ratio', default='4', type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--num_train_timesteps', default=100, type=int, help="扩散步数") # Increased default for better diffusion
    parser.add_argument('--time_embedding_size', type=int, default=256)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--N', type=int, default=11)
    parser.add_argument('--feature_dim_fut', default='2', type=int, help="单个轨迹特征维度，例如x,y,v,a")
    parser.add_argument('--cross_feature_dim_fut', default='4', type=int, help="单个轨迹特征维度，例如x,y,v,a")
    parser.add_argument('--input_dim_fut', default='128', type=int, help="模型输入嵌入维度")
    parser.add_argument('--hidden_dim_fut', default='128', type=int, help="Transformer隐藏层维度")
    parser.add_argument('--output_dim_fut', default='2', type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument('--latent_dim_fut', default='64', type=int, help="潜在空间维度 (Deprecated in new design)")
    parser.add_argument('--heads_fut', default='4', type=int, help="Transformer多头注意力头数")
    parser.add_argument('--depth_fut', default='4', type=int, help="Transformer层数")
    parser.add_argument('--dropout_fut', default='0.1', type=float, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--mlp_ratio_fut', default='4', type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--num_train_timesteps_fut', default=100, type=int, help="扩散步数")
    parser.add_argument('--time_embedding_size_fut', type=int, default=256)
    parser.add_argument('--T_f', type=int, default=25)
    parser.add_argument('--N_f', type=int, default=11)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=256) # Adjusted for larger model
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str, default='/home/lq/MaDiff/method_diffusion/checkpoints') # /home/lq/MaDiff/method_diffusion/checkpoints
    parser.add_argument('--resume', default='none', type=str, help="none/latest/best/或直接给定checkpoint路径")
    parser.add_argument('--mask_prob', type=float, default=0.4, help="历史轨迹随机掩码概率")
    parser.add_argument('--num_inference_steps', type=int, default=5, help="DDIM推理步数")
    parser.add_argument('--preview_limit', type=int, default=16, help="评估阶段最多保存的示例数")
    parser.add_argument('--num_samples', type=int, default=10, help="evaluate_test 随机抽样的样本数量")
    parser.add_argument('--sample_ids', type=str, default='', help="逗号分隔的样本索引列表，设置后覆盖 num_samples")
    parser.add_argument('--sample_seed', type=int, default=None, help="随机抽样时使用的随机种子")

    # 分步训练参数
    parser.add_argument('--train_mode', type=str, default='fut_only', choices=['joint', 'past_only', 'fut_only'], help="训练模式：joint(联合), past_only(仅历史), fut_only(仅未来)")
    parser.add_argument('--pretrained_past', type=str, default='', help="预训练的Past模型路径")
    parser.add_argument('--pretrained_fut', type=str, default='', help="预训练的Fut模型路径")

    return parser
