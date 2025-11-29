import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set diffusion predicter', add_help=False)
    # 模型参数
    parser.add_argument('--dataset', default='ngsim', type=str, help="训练/评估所用数据集，可选ngsim或highd")
    parser.add_argument('--feature_dim', default='6', type=int, help="单个轨迹特征维度，例如x,y,v,a,laneID,class")
    parser.add_argument('--input_dim', default='128', type=int, help="模型输入嵌入维度")
    parser.add_argument('--hidden_dim', default='128', type=int, help="Transformer隐藏层维度")
    parser.add_argument('--output_dim', default='2', type=int, help="预测输出维度，如未来位置(x,y)")
    parser.add_argument('--heads', default='4', type=int, help="Transformer多头注意力头数")
    parser.add_argument('--depth', default='4', type=int, help="Transformer层数")
    parser.add_argument('--dropout', default='0.1', type=float, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--mlp_ratio', default='4', type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--num_train_timesteps', default='100', type=int, help="前馈网络扩展倍数(即hidden_dim乘以该倍率)")
    parser.add_argument('--time_embedding_size', type=int, default=256)

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')

    return parser
