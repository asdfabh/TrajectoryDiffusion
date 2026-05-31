from pathlib import Path

from method_diffusion.dataset.ngsim_dataset import NgsimDataset
from method_diffusion.dataset.ngsim_hist_dataset import NgsimHistDataset
from method_diffusion.dataset.round_dataset import RoundDataset, RoundHistDataset


SUPPORTED_DATASETS = ("ngsim", "highd", "round")

# 数据集默认时间参数: (t_h, t_f, d_s, fut_steps)
#   t_h, t_f = 原始帧数 (0.1s/帧)
#   d_s       = 降采样步长 (2 → 0.2s/步)
#   fut_steps = 模型输出的未来步数 (= t_f / d_s)
_DATASET_TIME_PARAMS = {
    "ngsim":  (30, 50, 2, 25),
    "highd":  (30, 50, 2, 25),
    "round":  (20, 40, 2, 20),
}


def normalize_dataset_name(dataset):
    dataset_name = str(dataset).strip().lower()
    if dataset_name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS)
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return dataset_name


def get_time_params(dataset):
    """返回 (t_h, t_f, d_s, fut_steps)。调用方可用 fut_steps 设置模型 T_f。"""
    return _DATASET_TIME_PARAMS[normalize_dataset_name(dataset)]


def get_data_root(args, dataset):
    dataset_name = normalize_dataset_name(dataset)
    attr_name = f"data_root_{dataset_name}"
    if not hasattr(args, attr_name):
        raise AttributeError(f"Missing args.{attr_name}")
    return Path(getattr(args, attr_name))


def get_split_path(args, dataset, split_name):
    return get_data_root(args, dataset) / f"{split_name}Set.mat"


def get_test_split_path(args, dataset):
    test_path = get_split_path(args, dataset, "Test")
    if not test_path.exists():
        return get_split_path(args, dataset, "Val")
    return test_path


def build_trajectory_dataset(mat_path, dataset, **kwargs):
    """构造轨迹预测 Dataset。按数据集自动注入 t_h/t_f/d_s 默认值。"""
    dataset_name = normalize_dataset_name(dataset)
    defaults = dict(zip(("t_h", "t_f", "d_s"), _DATASET_TIME_PARAMS[dataset_name][:3]))
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    if dataset_name == "round":
        return RoundDataset(str(mat_path), **kwargs)
    return NgsimDataset(str(mat_path), **kwargs)


def build_hist_dataset(mat_path, dataset, **kwargs):
    """构造 History 重建 Dataset。按数据集自动注入 t_h/d_s 默认值。"""
    dataset_name = normalize_dataset_name(dataset)
    t_h, _, d_s, _ = _DATASET_TIME_PARAMS[dataset_name]
    kwargs.setdefault("t_h", t_h)
    kwargs.setdefault("d_s", d_s)
    if dataset_name == "round":
        return RoundHistDataset(str(mat_path), **kwargs)
    return NgsimHistDataset(str(mat_path), **kwargs)


def meter_per_unit(dataset):
    dataset_name = normalize_dataset_name(dataset)
    if dataset_name == "round":
        return 1.0
    return 0.3048
