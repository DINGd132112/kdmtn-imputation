"""
K-DMTN 数据集加载模块

支持数据集(自动从benchpots获取并缓存到data/):
  PhysioNet2012, PhysioNet2019, BeijingAirQuality, Electricity,
  ETTh1, ETTm1, ItalyAirQuality, PEMSTraffic, SolarAlabama, RandomWalk

支持缺失率参数, 从 data/{file_tag}_rate{rate}.npz 加载不同缺失率数据

用法:
  # 默认缺失率
  ds = KDMINDataset("physionet2012", split="train")

  # 指定缺失率
  ds = KDMINDataset("physionet2012", missing_rate=0.3)
"""
import os, pickle, numpy as np, torch
from torch.utils.data import Dataset


# ---- 数据集注册表 ----
DATASET_REGISTRY = {
    "physionet2012": {
        "func": "preprocess_physionet2012",
        "kwargs": {"subset": "set-a", "rate": 0.1},
        "file_tag": "physionet2012",
    },
    "physionet2019": {
        "func": "preprocess_physionet2019",
        "kwargs": {"subset": "training_setA", "rate": 0.1},
        "file_tag": "physionet2019_fallback",
    },
    "beijing_air_quality": {
        "func": "preprocess_beijing_air_quality",
        "kwargs": {"n_steps": 24, "rate": 0.1},
        "file_tag": "beijing_air_quality",
    },
    "electricity": {
        "func": "preprocess_electricity_load_diagrams",
        "kwargs": {"n_steps": 24, "rate": 0.1},
        "file_tag": "electricity",
    },
    "etth1": {
        "func": "preprocess_ett",
        "kwargs": {"subset": "ETTh1", "n_steps": 24, "rate": 0.1},
        "file_tag": "etth1",
    },
    "ettm1": {
        "func": "preprocess_ett",
        "kwargs": {"subset": "ETTm1", "n_steps": 24, "rate": 0.1},
        "file_tag": "ettm1",
    },
    "italy_air_quality": {
        "func": "preprocess_italy_air_quality",
        "kwargs": {"n_steps": 24, "rate": 0.1},
        "file_tag": "italy_air_quality",
    },
    "pems_traffic": {
        "func": "preprocess_pems_traffic",
        "kwargs": {"n_steps": 24, "rate": 0.1},
        "file_tag": "pems_traffic",
    },
    "solar_alabama": {
        "func": "preprocess_solar_alabama",
        "kwargs": {"n_steps": 24, "rate": 0.1},
        "file_tag": "solar_alabama",
    },
    "random_walk": {
        "func": "preprocess_random_walk",
        "kwargs": {"n_samples_each_class": 500, "n_features": 12,
                    "n_classes": 2, "n_steps": 24, "missing_rate": 0.1},
        "file_tag": "random_walk",
    },
}

# 别名映射 (大小写兼容)
DATASET_ALIASES = {
    "physionet2012": "physionet2012",
    "PhysioNet2012": "physionet2012",
    "beijing_air_quality": "beijing_air_quality",
    "BeijingAirQuality": "beijing_air_quality",
    "electricity": "electricity",
    "Electricity": "electricity",
    "etth1": "etth1",
    "ETTh1": "etth1",
    "ettm1": "ettm1",
    "ETTm1": "ettm1",
    "italy_air_quality": "italy_air_quality",
    "ItalyAirQuality": "italy_air_quality",
    "pems_traffic": "pems_traffic",
    "PEMSTraffic": "pems_traffic",
    "solar_alabama": "solar_alabama",
    "SolarAlabama": "solar_alabama",
    "random_walk": "random_walk",
    "RandomWalk": "random_walk",
}


def resolve_dataset_name(name):
    """将别名解析为标准的注册表名称"""
    if name in DATASET_REGISTRY:
        return name
    if name in DATASET_ALIASES:
        return DATASET_ALIASES[name]
    raise ValueError(f"未知数据集 {name}. 可选: {list(DATASET_REGISTRY.keys())}")


def _load_or_download(dataset_name, cache_dir="data"):
    """加载数据集: 优先本地缓存，否则从benchpots下载并缓存"""
    dataset_name = resolve_dataset_name(dataset_name)
    assert dataset_name in DATASET_REGISTRY, \
        f"未知数据集 {dataset_name}. 可选: {list(DATASET_REGISTRY.keys())}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_cache.pkl")

    if os.path.exists(cache_path):
        print(f"[CACHE] 从本地加载: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    info = DATASET_REGISTRY[dataset_name]
    print(f"[DOWNLOAD] 从benchpots获取: {dataset_name}...")
    import benchpots.datasets as bpds
    func = getattr(bpds, info["func"])
    data = func(**info["kwargs"])

    # 统一为float32 numpy数组
    cached = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            cached[k] = v.astype(np.float32)
        elif isinstance(v, (int, float, str)):
            cached[k] = v

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)
    print(f"[CACHE] 已保存: {cache_path}")
    return cached


def _load_rate_data(dataset_name, missing_rate, cache_dir="data"):
    """从 data/{file_tag}_rate{rate}.npz 加载指定缺失率的数据"""
    dataset_name = resolve_dataset_name(dataset_name)
    file_tag = DATASET_REGISTRY[dataset_name]["file_tag"]
    rate_path = os.path.join(cache_dir, f"{file_tag}_rate{missing_rate:.1f}.npz")

    if os.path.exists(rate_path):
        print(f"[CACHE] 从缺失率缓存加载: {rate_path}")
        with np.load(rate_path, allow_pickle=True) as f:
            data = {k: f[k] for k in f.keys()}
        return data
    return None


class KDMINDataset(Dataset):
    """
    K-DMTN 统一数据集类

    参数:
        dataset_name: 数据集名称(支持别名, 如 "physionet2012" 或 "PhysioNet2012")
        split: "train" / "val" / "test"
        cache_dir: 缓存目录
        quick_test: True则采样quick_n_samples条数据
        quick_n_samples: 快速测试时每split的采样数
        missing_rate: 缺失率, 如 0.1, 0.3, 0.5, 0.7, 0.9; None 则用默认配置
    """

    def __init__(self, dataset_name, split="train", cache_dir="data",
                 quick_test=False, quick_n_samples=1000, missing_rate=None,
                 pre_impute="zero"):
        super().__init__()
        self.dataset_name = resolve_dataset_name(dataset_name)
        self.split = split
        self.quick_test = quick_test
        self.missing_rate = missing_rate
        self.pre_impute = pre_impute

        # 优先从 rate 缓存文件加载
        data = None
        if missing_rate is not None:
            data = _load_rate_data(self.dataset_name, missing_rate, cache_dir)
        if data is None:
            data = _load_or_download(self.dataset_name, cache_dir)

        self.X = torch.from_numpy(data[f"{split}_X"]).float()
        # mask: NaN -> 0, 否则1
        raw_mask = ~torch.isnan(self.X)
        self.M0 = raw_mask.float()
        self.X = self._pre_impute(self.X.numpy(), self.M0.numpy())  # 预填补缺失 (输入改进)

        # ground truth (部分数据集有 _ori)
        ori_key = f"{split}_X_ori"
        if ori_key in data:
            self.X_ori = torch.from_numpy(data[ori_key]).float()
            raw_ori_mask = ~torch.isnan(self.X_ori)
            self.M0_ori = raw_ori_mask.float()
            self.X_ori = torch.nan_to_num(self.X_ori, nan=0.0)
            self.indicating_mask = (~raw_mask & raw_ori_mask).float()
        else:
            self.X_ori = None
            self.indicating_mask = None

        # 快速测试: 采样
        if quick_test:
            n = min(len(self.X), quick_n_samples)
            idx = torch.randperm(len(self.X))[:n]
            self.X = self.X[idx]
            self.M0 = self.M0[idx]
            if self.X_ori is not None:
                self.X_ori = self.X_ori[idx]
                self.M0_ori = self.M0_ori[idx]
                self.indicating_mask = self.indicating_mask[idx]

        self.T = self.X.shape[1]
        self.N = self.X.shape[2]
        missing = 1.0 - self.M0.mean().item()
        extra = f", rate={missing_rate:.1f}" if missing_rate is not None else ""
        print(f"[DATA] {self.dataset_name}/{split}{extra}: "
              f"shape=({len(self.X)},{self.T},{self.N}), missing={missing:.1%}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        result = {
            "X_obs": self.X[idx],
            "M0": self.M0[idx],
            "X_true": self.X[idx],  # 实际值(填充0后)
        }
        if self.X_ori is not None:
            result["X_true"] = self.X_ori[idx]
            result["M0_ori"] = self.M0_ori[idx]      # 真值有效掩码 (原始有观测=1)
            result["indicating_mask"] = self.indicating_mask[idx]
        return result

    def _pre_impute(self, X, M0):
        """预填补缺失作为模型输入 (评估时这些位点由 mask 排除, 不影响指标).

        pre_impute: "zero" (0填充, 默认) / "mean" (特征全局均值) / "context" (时间最近邻)
        """
        X = np.asarray(X)
        if self.pre_impute == "zero":
            X = np.nan_to_num(X, nan=0.0)
        elif self.pre_impute == "mean":
            with np.errstate(all="ignore"):
                feat_mean = np.nanmean(X, axis=(0, 1))          # [N] 特征均值
            feat_mean = np.nan_to_num(feat_mean, nan=0.0)
            mask = np.isnan(X)
            X = X.copy()
            X[mask] = np.take(feat_mean, np.where(mask)[2])
        elif self.pre_impute == "context":
            X = X.copy()
            ns, T, N = X.shape
            idx = np.arange(T)
            for n in range(ns):
                for f in range(N):
                    seq = X[n, :, f]
                    if not np.isnan(seq).any():
                        continue
                    good = ~np.isnan(seq)
                    if good.any():
                        gi = idx[good]
                        gv = seq[gi]
                        X[n, :, f] = np.array([gv[np.argmin(np.abs(gi - t))] for t in idx])
                    else:
                        X[n, :, f] = 0.0
        else:
            X = np.nan_to_num(X, nan=0.0)
        return torch.from_numpy(X).float()

    @staticmethod
    def list_datasets():        return sorted(DATASET_REGISTRY.keys())