# dataset.py
"""数据生成与预处理模块
包含真随机数(NumPy)与伪随机数(LCG)生成、归一化、滑动窗口与标签编码
"""
import numpy as np
from sklearn.utils import shuffle
import config

def lcg(size, seed=20, a=1103513245, c=12345, m=2**32):
    """线性同余发生器 (LCG) 生成伪随机数序列

    说明：当模数 m > 2**32 时，使用 uint64 存储以避免溢出。
    """
    # 选择合适的存储精度
    use_uint64 = int(m) > np.iinfo(np.uint32).max
    dtype = np.uint64 if use_uint64 else np.uint32

    x = np.zeros(size, dtype=dtype)
    x[0] = dtype(seed)
    for i in range(1, size):
        # 使用 Python int 计算再取模，最后安全写回到对应 dtype
        value = (int(a) * int(x[i-1]) + int(c)) % int(m)
        x[i] = dtype(value)
    return x

def _gen_sequences(size, moduli, seed_offset=0):
    """按给定模数列表生成一对（真/伪）序列集合。
    返回：list[(true_rand_u8, pseudo_u8)] 长度 = len(moduli)
    """
    seq_pairs = []
    for idx, m in enumerate(moduli):
        # 独立的真随机序列（每个模数一段，使用不同种子偏移）
        np.random.seed(int(config.SEED) + int(seed_offset) + idx)
        true_rand = np.random.randint(0, 256, size, dtype=np.uint8)

        # 伪随机（LCG）——与真随机同长度
        pseudo_raw = lcg(size, seed=int(config.SEED) + int(seed_offset) + idx, m=int(m))
        x_min, x_max = np.min(pseudo_raw), np.max(pseudo_raw)
        if x_max == x_min:
            pseudo_u8 = (pseudo_raw & 0xFF).astype(np.uint8)
        else:
            pseudo_u8 = ((pseudo_raw - x_min) / (x_max - x_min) * 255).astype(np.uint8)
        seq_pairs.append((true_rand, pseudo_u8))
    return seq_pairs

def generate_dataset_split(split='train', moduli=None):
    """基于分割标签生成数据集，避免泄漏。

    split: 'train' | 'val'
    策略：
      - separate_seeds（默认）：训练与验证分别用不同随机种子独立生成序列。
      - disjoint_ranges：在同一长序列上使用不相交区间滑窗（需要更多小心边界）。
    """
    size = config.NUM_BITS
    win_len, step = config.WINDOW_SIZE, config.STEP_SIZE
    moduli = moduli if moduli is not None else getattr(config, 'LCG_MODULI', [2 ** 32])

    X, y = [], []

    def sliding_window(arr, label, start_idx, end_idx):
        # 在 [start_idx, end_idx) 范围内枚举窗口起点
        last_start = max(start_idx, 0)
        limit = max(end_idx - win_len, last_start)
        for s in range(last_start, limit, step):
            window = arr[s:s + win_len]
            if len(window) == win_len:
                X.append(window)
                y.append(label)

    mode = getattr(config, 'SPLIT_MODE', 'separate_seeds')
    if mode == 'separate_seeds':
        seed_offset = config.TRAIN_SEED_OFFSET if split == 'train' else config.VAL_SEED_OFFSET
        pairs = _gen_sequences(size, moduli, seed_offset=seed_offset)
        for (true_rand, pseudo_u8) in pairs:
            sliding_window(true_rand, 0, 0, len(true_rand))
            sliding_window(pseudo_u8, 1, 0, len(pseudo_u8))
    else:  # disjoint_ranges
        # 同一随机源，按不相交区间分割（训练前段，验证后段），并预留一个窗口大小的安全间隔
        pairs = _gen_sequences(size, moduli, seed_offset=0)
        boundary = int(size * 0.8)  # 80/20 划分
        gap = win_len  # 安全间隔，避免跨边界窗口
        if split == 'train':
            start_idx, end_idx = 0, max(boundary - gap, 0)
        else:
            start_idx, end_idx = boundary, size
        for (true_rand, pseudo_u8) in pairs:
            sliding_window(true_rand, 0, start_idx, end_idx)
            sliding_window(pseudo_u8, 1, start_idx, end_idx)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int8)
    X, y = shuffle(X, y, random_state=config.SEED)
    return X[..., np.newaxis], y

def generate_dataset(moduli=None):
    """生成真随机数与伪随机数，构建二分类数据集（支持多模数）。"""
    np.random.seed(config.SEED)
    size = config.NUM_BITS
    win_len, step = config.WINDOW_SIZE, config.STEP_SIZE
    moduli = moduli if moduli is not None else getattr(config, 'LCG_MODULI', [2 ** 32])

    X, y = [], []

    def sliding_window(arr, label):
        for start in range(0, len(arr) - win_len, step):
            window = arr[start:start + win_len]
            X.append(window)
            y.append(label)

    for m in moduli:
        # 真随机数 (Random.org 模拟)
        true_rand = np.random.randint(0, 256, size, dtype=np.uint8)
        # 伪随机数 (LCG with modulus m)
        pseudo = lcg(size, m=m)

        # Min-Max 归一化伪随机数到 [0, 255]
        x_min, x_max = np.min(pseudo), np.max(pseudo)
        if x_max == x_min:
            pseudo_u8 = (pseudo & 0xFF).astype(np.uint8)
        else:
            pseudo_u8 = ((pseudo - x_min) / (x_max - x_min) * 255).astype(np.uint8)

        # 构建样本
        sliding_window(true_rand, 0)  # Label 0: 真随机数
        sliding_window(pseudo_u8, 1)  # Label 1: 伪随机数

    X = np.array(X, dtype=np.float32) / 255.0  # 归一化至 [0,1]
    y = np.array(y, dtype=np.int8)
    X, y = shuffle(X, y, random_state=config.SEED)
    return X[..., np.newaxis], y


# ===== 组级数据生成（按论文：每类 2^18 组，混合后 8:2 划分；测试集重新生成 65536 组） =====
def _distribute_counts(total, k):
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def _make_true_groups(n_groups, length, base_seed):
    rng = np.random.RandomState(base_seed)
    return [rng.randint(0, 256, length, dtype=np.uint8) for _ in range(n_groups)]


def _make_pseudo_groups(n_per_mod, moduli, length, base_seed):
    groups = []
    for idx, (m, cnt) in enumerate(zip(moduli, n_per_mod)):
        for j in range(cnt):
            seq = lcg(length, seed=int(base_seed) + idx * 100000 + j, m=int(m))
            groups.append((seq & 0xFF).astype(np.uint8))
    return groups


def generate_group_dataset(split='train', moduli=None):
    """组级数据生成：
    - 训练/验证：各生成 GROUPS_PER_CLASS 组，混合后 8:2 划分
    - 测试：重新获取随机数，生成 TEST_GROUPS_TOTAL 组（真/伪各一半）
    返回：X (N, WINDOW_SIZE, 1), y (N,)
    """
    length = config.WINDOW_SIZE
    moduli = moduli if moduli is not None else getattr(config, 'LCG_MODULI', [2 ** 32])

    if split in ('train', 'val'):
        per_class = getattr(config, 'GROUPS_PER_CLASS', 2 ** 18)
        # 真随机
        true_groups = _make_true_groups(per_class, length, base_seed=int(config.SEED) + 10)
        # 伪随机：在模数间均分组数
        n_per_mod = _distribute_counts(per_class, len(moduli))
        pseudo_groups = _make_pseudo_groups(n_per_mod, moduli, length, base_seed=int(config.SEED) + 20)

        X_list = true_groups + pseudo_groups
        y_list = [0] * len(true_groups) + [1] * len(pseudo_groups)
        X = np.array(X_list, dtype=np.uint8)
        y = np.array(y_list, dtype=np.int8)

        X, y = shuffle(X, y, random_state=config.SEED)
        n_total = len(X)
        n_train = int(n_total * 0.8)
        if split == 'train':
            X_split, y_split = X[:n_train], y[:n_train]
        else:
            X_split, y_split = X[n_train:], y[n_train:]
    elif split == 'test':
        test_total = getattr(config, 'TEST_GROUPS_TOTAL', 65536)
        true_cnt = test_total // 2 + (test_total % 2)
        pseudo_cnt = test_total - true_cnt
        true_groups = _make_true_groups(true_cnt, length, base_seed=int(config.SEED) + 10010)
        n_per_mod = _distribute_counts(pseudo_cnt, len(moduli))
        pseudo_groups = _make_pseudo_groups(n_per_mod, moduli, length, base_seed=int(config.SEED) + 10020)

        X_list = true_groups + pseudo_groups
        y_list = [0] * len(true_groups) + [1] * len(pseudo_groups)
        X_split = np.array(X_list, dtype=np.uint8)
        y_split = np.array(y_list, dtype=np.int8)
        X_split, y_split = shuffle(X_split, y_split, random_state=config.SEED + 999)
    else:
        raise ValueError("split 必须是 'train' | 'val' | 'test'")

    X_split = (X_split.astype(np.float32) / 255.0)[..., np.newaxis]
    return X_split, y_split
