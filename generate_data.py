#!/usr/bin/env python3
"""
严格按照论文要求生成训练/验证/测试数据集（滑动窗口模式）

- 真随机：优先从文件读取，不足时用 os.urandom 补充（仅用于补足，非首选）
- 伪随机：LCG，5 个模数（2^26, 2^28, 2^30, 2^32, 2^34）
- 训练+验证：各 2^18 = 262144 样本（真+伪各半）
- 测试集：65536 样本（真/伪各 32768），独立生成
- 所有划分均在原始字节流层面进行，确保无时间重叠
- LCG 按论文公式 4-2 进行离差归一化到 [0,255]
"""

import os
import numpy as np

# ===== 配置 =====
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 256   # S
STEP_SIZE = 3       # L
SEED = 42

# LCG 参数（论文 Table 4-1）
LCG_MODULI = [2**26, 2**28, 2**30, 2**32, 2**34]
LCG_A = 1103513245
LCG_C = 12345

# 数据量
N_TRAIN_VAL_PER_CLASS = 2**18      # 262144
N_TEST_TOTAL = 65536               # 测试集总数
N_TEST_TRUE = N_TEST_TOTAL // 2    # 32768
N_TEST_FAKE = N_TEST_TOTAL - N_TEST_TRUE  # 32768

# 文件路径（按优先级）
PRIMARY_TRUE_PATH = "detection/random_bin/true_random.bin"
FALLBACK_TRUE_PATH = "random_200M_blocking.bin"


# ===== 工具函数 =====
def generate_lcg_integers(n_ints: int, seed: int, m: int, a: int = 1103513245, c: int = 12345) -> np.ndarray:
    """
    生成 n_ints 个 LCG 整数（范围 [0, m)）
    返回 dtype 根据 m 自动选择（uint32 / uint64）
    使用 Python 内置大整数避免溢出
    """
    if m <= 0:
        raise ValueError("Modulus m must be positive")
    
    # 选择合适的数据类型
    if m <= 2**32:
        dtype = np.uint32
        # 使用 Python int 进行计算，避免溢出
        x = seed % m
        out = np.empty(n_ints, dtype=dtype)
        
        for i in range(n_ints):
            # 使用 Python int 进行计算
            x = (a * x + c) % m
            out[i] = dtype(x)
    elif m <= 2**64:
        dtype = np.uint64
        # 使用 Python int 进行计算，避免溢出
        x = seed % m
        out = np.empty(n_ints, dtype=dtype)
        
        for i in range(n_ints):
            # 使用 Python int 进行计算
            x = (a * x + c) % m
            out[i] = dtype(x)
    else:
        raise ValueError("m too large, not supported")
    
    return out


def min_max_normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    按论文公式 (4-2) 进行离差归一化：
        x' = floor( (x - xmin) / (xmax - xmin) * 255 )
    输入：任意整数数组（LCG 输出）
    输出：uint8 数组，值域 [0, 255]
    """
    if arr.size == 0:
        return np.array([], dtype=np.uint8)
    
    xmin = arr.min()
    xmax = arr.max()
    
    if xmin == xmax:
        # 所有值相同（理论上不会发生），返回全 0
        normalized = np.zeros_like(arr, dtype=np.float32)
    else:
        # 转为 float64 避免溢出
        arr_f = arr.astype(np.float64)
        normalized = (arr_f - xmin) / (xmax - xmin) * 255.0
    
    # 截断并转为 uint8
    return np.clip(normalized, 0, 255).astype(np.uint8)


def read_true_random_bytes(n_bytes: int) -> np.ndarray:
    """按优先级读取真随机字节"""
    data = bytearray()
    
    # 1. primary file
    if os.path.isfile(PRIMARY_TRUE_PATH):
        with open(PRIMARY_TRUE_PATH, 'rb') as f:
            data.extend(f.read())
    
    # 2. fallback file
    if len(data) < n_bytes and os.path.isfile(FALLBACK_TRUE_PATH):
        with open(FALLBACK_TRUE_PATH, 'rb') as f:
            data.extend(f.read(n_bytes - len(data)))
    
    # 3. 最后用 os.urandom 补足（应尽量避免）
    if len(data) < n_bytes:
        need = n_bytes - len(data)
        print(f"Warning: using os.urandom to fill {need} bytes")
        data.extend(os.urandom(need))
    
    return np.frombuffer(bytes(data[:n_bytes]), dtype=np.uint8)


def sliding_window_view(arr, window_size, step):
    """高效滑动窗口（避免循环）"""
    if len(arr) < window_size:
        return np.empty((0, window_size), dtype=arr.dtype)
    n_windows = (len(arr) - window_size) // step + 1
    if n_windows <= 0:
        return np.empty((0, window_size), dtype=arr.dtype)
    from numpy.lib.stride_tricks import sliding_window_view as _swv
    windows = _swv(arr, window_shape=window_size)[::step]
    return windows


def split_stream_for_train_val(stream: np.ndarray, val_ratio=0.2):
    """在字节流层面切分，确保无重叠"""
    total = len(stream)
    val_start = int(total * (1 - val_ratio))
    # 确保 val_start 至少能容纳一个窗口
    val_start = max(val_start, WINDOW_SIZE)
    
    train_stream = stream[:val_start]
    val_stream = stream[val_start:]
    
    X_train = sliding_window_view(train_stream, WINDOW_SIZE, STEP_SIZE)
    X_val = sliding_window_view(val_stream, WINDOW_SIZE, STEP_SIZE)
    
    return X_train, X_val


def distribute_counts(total: int, k: int):
    """均匀分配 total 到 k 份"""
    base = total // k
    rem = total % k
    return [base + (1 if i < rem else 0) for i in range(k)]


def bytes_needed(n_windows, window_size, step_size):
    """计算生成 n_windows 个滑动窗口所需的原始字节数"""
    if n_windows <= 0:
        return 0
    return window_size + (n_windows - 1) * step_size


# === 主函数 ===
def main():
    np.random.seed(SEED)
    
    # ===== 真随机（训练+验证）=====
    # 需要总共 N_TRAIN_VAL_PER_CLASS 个窗口，按 8:2 切分
    # 所以需要的字节数是：按总窗口数计算所需的字节数，然后在流层面切分
    true_total_bytes = bytes_needed(N_TRAIN_VAL_PER_CLASS, WINDOW_SIZE, STEP_SIZE)
    true_stream = read_true_random_bytes(n_bytes=true_total_bytes)
    X_true_train, X_true_val = split_stream_for_train_val(true_stream, val_ratio=0.2)

    # ===== 伪随机（训练+验证）=====
    # 每个模数负责一部分，然后每个模数内部按 8:2 切分
    fake_counts = distribute_counts(N_TRAIN_VAL_PER_CLASS, len(LCG_MODULI))
    fake_train_parts, fake_val_parts = [], []
    
    for i, (m, total_cnt_for_mod) in enumerate(zip(LCG_MODULI, fake_counts)):
        if total_cnt_for_mod == 0:
            continue
            
        # 为这个模数生成足够长的整数序列
        total_bytes_needed = bytes_needed(total_cnt_for_mod, WINDOW_SIZE, STEP_SIZE)
        lcg_ints = generate_lcg_integers(total_bytes_needed, seed=SEED + 10000 + i, m=m, a=LCG_A, c=LCG_C)
        lcg_u8 = min_max_normalize_to_uint8(lcg_ints)
        
        # 对这个模数的字节流进行 8:2 切分
        X_m_train, X_m_val = split_stream_for_train_val(lcg_u8, val_ratio=0.2)
        fake_train_parts.append(X_m_train)
        fake_val_parts.append(X_m_val)
    
    X_fake_train = np.concatenate(fake_train_parts, axis=0) if fake_train_parts else np.empty((0, WINDOW_SIZE), dtype=np.uint8)
    X_fake_val = np.concatenate(fake_val_parts, axis=0) if fake_val_parts else np.empty((0, WINDOW_SIZE), dtype=np.uint8)

    # ===== 构建训练/验证集 =====
    X_train_u8 = np.concatenate([X_true_train, X_fake_train], axis=0)
    y_train = np.concatenate([np.zeros(len(X_true_train)), np.ones(len(X_fake_train))])
    
    X_val_u8 = np.concatenate([X_true_val, X_fake_val], axis=0)
    y_val = np.concatenate([np.zeros(len(X_true_val)), np.ones(len(X_fake_val))])
    
    # 打乱
    idx = np.random.permutation(len(X_train_u8))
    X_train_u8, y_train = X_train_u8[idx], y_train[idx].astype(np.int8)
    idx = np.random.permutation(len(X_val_u8))
    X_val_u8, y_val = X_val_u8[idx], y_val[idx].astype(np.int8)
    
    # 归一化到 [0,1] 并加通道维（用于 CNN）
    X_train = (X_train_u8.astype(np.float32) / 255.0)[..., np.newaxis]
    X_val = (X_val_u8.astype(np.float32) / 255.0)[..., np.newaxis]

    # ===== 测试集（独立生成）=====
    # 真随机测试
    true_test_bytes_needed = bytes_needed(N_TEST_TRUE, WINDOW_SIZE, STEP_SIZE)
    true_test_stream = read_true_random_bytes(n_bytes=true_test_bytes_needed)
    X_true_test = sliding_window_view(true_test_stream, WINDOW_SIZE, STEP_SIZE)
    
    # 伪随机测试
    fake_test_counts = distribute_counts(N_TEST_FAKE, len(LCG_MODULI))
    fake_test_parts = []
    for i, (m, test_cnt_for_mod) in enumerate(zip(LCG_MODULI, fake_test_counts)):
        if test_cnt_for_mod == 0:
            continue
        test_bytes_needed = bytes_needed(test_cnt_for_mod, WINDOW_SIZE, STEP_SIZE)
        lcg_ints = generate_lcg_integers(test_bytes_needed, seed=SEED + 20000 + i, m=m, a=LCG_A, c=LCG_C)
        lcg_u8 = min_max_normalize_to_uint8(lcg_ints)
        X_m_test = sliding_window_view(lcg_u8, WINDOW_SIZE, STEP_SIZE)
        fake_test_parts.append(X_m_test)
    
    X_fake_test = np.concatenate(fake_test_parts, axis=0)
    X_test_u8 = np.concatenate([X_true_test, X_fake_test], axis=0)
    y_test = np.concatenate([np.zeros(len(X_true_test)), np.ones(len(X_fake_test))])
    
    idx = np.random.permutation(len(X_test_u8))
    X_test_u8, y_test = X_test_u8[idx], y_test[idx].astype(np.int8)
    X_test = (X_test_u8.astype(np.float32) / 255.0)[..., np.newaxis]

    # ===== 保存 =====
    np.savez_compressed(os.path.join(OUTPUT_DIR, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(OUTPUT_DIR, "test.npz"), X=X_test, y=y_test)
    
    # 打印详细统计
    print("✅ Done!")
    print(f"Train: {X_train.shape}")
    print(f"  - True: {len(X_true_train)}, Fake: {len(X_fake_train)}")
    print(f"  - Total: {len(X_train)} (Expected: {N_TRAIN_VAL_PER_CLASS * 2})")
    print(f"Val: {X_val.shape}")
    print(f"  - True: {len(X_true_val)}, Fake: {len(X_fake_val)}")
    print(f"  - Total: {len(X_val)} (Expected: {N_TRAIN_VAL_PER_CLASS * 2 // 4})")
    print(f"Test: {X_test.shape}")
    print(f"  - True: {len(X_true_test)}, Fake: {len(X_fake_test)}")
    print(f"  - Total: {len(X_test)} (Expected: {N_TEST_TOTAL})")

if __name__ == "__main__":
    main()