# dataset.py
"""数据生成与预处理模块
包含真随机数(NumPy)与伪随机数(LCG)生成、归一化、滑动窗口与标签编码
"""
import numpy as np
from sklearn.utils import shuffle
import config

def lcg(size, seed=20, a=1103513245, c=12345, m=2**32):
    """线性同余发生器 (LCG) 生成伪随机数序列"""
    x = np.zeros(size, dtype=np.uint32)
    x[0] = seed
    for i in range(1, size):
        # 使用 Python int 计算再取模，最后安全转回 uint32
        value = (int(a) * int(x[i-1]) + int(c)) % int(m)
        x[i] = np.uint32(value)
    return x

def generate_dataset():
    """生成真随机数与伪随机数，构建二分类数据集"""
    np.random.seed(config.SEED)
    size = config.NUM_BITS
    win_len, step = config.WINDOW_SIZE, config.STEP_SIZE

    # 真随机数 (Random.org 模拟)
    true_rand = np.random.randint(0, 256, size, dtype=np.uint8)
    # 伪随机数 (LCG)
    pseudo = lcg(size)

    # Min-Max 归一化伪随机数到 [0, 255]
    x_min, x_max = np.min(pseudo), np.max(pseudo)
    pseudo = ((pseudo - x_min) / (x_max - x_min) * 255).astype(np.uint8)

    X, y = [], []
    def sliding_window(arr, label):
        for start in range(0, len(arr) - win_len, step):
            window = arr[start:start + win_len]
            X.append(window)
            y.append(label)

    # 构建样本
    sliding_window(true_rand, 0)  # Label 0: 真随机数
    sliding_window(pseudo, 1)     # Label 1: 伪随机数

    X = np.array(X, dtype=np.float32) / 255.0  # 归一化至 [0,1]
    y = np.array(y, dtype=np.int8)
    X, y = shuffle(X, y, random_state=config.SEED)
    return X[..., np.newaxis], y
