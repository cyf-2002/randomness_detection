# config.py
"""项目配置文件：定义实验参数、模型超参及随机种子"""
import os

# 随机数参数
NUM_BITS = 2 ** 18          # 生成的随机数总数
WINDOW_SIZE = 256           # 滑动窗口长度
STEP_SIZE = 3               # 滑动窗口步长

# 模型超参数
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
ATTENTION_HEADS = 4
GRU_UNITS = 64

# 路径
ROOT_DIR = os.path.dirname(__file__)
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, "saved_model")
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 随机种子保证可复现
SEED = 42
