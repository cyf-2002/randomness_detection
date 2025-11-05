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

# LCG 伪随机数的模数列表（支持多模数复现实验）
# 论文中的次方：26、28、30、32、34
LCG_MODULI = [2 ** 26, 2 ** 28, 2 ** 30, 2 ** 32, 2 ** 34]

# 数据分割设置：为避免滑窗重叠导致的数据泄漏，建议使用不同随机种子生成独立的 Train/Val 序列
SPLIT_MODE = 'separate_seeds'  # 可选：'separate_seeds' | 'disjoint_ranges'
TRAIN_SEED_OFFSET = 0
VAL_SEED_OFFSET = 1000

# 组级数据规模（按论文）
GROUPS_PER_CLASS = 2 ** 18
TEST_GROUPS_TOTAL = 65536

# 快速跑（sanity check）开关：设置环境变量 QUICK_RUN=1 启用
# 可选覆盖值：QUICK_GROUPS_PER_CLASS、QUICK_TEST_GROUPS_TOTAL、QUICK_EPOCHS、QUICK_BATCH_SIZE
if os.getenv("QUICK_RUN", "0").lower() in ("1", "true", "yes"): 
	GROUPS_PER_CLASS = int(os.getenv("QUICK_GROUPS_PER_CLASS", 2 ** 12))  # 默认每类 4096 组
	TEST_GROUPS_TOTAL = int(os.getenv("QUICK_TEST_GROUPS_TOTAL", 1024))   # 默认测试 1024 组
	EPOCHS = int(os.getenv("QUICK_EPOCHS", 2))                            # 默认训练 2 轮
	BATCH_SIZE = int(os.getenv("QUICK_BATCH_SIZE", 128))                  # 默认批大小 128
