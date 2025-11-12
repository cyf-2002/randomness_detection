# 验证标签分布
import numpy as np

train_data = np.load("data/train.npz")
print(f"训练集标签分布:")
print(f"真随机 (0): {np.sum(train_data['y'] == 0)}")
print(f"伪随机 (1): {np.sum(train_data['y'] == 1)}")
print(f"总样本数: {len(train_data['y'])}")