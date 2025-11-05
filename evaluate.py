# evaluate.py
"""模型评估与随机性检测流程
包括 AO (平均输出) 与 ΔAO (平均偏差)
"""
import numpy as np
import tensorflow as tf
from dataset import generate_group_dataset
import config

AO_RANGE = (45, 55)

def detect_randomness(model_path):
    model = tf.keras.models.load_model(model_path)
    # 使用独立测试集，基于不同随机种子生成，确保与训练/验证隔离
    X, y = generate_group_dataset('test')
    y_pred = model.predict(X).ravel() * 100  # 转换为百分比

    AO_true = np.mean(y_pred[y == 0])   # 真随机数平均输出
    AO_fake = np.mean(y_pred[y == 1])   # 伪随机数平均输出
    delta_AO = abs(AO_true - AO_fake)

    print(f"AO(TrueRand)={AO_true:.2f}% | AO(LCG)={AO_fake:.2f}% | ΔAO={delta_AO:.2f}%")

    cond = (
            AO_RANGE[0] < AO_true < AO_RANGE[1] and
            AO_RANGE[0] < AO_fake < AO_RANGE[1] and
            delta_AO < 10
    )
    print("✅ 通过随机性检测" if cond else "❌ 存在可检测的非随机性")
    return AO_true, AO_fake, delta_AO

if __name__ == "__main__":
    detect_randomness(f"{config.MODEL_SAVE_PATH}/bcmnet_best.keras")
