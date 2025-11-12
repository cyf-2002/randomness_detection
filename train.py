#!/usr/bin/env python3
"""
BCM-Net 训练脚本
基于论文第四章描述的网络架构：
- CNNs (3层卷积)
- Bi-GRU 
- Multi-Head Attention
- 全连接 + Sigmoid 输出
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 设置随机种子以确保可重现性
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 配置
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
WINDOW_SIZE = 256
NUM_HEADS = 4
ATTENTION_DIM = 256

def create_bcm_net(input_shape=(256, 1)):
    """创建 BCM-Net 网络"""
    
    # 输入层
    inputs = layers.Input(shape=input_shape)
    
    # ===== CNNs 模块 =====
    # 第一层卷积
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # 第二层卷积
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # 第三层卷积
    x = layers.Conv1D(filters=64, kernel_size=5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.PReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # ===== Bi-GRU 模块 =====
    gru_forward = layers.GRU(64, return_sequences=True, dropout=0.3)
    gru_backward = layers.GRU(64, return_sequences=True, dropout=0.3, go_backwards=True)
    
    forward_output = gru_forward(x)
    backward_output = gru_backward(x)
    
    # 拼接正向和反向输出
    x = layers.Concatenate()([forward_output, backward_output])
    
    # ===== Multi-Head Attention 模块 =====
    # 线性变换得到 Q, K, V
    query = layers.Dense(ATTENTION_DIM)(x)
    key = layers.Dense(ATTENTION_DIM)(x)
    value = layers.Dense(ATTENTION_DIM)(x)
    
    # 多头注意力
    attention_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=ATTENTION_DIM // NUM_HEADS,
        dropout=0.1
    )(query, key, value)
    
    # 残差连接
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)
    
    # 全局平均池化
    x = layers.GlobalAveragePooling1D()(x)
    
    # ===== 全连接层 =====
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # 输出层 (二分类)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def load_data():
    """加载训练、验证和测试数据"""
    print("Loading data...")
    
    # 加载训练数据
    train_data = np.load(os.path.join(DATA_DIR, "train.npz"))
    X_train, y_train = train_data['X'], train_data['y']
    
    # 加载验证数据
    val_data = np.load(os.path.join(DATA_DIR, "val.npz"))
    X_val, y_val = val_data['X'], val_data['y']
    
    # 加载测试数据
    test_data = np.load(os.path.join(DATA_DIR, "test.npz"))
    X_test, y_test = test_data['X'], test_data['y']
    
    print(f"Training data shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, labels: {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, labels: {y_test.shape}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def create_callbacks():
    """创建训练回调函数"""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]
    return callbacks

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    print("\nEvaluating model on test set...")
    
    # 预测
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    
    # 计算准确率
    from sklearn.metrics import accuracy_score, classification_report
    
    y_pred_binary = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, target_names=['True Random', 'Pseudo Random']))
    
    # 计算平均输出值（用于论文中的 AO 指标）
    true_random_mask = (y_test == 0)
    pseudo_random_mask = (y_test == 1)
    
    ao_true = np.mean(y_pred[true_random_mask])
    ao_pseudo = np.mean(y_pred[pseudo_random_mask])
    delta_ao = abs(ao_true - ao_pseudo)
    
    print(f"\nAO Analysis:")
    print(f"  True Random AO: {ao_true:.4f} ({ao_true*100:.2f}%)")
    print(f"  Pseudo Random AO: {ao_pseudo:.4f} ({ao_pseudo*100:.2f}%)")
    print(f"  ΔAO: {delta_ao:.4f} ({delta_ao*100:.2f}%)")
    
    # 判断是否通过检测（根据论文 4.3.5 节）
    if (0.45 <= ao_true <= 0.55 and 0.45 <= ao_pseudo <= 0.55 and delta_ao < 0.1):
        print("✅ Pseudo random passes the randomness test!")
    else:
        print("❌ Pseudo random fails the randomness test!")
    
    return {
        'accuracy': accuracy,
        'ao_true': ao_true,
        'ao_pseudo': ao_pseudo,
        'delta_ao': delta_ao,
        'predictions': y_pred
    }

def main():
    """主训练函数"""
    print("Starting BCM-Net training...")
    
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    # 创建模型
    print("Creating BCM-Net model...")
    model = create_bcm_net()
    model.summary()
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # 创建回调函数
    callbacks = create_callbacks()
    
    # 训练模型
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    results = evaluate_model(model, X_test, y_test)
    
    # 保存最终模型
    model.save(os.path.join(MODEL_DIR, "final_model.h5"))
    print(f"\nModel saved to {MODEL_DIR}/")
    
    return model, results

if __name__ == "__main__":
    model, results = main()