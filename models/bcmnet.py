# models/bcmnet.py
"""BCM-Net: 双向卷积多头注意力随机数检测网络
论文结构：CNN → Bi-GRU → Multihead Attention → Dense → Sigmoid
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import config

def build_bcmnet(input_shape=(config.WINDOW_SIZE, 1)):
    inp = layers.Input(shape=input_shape)

    # --- CNN 模块: 提取序列局部特征 ---
    x = inp
    for kernel in [3, 5, 5]:
        x = layers.Conv1D(64, kernel_size=kernel, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Dropout(0.2)(x)

    # --- Bi-GRU: 发现前后依赖关系 ---
    x = layers.Bidirectional(layers.GRU(config.GRU_UNITS, return_sequences=True))(x)

    # --- 多头注意力机制: 全局依赖捕获 ---
    attn = layers.MultiHeadAttention(num_heads=config.ATTENTION_HEADS,
                                     key_dim=config.GRU_UNITS)(x, x)
    x = layers.Add()([x, attn])  # 残差连接
    x = layers.LayerNormalization()(x)

    # --- 全连接分类层 ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inp, outputs=out, name="BCM-Net")
    return model
