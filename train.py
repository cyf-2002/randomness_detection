# train.py
"""模型训练脚本:
加载数据 -> 构建 BCM-Net -> 训练 -> 保存最优模型
"""
import tensorflow as tf
from dataset import generate_group_dataset
from models.bcmnet import build_bcmnet
import config

def train_model():
    # 设定随机种子，增强可复现性
    tf.keras.utils.set_random_seed(config.SEED)

    # 组级数据：每类 2^18 组，混合后 8:2 划分
    X_train, y_train = generate_group_dataset('train')
    X_val, y_val = generate_group_dataset('val')

    model = build_bcmnet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.MODEL_SAVE_PATH}/bcmnet_best.keras",
            save_best_only=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks
    )
    return model, history

if __name__ == "__main__":
    model, history = train_model()
