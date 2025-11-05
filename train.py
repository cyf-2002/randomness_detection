# train.py
"""模型训练脚本:
加载数据 -> 构建 BCM-Net -> 训练 -> 保存最优模型
"""
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dataset import generate_dataset
from models.bcmnet import build_bcmnet
import config

def train_model():
    X, y = generate_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED)

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
