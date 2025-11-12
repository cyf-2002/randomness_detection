import tensorflow as tf

# 关闭 TensorFlow 的不必要日志（可选，仅为输出整洁）
tf.get_logger().setLevel('ERROR')

# 查看 GPU 数量和详情
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"可用 GPU 数量：{len(gpu_devices)}")
if gpu_devices:
    print(f"GPU 详情：{gpu_devices}")
    # 测试 GPU 是否能正常使用
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
    print("GPU 计算结果：", c.numpy())
else:
    print("未识别到 GPU，请检查环境变量或版本匹配")