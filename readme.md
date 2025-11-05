# BCM-Net 随机数检测复现工程

## 一、环境要求
- Python ≥ 3.8
- TensorFlow 2.9+
- NumPy, Scikit‑learn

## 二、文件说明
| 文件 | 功能 |
|------|------|
| `dataset.py` | 生成真随机数与伪随机数，数据预处理 |
| `models/bcmnet.py` | BCM-Net 模型结构定义 |
| `train.py` | 模型训练与保存 |
| `evaluate.py` | 检测与判定 AO、ΔAO |
| `config.py` | 全局参数设置 |

## 三、运行步骤
1. 安装依赖
   ```bash
   pip install tensorflow numpy scikit-learn
2. 训练模型
    ```bash
    python train.py
3. 执行随机性检测
    ```bash
    python evaluate.py
