#!/bin/bash

# 1. 激活 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 设置日志级别 (3=Error, 0=Debug, 减少不必要的屏幕输出)
export ASCEND_GLOBAL_LOG_LEVEL=3
# 屏蔽掉 Python 的部分警告
export PYTHONWARNINGS="ignore"

# 3. 运行训练
# 第一次运行时会自动下载数据集到 ./data 目录
echo "开始在 Ascend 910B 上训练 ResNet50 (CIFAR-10)..."

python train_cifar10_npu.py \
    --epochs 10 \
    --batch-size 256 \
    --lr 0.1 \
    --num-workers 32 \
    > train.log 2>&1 &

echo "训练已在后台启动，PID: $!"
echo "查看日志请运行: tail -f train.log"
