#!/bin/bash

# 设置脚本为严格模式，遇到错误时立即退出, 打印错误信息
set -ex

# 定义环境名称和 Python 版本
ENV_NAME="finbot"
PYTHON_VERSION="3.8"  ## pyqlib需要version=3.8

# 创建环境并激活
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
conda activate $ENV_NAME

# 在环境中安装 Python 包
echo "Installing Python packages..."
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm modelscope hf_transfer matplotlib pyqlib catboost xgboost openpyxl polars_ta

echo "Setup complete!"
