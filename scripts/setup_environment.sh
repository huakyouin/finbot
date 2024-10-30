#!/bin/bash

# 设置脚本为严格模式，遇到错误时立即退出, 打印错误信息
set -ex

# 定义环境名称和 Python 版本
ENV_NAME="finbot"
PYTHON_VERSION="3.10"

# 创建并激活环境
echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
source activate $ENV_NAME

# 安装 Python 包
echo "Installing Python packages..."
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm modelscope hf_transfer matplotlib

# 下载开源模型
echo "Downloading models"
mkdir -p models

download_model() {
    local source="$1" 
    local model_name="$2"
    local model_dir="$3"
    echo "Downloading $model_name from $source to ./models/$model_dir..."
    if [ "$source" = "modelscope" ]; then
        modelscope download --model "$model_name" --local_dir "./models/$model_dir"
    elif [ "$source" = "hf" ]; then
        export HF_ENDPOINT=https://hf-mirror.com
        export HF_HUB_ENABLE_HF_TRANSFER=1  # 激活huggingface-cli加速模块, 网络不好可关掉
        huggingface-cli download --resume-download --local-dir-use-symlinks False \
        "$model_name" --local-dir "./models/$model_dir"
    else
        echo "Error: Unsupported model source '$source'."
        echo " Please use 'modelscope' or 'hf'."
        return 1  # 返回非零值表示出错
    fi
}

download_model modelscope BAAI/bge-large-zh-v1.5 bge-large-zh-v1.5
download_model modelscope Qwen/Qwen2.5-3B-Instruct Qwen2.5-3B-Instruct
download_model modelscope qolaris/FinBert FinBert

## huggingface容易下载慢/卡住, 可以清楚缓存再单独跑
download_model hf ProsusAI/finbert FinBert
download_model hf openbmb/MiniCPM3-4B MiniCPM3-4B


echo "Setup complete!"
