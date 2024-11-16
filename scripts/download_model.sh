#!/bin/bash

# 设置脚本为严格模式，遇到错误时立即退出, 打印错误信息
set -ex

ENV_NAME="finbot"

# 下载开源模型
echo "Downloading models"
mkdir -p models
conda acitvate $ENV_NAME

modelscope download --model BAAI/bge-large-zh-v1.5 --local_dir ./models/bge-large-zh-v1.5
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir ./models/Qwen2.5-3B-Instruct
modelscope download --model qolaris/FinBert --local_dir ./models/FinBert-valuesimplex

## huggingface容易下载慢/卡住, 可以清楚缓存再单独跑
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1  # 激活huggingface-cli加速模块, 网络不好可关掉
huggingface-cli download --resume-download --local-dir-use-symlinks False ProsusAI/finbert --local-dir ./models/FinBert-ProsusAI
huggingface-cli download --resume-download --local-dir-use-symlinks False openbmb/MiniCPM3-4B --local-dir ./models/MiniCPM3-4B