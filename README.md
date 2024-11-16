### 环境安装

```bash
ENV_NAME="finbot"
PYTHON_VERSION="3.8"

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
conda activate $ENV_NAME

echo "Installing Python packages..."
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm modelscope hf_transfer matplotlib catboost xgboost openpyxl polars_ta peft bitsandbytes jinjia2

```

### 数据来源

- [股市数据](https://github.com/chenditc/investment_data/releases/download/2024-08-09/qlib_bin.tar.gz)
- [中文金融情绪分类数据集](https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification/blob/master/)
- [金融多任务数据集](https://hf-mirror.com/datasets/Maciel/FinCUGE-Instruction)
- [QA检索任务数据集](https://hf-mirror.com/datasets/AIR-Bench/qa_finance_zh)


### 模型来源

- modelscope
    - [BAAI/bge-large-zh-v1.5]()
    - [Qwen/Qwen2.5-3B-Instruct]()
    - [qolaris/FinBert]()
终端下载指令：
```bash
MODEL=
LOCAL_PATH=
modelscope download --model $MODEL --local_dir $LOCAL_PATH
```

- huggingface
    - [openbmb/MiniCPM3-4B]()
    - [ProsusAI/finbert]()
终端下载指令：
```bash
MODEL=
LOCAL_PATH=
huggingface-cli download --resume-download --local-dir-use-symlinks False $MODEL --local-dir $LOCAL_PATH
```