### 基础环境安装

```bash
ENV_NAME="finbot"
PYTHON_VERSION="3.10"

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
conda activate $ENV_NAME

echo "Installing Python packages..."
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install \
  vllm peft FlagEmbedding bitsandbytes \
  catboost xgboost  polars_ta \
  modelscope hf_transfer addict simplejson sortedcontainers openpyxl matplotlib \
  segeval backtrader deepeval
```


### 模型下载

- modelscope
    - [BAAI/bge-large-zh-v1.5](https://modelscope.cn/models/AI-ModelScope/bge-large-zh-v1.5)
    - [Qwen/Qwen2.5-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct)
    - [qolaris/FinBert](https://modelscope.cn/models/qolaris/FinBert)
    - [iic/nlp_bert_document-segmentation_chinese-base](https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/summary) 
    - [Qwen/Qwen2.5-14B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct)

终端下载指令：
```bash
MODEL=
LOCAL_PATH=
modelscope download --model $MODEL --local_dir $LOCAL_PATH
```

- huggingface
    - [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B)
    - [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)  

终端下载指令：
```bash
MODEL=
LOCAL_PATH=
huggingface-cli download --resume-download --local-dir-use-symlinks False $MODEL --local-dir $LOCAL_PATH
```

### LLM微调(可选)

1. 安装LLaMA-Factory

```bash
# 确保当前目录是 $project-root/tools
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install deepspeed=0.15.4 # 此处版本号重要! 高版本有坑
llamafactory-cli version
```

2. 在`$LLaMA-Factory/data/dataset_info.json`中注册数据集

```json
"FinCUGE_FINNA_train": {
      "file_name": "../../../resources/data/cleaned/FinCUGE_FINNA_train.jsonl",
      "formatting": "sharegpt",
      "columns": {
        "messages": "messages"
      },
      "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
        "system_tag": "system"
      }
  }
```

3. 终端通过以下命令启动训练

```bash
# 确保当前目录是项目根目录
source dev/sft_qwen2_5_3B_for_FINNA.sh
```

4. 合并权重

```bash
# 确保当前目录是项目根目录
source dev/merge_lora_adapter.sh
```