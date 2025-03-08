### 基础环境安装

```bash
ENV_NAME="finbot"
PYTHON_VERSION="3.10"

## 创建虚拟环境并激活
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
conda activate $ENV_NAME

## 设置pip国内源
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

## 安装基础依赖
pip install \
  addict simplejson sortedcontainers openpyxl matplotlib \
  vllm peft FlagEmbedding bitsandbytes modelscope hf_transfer \
  catboost xgboost polars_ta lightgbm 

## 安装评测依赖
pip install segeval backtrader deepeval 

## 安装LLM微调包--LLaMAfactory
git submodule add --force https://github.com/hiyouga/LLaMA-Factory.git tools/LLaMA-Factory
cd tools/LLaMA-Factory && pip install -e ".[torch,metrics]" && pip install deepspeed==0.15.4 && cd ../..
llamafactory-cli version

## 安装rag包--minirag v0.0.1
git submodule add --force https://github.com/HKUDS/MiniRAG.git tools/MiniRAG
cd tools/MiniRAG && git fetch --tags && git checkout tags/v0.0.1 && pip install -e . && cd ../..
```

### 部署LLM服务

通过vllm server来启动大模型，在终端输入：

``` 基座&摘要模型
CUDA_VISIBLE_DEVICES=0 vllm serve \
resources/open_models/Qwen2.5-3B-Instruct --trust-remote-code --served-model-name base   \
--enable-lora --lora-modules lora=resources/ckpts/Qwen2.5-3B-Instruct/lora_adapter \
--max-model-len 5000 --max-num-seqs 16 --quantization fp8 --gpu-memory-utilization 0.25 \
--port 12239
```

``` 评审模型
CUDA_VISIBLE_DEVICES=0,1 vllm serve \
resources/open_models/Qwen2.5-14B-Instruct --trust-remote-code --served-model-name judger  \
--max-model-len 5000 --max-num-seqs 30 --quantization fp8 --kv-cache-dtype fp8 \
--gpu-memory-utilization 0.4  --tensor-parallel-size 2 \
--port 12235 
```

### 模型基座

- modelscope
    - [AI-ModelScope/bge-large-zh-v1.5](https://modelscope.cn/models/AI-ModelScope/bge-large-zh-v1.5)
    - [Qwen/Qwen2.5-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct)
    - [qolaris/FinBert](https://modelscope.cn/models/qolaris/FinBert)
    - [iic/nlp_bert_document-segmentation_chinese-base](https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/summary) 
    - [Qwen/Qwen2.5-14B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-14B-Instruct)
    - [LLM-Research/Llama-3.2-3B-Instruct](https://modelscope.cn/models/LLM-Research/Llama-3.2-3B-Instruct)

终端下载示例：
```bash
MODEL=Qwen/Qwen2.5-14B-Instruct
LOCAL_PATH=resources/open_models/Qwen2.5-14B-Instruct
# 注意本地路径最后一级会直接作为模型文件夹
modelscope download --model $MODEL --local_dir $LOCAL_PATH
```

- huggingface
    - [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B)
    - [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)  

终端下载示例：
```bash
MODEL=ProsusAI/finbert
LOCAL_PATH=resources/open_models/ProsusAI-finbert
# 注意本地路径最后一级会直接作为模型文件夹
huggingface-cli download --resume-download --local-dir-use-symlinks False $MODEL --local-dir $LOCAL_PATH
```


### 数据源

- [金融情绪提取](https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification)
- [股价预测--市场数据](https://github.com/chenditc/investment_data)
- [股价预测--新闻数据(自爬)](https://www.eastmoney.com/)
- [新闻摘要生成](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction)
- [文档主题分割](https://github.com/fjiangAI/CPTS)

### LLM微调

这一部分主要根据[qwen2.5训练文档](https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/finetune-zh.md)改写而来。

1. 数据准备

按照`alpaca`或`sharegpt`格式之一准备和注册数据集:

- alpaca格式:

  - 单条数据形式
    ```json
    [
      {
        "instruction": "user instruction (required)",
        "input": "user input (optional)",
        "output": "model response (required)",
        "system": "system prompt (optional)",
        "history": [
          ["user instruction in the first round (optional)", "model response in the first round (optional)"],
          ["user instruction in the second round (optional)", "model response in the second round (optional)"]
        ]
      }
    ]
    ```

  - 在`$resources/dataset_info.json`中注册
    ```json
    "dataset_name": {
      "file_name": "path/to/dataset",
      "columns": {
        "prompt": "instruction",
        "query": "input",
        "response": "output",
        "system": "system",
        "history": "history"
      }
    }
    ```

- sharegpt:

  - 单条数据形式
    ```json
    [
      {
        "messages": [
          {"from": "user", "value": "user instruction"},
          {"from": "assistant", "value": "model response"}
        ],
        "system": "system prompt (optional)",
        "tools": "tool description (optional)"
      }
    ]
    ```

  - 在`$resources/dataset_info.json`中添加
    ```json
    "dataset_name": {
          "file_name": "path/to/dataset",
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


2. 启动训练

训练脚本存放为$project-root/dev/sft_*.sh的形式

调整好其中的参数后从本项目根目录通过以下命令启动:

```bash
source train/sft_qwen2_5_3B_for_FINNA.sh
```

Note: 注意DATA_SETTINGS中`template`参数与所选模型匹配,详见https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models
