### 环境配置

- 安装依赖
  ```bash
  pip install uv
  uv sync
  ```

- 激活环境
  ```bash
  source .venv/bin/activate
  ```

### 部署LLM服务

通过vllm server来启动大模型，在终端输入：

- 基座&摘要模型

  ```bash
  CUDA_VISIBLE_DEVICES=1 vllm serve \
  resources/open_models/Qwen2.5-3B-Instruct --served-model-name base \
  --max-model-len 5000 --max-num-seqs 10 --dtype auto --gpu-memory-utilization 0.65 \
  --port 12239 --trust-remote-code

  CUDA_VISIBLE_DEVICES=1 vllm serve \
  resources/open_models/Qwen2.5-3B-Instruct --served-model-name base \
  --enable-lora --lora-modules lora=resources/ckpts/Qwen2.5-3B-Instruct/lora_adapter \
  --max-model-len 20000 --max-num-seqs 10 --dtype auto --gpu-memory-utilization 0.65 \
  --port 12239 --trust-remote-code
  ```

- 评审模型

  ```bash
  CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve \
  resources/open_models/Qwen2.5-14B-Instruct  --served-model-name judger  \
  --max-model-len 5000 --max-num-seqs 10 --gpu-memory-utilization 0.6 --dtype auto --tensor-parallel-size 4 \
  --port 12235 --trust-remote-code
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
MODEL=iic/nlp_bert_document-segmentation_chinese-base
LOCAL_PATH=resources/open_models/nlp_bert_document-segmentation_chinese-base
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
- [RAGQA](https://www.kaggle.com/competitions/icaif-24-finance-rag-challenge/data)

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

  - 在`$resources/data/training/dataset_info.json`中注册
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

  - 在`$resources/data/training/dataset_info.json`中添加
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

参考：

```bash
source moddules/llm/sft_qwen2_5_3B_for_FINNA.sh
```

Note: 注意DATA_SETTINGS中`template`参数与所选模型匹配,详见https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models
