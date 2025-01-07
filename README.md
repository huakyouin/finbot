### 基础环境安装

```bash
ENV_NAME="finbot"
PYTHON_VERSION="3.10"

echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION notebook -y
conda activate $ENV_NAME

echo "Installing Python packages..."
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm peft FlagEmbedding bitsandbytes
pip install hf_transfer matplotlib catboost xgboost openpyxl polars_ta
pip install modelscope addict simplejson sortedcontainers
pip install segeval backtrader
```


### 模型下载

- modelscope
    - [BAAI/bge-large-zh-v1.5](https://modelscope.cn/models/AI-ModelScope/bge-large-zh-v1.5)
    - [Qwen/Qwen2.5-3B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct)
    - [qolaris/FinBert](https://modelscope.cn/models/qolaris/FinBert)
    - [iic/nlp_bert_document-segmentation_chinese-base](https://modelscope.cn/models/iic/nlp_bert_document-segmentation_chinese-base/summary)  

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

### LLM微调(可选)

1. 安装LLaMA-Factory

```bash
# 确保当前目录是 $project-root/tools
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
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
# 确保当前目录是 $project-root
MODEL_SETTINGS="\
--model_name_or_path resources/open_models/Qwen2.5-3B-Instruct \
--stage sft \
--do_train \
--finetuning_type lora \
--lora_target all \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
"

DATA_SETTINGS="
--dataset_dir tools/LLaMA-Factory/data \
--dataset FinCUGE_FINNA_train \
--template qwen \
--cutoff_len 3072 \
--overwrite_cache \
--overwrite_output_dir \
--preprocessing_num_workers 16 \
"

OUTPUT_SETTINGS="
--output_dir resources/ckpts/qwen2.5-3B-Instruct/lora_adapter \
--logging_steps 100 \
--save_steps 100 \
--plot_loss \
"

TRAIN_SETTINGS="\
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 1.0e-4 \
--num_train_epochs 0.1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.1 \
--bf16 \
--ddp_timeout 180000000 \
"

EVAL_SETTINGS="\
--val_size 0.1 \
--per_device_eval_batch_size 1 \
--eval_strategy steps \
--eval_steps 500 \
"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
llamafactory-cli train \
$MODEL_SETTINGS $DATA_SETTINGS $OUTPUT_SETTINGS $TRAIN_SETTINGS $EVAL_SETTINGS
```

4. 合并权重

```bash
MODEL_SETTINGS="\
--model_name_or_path Qwen/Qwen2-7B-Instruct \
--adapter_name_or_path PATH-TO-LORA \
--template qwen \
--finetuning_type lora \
"

EXPORT_SETTINGS="\
--export_dir models/qwen2-7b-sft-lora-merged \
--export_size 2 \
--export_device cpu \
--export_legacy_format false \
"
llamafactory-cli export $MODEL_SETTINGS $EXPORT_SETTINGS
```