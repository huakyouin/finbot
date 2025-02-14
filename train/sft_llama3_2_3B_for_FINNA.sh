#!/bin/bash

# 说明：此脚本中相对路径是相对于终端工作目录而言的。
# ：https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/finetune-zh.md

MODEL_SETTINGS="--model_name_or_path resources/open_models/Llama-3.2-3B-Instruct"

METHOD_SETTINGS="\
--stage sft \
--do_train \
--finetuning_type lora \
--lora_target all \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--deepspeed tools/deepspeed_z2.json \
"

# 注意template参数,
# https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#supported-models
DATA_SETTINGS="\
--dataset_dir tools/LLaMA-Factory/data \
--dataset FinCUGE_FINNA_train \
--template llama3 \
--cutoff_len 3072 \
--preprocessing_num_workers 16 \
"

OUTPUT_SETTINGS="\
--overwrite_cache \
--overwrite_output_dir \
--output_dir resources/ckpts/llama3.2-3B-Instruct/lora_adapter \
--logging_steps 100 \
--save_steps 100 \
--plot_loss \
"

TRAIN_SETTINGS="\
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 1.0e-4 \
--num_train_epochs 2.0 \
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
$MODEL_SETTINGS $METHOD_SETTINGS $DATA_SETTINGS $OUTPUT_SETTINGS $TRAIN_SETTINGS $EVAL_SETTINGS