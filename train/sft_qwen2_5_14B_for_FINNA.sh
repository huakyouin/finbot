#!/bin/bash

# 说明：此脚本中相对路径是相对于终端工作目录而言的。
# reference：https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/finetune-zh.md

MODEL_SETTINGS="\
--model_name_or_path resources/open_models/Qwen2.5-14B-Instruct \
--quantization_bit 4
--quantization_method bitsandbytes  
"
# quant choices: [bitsandbytes (4/8), hqq (2/3/4/5/6/8), eetq (8)]

METHOD_SETTINGS="\
--stage sft \
--do_train \
--finetuning_type lora \
--lora_target all \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.05 \
--deepspeed tools/deepspeed_z3.json \
"

DATA_SETTINGS="\
--dataset_dir resources \
--dataset FinCUGE_FINNA_train \
--template qwen \
--cutoff_len 3072 \
--preprocessing_num_workers 16 \
"

OUTPUT_SETTINGS="\
--overwrite_cache \
--overwrite_output_dir \
--output_dir resources/ckpts/Qwen2.5-14B-Instruct/qlora_adapter \
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

FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
llamafactory-cli train \
$MODEL_SETTINGS $METHOD_SETTINGS $DATA_SETTINGS $OUTPUT_SETTINGS $TRAIN_SETTINGS $EVAL_SETTINGS