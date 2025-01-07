#!/bin/bash

# 说明：此脚本假定在项目的根目录中运行

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