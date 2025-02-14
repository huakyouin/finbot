通过vllm server来启动大模型，在终端输入：

``` 摘要模型
vllm serve resources/open_models/Qwen2.5-3B-Instruct --trust-remote-code \
--served-model-name base \
--max-model-len 3072 --max-num-seqs 16 \
--tensor-parallel-size 4 --pipeline-parallel-size 2 --gpu-memory-utilization 0.15 \
--quantization fp8 \
--enable-lora \
--lora-modules lora=resources/ckpts/qwen2.5-3B-Instruct/lora_adapter \
--port 12239
```
