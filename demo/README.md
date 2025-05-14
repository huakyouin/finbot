## demo说明

首先安装前端依赖：
```
cd frontend
npm install
```

然后用两个终端分别启动前后端，以及rag所需的llm：

```bash
cd frontend
npm run dev
```

```bash
cd backend
python app.py
```

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
resources/open_models/Qwen2.5-3B-Instruct --served-model-name base \
--max-model-len 5000 --max-num-seqs 16 --quantization fp8 --gpu-memory-utilization 0.25 \
--port 12239 --trust-remote-code
```