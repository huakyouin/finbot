from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS
import random
import time
from openai import OpenAI
import pandas as pd
import asyncio

import os
import csv
import logging
import json_repair
import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from functools import partial
from dataclasses import asdict

from minirag.prompt import PROMPTS
from minirag import MiniRAG
from minirag.utils import EmbeddingFunc, list_of_list_to_csv,compute_mdhash_id
from minirag.llm import openai_complete_if_cache, hf_embedding

import sys; sys.path.append("../..")
from utils.rag import prompts,retrieval,get_keyword,naive_retrival_and_answer

from transformers import AutoTokenizer, AutoModel

llm_args = dict(model="base", base_url="http://localhost:12239/v1",api_key="empty")
EMBED_MODEL_PATH = "../../resources/open_models/bge-large-zh-v1.5" 
MARKET_DATA_PATH = "../2019Q4股票预测.xlsx"
NEWS_CHUNK_PATH = "../2019-10-08_相关新闻片段.json"
RAG_ROOT = "../"

# 股票数据
print("加载数据...")
df_market_data = pd.read_excel(MARKET_DATA_PATH,index_col=0, dtype={"instrument": str})
df_market_data.rename(columns={"instrument":"id", "predict":"score","SENTI": "sentiScore"},inplace=True)
df_news_chunk = pd.read_json(NEWS_CHUNK_PATH)
print("加载RAG引擎")
PROMPTS.update(prompts) 
# 设置日志级别
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)

embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_PATH, model_max_length=512) 
embed_model = AutoModel.from_pretrained(EMBED_MODEL_PATH)

llm_client = OpenAI(base_url=llm_args['base_url'],api_key=llm_args['api_key'])
os.makedirs(os.path.join(RAG_ROOT,"rag_data"),exist_ok=True)
chunk_mapper = dict(zip(df_news_chunk['chunk_id'], df_news_chunk['content']))
rag = MiniRAG(
    working_dir=os.path.join(RAG_ROOT,"rag_data"),
    llm_model_func=lambda prompt,**kwargs: openai_complete_if_cache(prompt=prompt,**llm_args, **kwargs,), 
    llm_model_max_token_size=1000, 
    llm_model_name=llm_args["model"],
    embedding_func=EmbeddingFunc(
        embedding_dim=embed_model.config.hidden_size,
        max_token_size=embed_model.config.max_position_embeddings,
        func=partial(hf_embedding, embed_model=embed_model, tokenizer=embed_tokenizer)
    )
)

## flask相关工程
app = Flask(__name__)
CORS(app)

# 按日期获取股票集合
@app.route('/api/get_stock_option_from_date', methods=['GET'])
def get_stock_option_from_date():
    date = request.args.get('date')
    print(f"获取推荐股票：时间 {date}")
    filtered_data = df_market_data[df_market_data["datetime"] == date]
    return jsonify(filtered_data[["id","name","score","sentiScore"]].drop_duplicates().to_dict(orient="records"))

# 回测股票组合
@app.route('/api/backtest', methods=['POST'])
def backtest():
    data = request.json
    stock_ids = data.get('stocks', [])
    date = data.get('date')
    print(f"回测股票组合：{stock_ids}, 时间：{date}")

    # 确保 datetime 是日期格式
    df_market_data["datetime"] = pd.to_datetime(df_market_data["datetime"])
    date = pd.to_datetime(date)

    # 过滤指定股票
    df_filtered = df_market_data[df_market_data["id"].isin(stock_ids)].copy()
    df_filtered.sort_values(["id", "datetime"], inplace=True)

    # 计算未来收益率
    def get_future_return(series, days):
        return series.shift(-days) / series - 1  # (未来价格 - 当前价格) / 当前价格

    # 按股票 ID 分组计算
    for days in [1, 7, 30]:
        df_filtered[f"{days}d_return"] = df_filtered.groupby("id")["adjclose"].transform(lambda g: get_future_return(g, days))
    # 取出 `date` 当天的结果
    df_result = df_filtered[df_filtered["datetime"] == date][["id", "1d_return", "7d_return", "30d_return"]]

    return df_result.drop(columns=["id"]).mean().to_dict()

# 聊天机器人
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')

    type_kw,ent_kw = asyncio.run(get_keyword(question,rag.chunk_entity_relation_graph,asdict(rag))) 
    recalled_entities, recalled_chunk_ids = asyncio.run(retrieval(
        question,type_kw,ent_kw, 
        rag.chunk_entity_relation_graph,rag.entity_name_vdb,rag.relationships_vdb,rag.chunks_vdb,
        top_k= {"entity": 1, "chunk":5, "final":5}
    ))
    recalled_chunks = sorted(
        [[c_id, chunk_mapper.get(c_id,"")] for c_id in recalled_chunk_ids],
        key=lambda x: x[0]
    )
    sys_prompt = PROMPTS['answer_sys_prompt'].format(
            entities_context=list_of_list_to_csv([["entity", "score", "description"]]+recalled_entities), 
            text_units_context=list_of_list_to_csv([["id", "content"]]+recalled_chunks)
        )

    def generate():
        try:
            # 创建流式聊天请求
            chat_response = llm_client.chat.completions.create(
                model="base",  
                temperature=0.3, top_p=0.9,  
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question}
                ],
                stream=True  # 启用流式生成
            )

            # 处理 VLLM 流式数据
            for chunk in chat_response:
                content = chunk.choices[0].delta.content
                if content is None: 
                    continue # 过滤掉空数据
                yield f"{content}"  # 返回流式数据
            yield "[DONE]"  # 完成时发送结束标记

        except Exception as e:
            yield f"Error: {str(e)}\n\n"  # 错误信息

    # 设置头部避免缓存
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',  # 禁用缓存
    }

    # 返回流式响应
    return Response(generate(), headers=headers)

# 启动服务器
if __name__ == '__main__':
    app.run(host="0.0.0.0",port="12241",debug=False)