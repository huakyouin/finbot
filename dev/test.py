import os
import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import openai_complete_if_cache, openai_embedding
from minirag.utils import EmbeddingFunc
import logging
import numpy as np

# 设置日志级别
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model="llm",
        api_key="empty",
        base_url="http://localhost:12236/v1",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="ebd",
        api_key="empty",
        base_url="http://localhost:12237/v1"
    )

LLM_MODEL = "test"  # 模型名称
DATA_PATH = "../resources/data/cleaned/LiHua-World/data/"  # 数据路径
QUERY_PATH = "../resources/data/cleaned/LiHua-World/qa/query_set.csv"  # 查询路径
WORKING_DIR = "./rag_workdir" # 工作目录
OUTPUT_PATH = "./logs/Default_output.csv"  # 输出路径

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=1000,
        func=embedding_func
    )
)

# Now indexing
def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    return txt_files

WEEK_LIST = find_txt_files(DATA_PATH)
for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"{id}/{len(WEEK_LIST)}")
    with open(WEEK) as f:
        rag.insert(f.read())

# A toy query
query = "What does LiHua predict will happen in \"The Rings of Power\"?"
answer = rag.query(query, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", "")
print(answer)