from openai import AsyncOpenAI
import pandas as pd
import json_repair
import os
import json

eval_prompts = {}
eval_prompts['rag_relevance'] = """
你是一名专业的AI评审员，负责评估检索增强生成（RAG）系统的答案质量，重点考察其对输入问题的直接回答能力。

对于每个输入问题，系统会生成一个回答。你的任务是评估该回答是否清晰、完整地解决了用户的查询，并按照以下标准打分。

### 评估标准：
- **相关性分数 (0-5)**：答案是否直接、准确地回答了问题。
  - 5：回答完整、清晰、准确，直接解决了用户的问题，没有歧义或明显错误。
  - 4：回答基本解决了问题，但可能略显笼统或缺少一些细节。
  - 3：回答与问题主题相关，但完整度一般，仅回答了一部分问题。
  - 2：回答与问题相关，但仅提供过于通用或无实质内容的回应（如“无法确定”）。
  - 1：回答基本与问题无关，仅包含少量相关词汇或片段，可能是由于误解问题、答非所问。
  - 0：回答完全无关，或者与问题冲突。

- **理由**：请解释你的评分依据，指出回答是如何（或未能）直接解决问题的。

### 输入示例：
{
  "input": "苹果公司的最新财报表现如何？",
  "output": "苹果公司的财报显示收入增长了5%。"
}

### 输出格式：
请严格按照以下 JSON 格式返回你的评估结果，包含 `relevance_score` 和 `relevance_reason` 两个字段，不得在大括号外输出任何内容：
{
  "relevance_score": 5,
  "relevance_reason": "回答清晰、具体，直接提供了苹果公司财报的关键信息，完整解决了问题。"
}
"""

eval_prompts['rag_faithfulness'] = """
你是一名专业的AI评审员，负责评估检索增强生成（RAG）系统的答案忠实度。 

忠实度指的是生成的答案是否准确地基于检索到的文档内容，没有不准确或误导性的事实信息，但可以有额外的正确的常识.

### 评估标准：
- **忠实度分数 (0-5)**：衡量生成的答案是否基于真实的检索内容。
  - 5：答案完全忠实于检索到的文档，没有引入错误信息。
  - 4：答案基本忠实，但有轻微的措辞不准确或信息遗漏。
  - 3：答案有部分错误或夸张成分，但整体仍基于文档。
  - 2：答案有明显的错误或编造信息。
  - 1：答案存在大量事实性错误，与文档内容严重不符。
  - 0：答案完全虚构，没有任何依据。
- **理由**：请详细解释你的评分理由，并指出答案中具体的错误或不准确之处。

### 输入示例：
{
  "input": "苹果公司的最新财报表现如何？",
  "retrievals": {
    "chunk1": "苹果公司发布了2023年第二季度财报，收入增长5%。",
    "chunk2": "苹果公司在今年推出了多款新产品，包括iPhone和MacBook。",
    "entity-苹果公司": "指乔布斯任职过的公司"
  },
  "output": "苹果公司的财报显示收入增长了10%，并且在欧洲市场表现尤为突出。"
}

### 输出格式：
请严格按照以下 JSON 格式返回你的评估结果，包含faithfulness_score和faithfulness_reason两个字段，不得在大括号外输出任何内容：  
{
  "faithfulness_score": 2,
  "faithfulness_reason": "生成的答案声称收入增长了10%且欧洲市场表现突出，但检索到的文档只显示收入增长5%，并未提到欧洲市场表现。存在较明显的事实性错误。"
}
"""



async def eval_rag_performance(test_df: pd.DataFrame,judge_llm_config: dict,) -> pd.DataFrame:
    """
    对给定测试集逐条调用judge LLM进行相关性和可靠性评分，
    并生成带评分结果的 DataFrame，可选保存为 JSON。

    参数：
    - test_df: pd.DataFrame，包含至少 'input','output','retrievals'列
    - judge_llm: dict，LLM配置，如 {'model': ..., 'base_url': ..., 'api_key': ...}
    - eval_prompts: dict，包含 'relevance' 和 'faithfulness' 两个system prompt
    - result_save_path: str，可选，保存结果的JSON路径

    返回：
    - result_df: pd.DataFrame，包含原数据和评分结果
    """

    judger_client = AsyncOpenAI(base_url=judge_llm_config.pop('base_url'), api_key=judge_llm_config.pop('api_key'))
    judge = lambda messages, **kwargs: judger_client.chat.completions.create(
        messages=messages, **judge_llm_config, **kwargs
    )

    result_list = []

    for idx, row in test_df.iterrows():
        try:
            # 相关性评分
            relevance_resp = await judge(
                messages=[
                    {"role": "system", "content": eval_prompts['rag_relevance']},
                    {"role": "user", "content": str(row[['input', 'output']].to_dict())}
                ]
            )
            relevance = json_repair.loads(relevance_resp.choices[0].message.content)

            # 可靠性评分
            faithfulness_resp = await judge(
                messages=[
                    {"role": "system", "content": eval_prompts['rag_faithfulness']},
                    {"role": "user", "content": str(row[['input', 'retrievals', 'output']].to_dict())}
                ]
            )
            faithfulness = json_repair.loads(faithfulness_resp.choices[0].message.content)

            combined = dict(**row.to_dict(), **relevance, **faithfulness)
            result_list.append(combined)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            # 可选：跳过或插入空结果
            continue

    result_df = pd.DataFrame(result_list)

    mean_values = result_df[['relevance_score', 'faithfulness_score']].mean()
    print(f"Mean scores:\n{mean_values}")

    return result_df, mean_values
