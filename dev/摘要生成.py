from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
import time

def summarize_stock_news(from_dir, to_dir, fn_summarize):
    os.makedirs(to_dir,exist_ok=True)
    done_files = os.listdir(to_dir)
    total_time = 0  # 总耗时
    total_count = 0  # 总条数
    start_time = time.time()  # 开始时间
    for filename in os.listdir(from_dir):
        stock_id = filename.split('.')[0]
        if not filename.endswith(".json") or stock_id+".json" in done_files: continue
        print(f"当前id：{stock_id}")
        filepath = os.path.join(from_dir, filename)
        # 读取文件
        df = pd.read_json(filepath)
        df['date'] = df['date'].strftime("%Y-%m-%d %H:%M:%S")
        # 处理每行数据
        for index, row in tqdm(df.iterrows()):
            df.loc[index, 'summary'] = fn_summarize(row['chunk'])
            total_count += 1

        print(f"stock_id {stock_id} done.")  # 打印或保存结果DataFrame
        df.to_json(os.path.join(to_dir,f'{stock_id}.json'), force_ascii=False, orient='records', indent=2)
        
    end_time = time.time()  # 结束时间
    total_time = end_time - start_time  # 总耗时（秒）

    print(f"总耗时：{total_time:.2f} 秒")
    print(f"总条数：{total_count} 条")


summarizer = OpenAI(base_url="http://localhost:12239/v1",api_key="empty")

def fn_summarize(text):
    chat_completion = summarizer.chat.completions.create(
        model="lora",
        temperature=0.1, top_p=1, 
        messages=[{"role": "system", "content": "为以下新闻生成摘要。"}, {"role": "user","content": text}],
    )
    return chat_completion.choices[0].message.content

## warm up
chat_completion = summarizer.chat.completions.create(
    model="lora",
    temperature=0.1, top_p=0.9, 
    messages=[{"role": "user","content": "字节跳动是什么时候成立的？"}],
)
print(chat_completion.choices[0].message.content) 

# 调用函数处理文件夹中的所有文件
summarize_stock_news('../resources/data/CSI300news_chunked','../resources/data/CSI300news_chunked_summarized', fn_summarize)