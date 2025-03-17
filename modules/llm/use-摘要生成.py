from openai import OpenAI
import pandas as pd
import os
from tqdm import tqdm
import pandas as pd
import time
import requests

def get_stock_name(stock_code):
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"{'1.' if stock_code.startswith('6') else '0.'}{stock_code}",  # 6开头是沪市，其他是深市
        "fields": "f58"  # 只获取股票名称
    }
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {})
        return data.get("f58", "股票代码不存在").replace(" ", "")
    except requests.RequestException as e:
        return f"查询失败: {e}"

def summarize_stock_news(from_dir, to_dir, fn_summarize):
    os.makedirs(to_dir,exist_ok=True)
    done_files = os.listdir(to_dir)
    total_time = 0  # 总耗时
    total_count = 0  # 总条数
    start_time = time.time()  # 开始时间
    for filename in os.listdir(from_dir):
        stock_code = filename.split('.')[0]
        if not filename.endswith(".json"): continue
        # if stock_id+".json" in done_files: continue

        print(f"当前股票代码：{stock_code}")
        filepath = os.path.join(from_dir, filename)
        # 读取文件
        df = pd.read_json(filepath)
        df['date'] = df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        df['stock_id'] = df['stock_id'].astype(str)
        df['stock_name'] = get_stock_name(stock_code)
        # 处理每行数据
        for index, row in tqdm(df.iterrows()):
            df.loc[index, 'summary'] = fn_summarize(f"这是一条关于{row['stock_name']}的新闻," + row['chunk']) if len(row['chunk'])>20 else row['chunk']
            total_count += 1

        print(f"{stock_code} done.")  # 打印或保存结果DataFrame
        df.to_json(os.path.join(to_dir,f'{stock_code}.json'), force_ascii=False, orient='records', indent=2)
        
    end_time = time.time()  # 结束时间
    total_time = end_time - start_time  # 总耗时（秒）

    print(f"总耗时：{total_time:.2f} 秒")
    print(f"总条数：{total_count} 条")


summarizer = OpenAI(base_url="http://localhost:12239/v1",api_key="empty")

def fn_summarize(text):
    chat_completion = summarizer.chat.completions.create(
        model="lora",
        messages=[{"role": "system", "content": "为以下新闻生成摘要。"}, {"role": "user","content": text}],
        temperature=0.1, top_p=1, max_tokens=500,
    )
    return chat_completion.choices[0].message.content

## warm up
print(fn_summarize("字节跳动是什么时候成立的？")) 

# 调用函数处理文件夹中的所有文件
summarize_stock_news('../../resources/data/CSI300news_chunked','../../resources/data/CSI300news_chunked_summarized', fn_summarize)