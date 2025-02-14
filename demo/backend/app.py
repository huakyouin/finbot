from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS
import random
import time
from openai import OpenAI

llm = OpenAI(base_url="http://localhost:12239/v1",api_key="empty")


chat_completion = llm.chat.completions.create(
    model="base",
    temperature=0.1, top_p=0.9, 
    messages=[{"role": "system", "content": "为以下新闻生成摘要。"}, {"role": "user","content": "你好"}],
)
chat_completion.choices[0].message.content

app = Flask(__name__)
CORS(app)

# 模拟股票数据
stocks = [
    {"id": 1, "name": "股票A", "score": 0.95},
    {"id": 2, "name": "股票B", "score": 0.89},
    {"id": 3, "name": "股票C", "score": 0.78},
    {"id": 4, "name": "股票D", "score": 0.92},
    {"id": 5, "name": "股票E", "score": 0.85},
]

# 获取股票范围
@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    return jsonify(stocks)

# 获取推荐股票
@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    date = request.args.get('date')
    print(f"获取推荐股票：时间 {date}")
    return jsonify(stocks)

# 回测股票组合
@app.route('/api/backtest', methods=['POST'])
def backtest():
    data = request.json
    stock_ids = data.get('stocks', [])
    print(f"回测股票组合：{stock_ids}")

    result = {
        "1d_return": random.uniform(-0.05, 0.05),
        "7d_return": random.uniform(-0.1, 0.1),
        "30d_return": random.uniform(-0.2, 0.2),
    }
    return jsonify(result)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')

    def generate():
        try:
            # 创建流式聊天请求
            chat_response = llm.chat.completions.create(
                model="base",  
                temperature=0.3, top_p=0.9,  
                messages=[{"role": "user", "content": message}],  # 提供用户输入
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
    app.run(debug=True)