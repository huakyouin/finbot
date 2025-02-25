<template>
  <a-layout style="min-height: 100vh">
    <!-- 头部选择栏 -->
    <a-layout-header style="background: #fff; padding: 12px 24px">
      <div style="display: flex; gap: 20px; align-items: center">
        <a-date-picker v-model:value="selectedDate" placeholder="选择日期" style="width: 200px" />
        <a-select mode="multiple" placeholder="选择股票池" style="flex: 1" :options="stockOptions" />
      </div>
    </a-layout-header>

    <!-- 主体三列布局 -->
    <a-layout-content style="padding: 16px; height: calc(100vh - 64px)">
      <div class="panel-container">
        <!-- 推荐股票 -->
        <a-card class="panel" title="推荐股票">
          <div class="stock-list">
            <div 
              v-for="stock in recommendedStocks"
              :key="stock.id"
              class="stock-item"
              :style="{ background: getColor(stock.score) }"
            >
              <span class="stock-info">{{ stock.name }} ({{ (stock.score * 100).toFixed(1) }}分)</span>
              <a-button size="small" @click="addToPortfolio(stock)" style="margin-left: auto;">
                <ArrowRightOutlined />
              </a-button>
            </div>
          </div>
        </a-card>

        <!-- 未来真实收益面板 -->
        <a-card class="panel" title="未来真实收益">
          <!-- 子面板：股票组合 -->
          <a-card class="sub-panel" title="所选组合">
            <div class="portfolio">
              <span 
                v-for="s in portfolio" 
                :key="s.id"
                class="stock-tag"
                :style="{ background: getColor(s.score) }"
              >
                {{ s.name }}
                <a-button size="small" @click="removeFromPortfolio(s)" style="margin-left: 8px; padding: 0;" type="text">
                  <CloseOutlined />
                </a-button>
              </span>
            </div>
          </a-card>

          <!-- 子面板：回测结果与按钮 -->
          <a-card class="sub-panel" title="回测结果">
            <a-button block type="primary" @click="handleBacktest">测试收益</a-button>
            <div v-if="backtestResult" class="result">
              <p>1日: {{ (backtestResult['1d_return'] * 100).toFixed(1) }}%</p>
              <p>7日: {{ (backtestResult['7d_return'] * 100).toFixed(1) }}%</p>
              <p>30日: {{ (backtestResult['30d_return'] * 100).toFixed(1) }}%</p>
            </div>
          </a-card>
        </a-card>

        <!-- 智能助手 -->
        <a-card class="panel" title="智能助手">
          <div class="chat">
            <div class="messages">
              <div v-for="(msg, i) in messages" :key="i" :class="['message', msg.user]">
                {{ msg.content }}
              </div>
            </div>
            <div class="input-box">
              <a-input v-model:value="inputMsg" @pressEnter="handleSend" placeholder="输入问题..." />
              <a-button type="primary" @click="handleSend">发送</a-button>
            </div>
          </div>
        </a-card>
      </div>
    </a-layout-content>
  </a-layout>
</template>



<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import dayjs from 'dayjs';
import { ArrowRightOutlined, CloseOutlined } from '@ant-design/icons-vue';

const API_BASE_URL = 'http://localhost:12241';

// 状态管理
const selectedDate = ref(dayjs());
const stockOptions = ref([]);
const recommendedStocks = ref([]);
const portfolio = ref([]);
const backtestResult = ref(null);
const messages = ref([]);
const inputMsg = ref('');

// 初始化数据
onMounted(async () => {
  const res = await axios.get(`${API_BASE_URL}/api/stocks`);
  stockOptions.value = res.data.map(s => ({ value: s.id, label: s.name }));
  loadRecommendations();
});

// 颜色处理
const getColor = (score) => `hsl(${score * 120}, 70%, 90%)`;

// 股票管理
const addToPortfolio = (stock) => { 
  if (!portfolio.value.some(s => s.id === stock.id)) {
    portfolio.value.push(stock);
  }
};

const removeFromPortfolio = (stock) => {
  portfolio.value = portfolio.value.filter(s => s.id !== stock.id);
};

// 获取推荐股票
const loadRecommendations = async () => {
  const res = await axios.get(`${API_BASE_URL}/api/recommend`, {
    params: { date: selectedDate.value.format('YYYY-MM-DD') }
  });
  recommendedStocks.value = res.data;
};

// 回测请求
const handleBacktest = async () => {
  const res = await axios.post(`${API_BASE_URL}/api/backtest`, {
    date: selectedDate.value.format('YYYY-MM-DD'),
    stocks: portfolio.value.map(s => s.id)
  });
  backtestResult.value = res.data;
};

// 流式对话处理
const handleSend = async () => {
  const messageContent = inputMsg.value.trim();
  if (!messageContent) return;

  messages.value.push({ user: 'me', content: messageContent });
  inputMsg.value = '';

  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: messageContent })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const botMessageIndex = messages.value.length;
  messages.value.push({ user: 'bot', content: '' });

  let reply = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const decodedValue = decoder.decode(value, { stream: true }).trim();
    if (decodedValue === '[DONE]') break;

    reply += decodedValue;
    messages.value[botMessageIndex] = { user: 'bot', content: reply };
  }
};
</script>


<style scoped>

.panel-container {
  display: flex;
  gap: 16px;
  height: 100%;
}

.panel {
  flex: 1;
  display: flex;
  flex-direction: column;
}

.stock-item {
  padding: 8px;
  margin: 4px;
  border-radius: 4px;
  display: flex;
  justify-content: space-between; /* 确保按钮和文本分别位于两端 */
  align-items: center; /* 垂直居中对齐 */
}

.stock-item .ant-btn {
  background: none;        /* 去掉背景 */
  border: none;            /* 去掉边框 */
  padding: 0;              /* 去掉按钮的内边距 */
  font-size: 16px;         /* 设置合适的字体大小 */
}

/* 子面板样式 */
.sub-panel {
  margin-top: 16px;
  padding: 16px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
}

.sub-panel .result {
  margin-top: 16px;
  padding: 12px;
  background-color: #fff;
  border-radius: 4px;
  border: 1px solid #ddd;
}

.sub-panel .result p {
  margin: 8px 0;
}

/* 股票标签样式 */
.stock-tag {
  display: inline-flex;
  align-items: center;
  padding: 4px 8px;
  margin-right: 8px;
  border-radius: 12px;
  background: #e6f7ff;
  font-size: 14px;
}

.stock-tag .ant-btn {
  margin-left: 8px;
  padding: 0;
  font-size: 14px;
}

/* 聊天面板 */
.chat {
  flex: 1;
  overflow: auto;
  padding: 8px;
  padding-bottom: 56px;
}

.messages {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.message {
  padding: 8px;
  border-radius: 4px;
  max-width: 80%;
  word-wrap: break-word;
  white-space: pre-wrap;
  display: block;
}

.me {
  background: #e6f7ff;
  margin-left: auto;
}

.bot {
  background: #f0f0f0;
  margin-right: auto;
}

/* 输入框容器 */
.input-box {
  display: flex;
  gap: 8px;
  padding-top: 8px;
  position: absolute;
  bottom: 16px;
  left: 16px;
  right: 16px;
}
</style>
