# LiHua-World 数据集

![LiHuaWorld](https://files.mdnice.com/user/87760/39923168-2267-4caf-b715-7f28764549de.jpg)

LiHua-World 是一个专门为本地 RAG (检索增强生成)场景设计的数据集。该数据集包含了一个名为 LiHua 的虚拟用户一年内的聊天记录。

## 数据集特点

- 包含三种类型的问题:
  - 单跳问题 (Single-hop)
  - 多跳问题 (Multi-hop) 
  - 总结性问题 (Summary)
- 每个问题都配有人工标注的答案和支持文档
- 聊天记录涵盖了日常生活的多个方面,包括:
  - 社交互动
  - 健身训练
  - 娱乐活动
  - 生活事务
  - ...

## 数据集结构

数据集主要包含以下部分:

### 1. 原始聊天记录 (./data)
- 按时间顺序组织的聊天消息
- 每条消息包含:
  - 时间戳
  - 发送者
  - 消息内容
  - 消息类型
为了方便组织，每个文件夹包含的是一周的聊天记录。

### 2. 问答数据 (/qa)
- query_set.csv: 包含问题、标准答案和证据
- query_set.json: CSV文件的JSON格式版本

### 3. 元数据
- 用户信息
- 时间范围: 2026年1月至12月
- 对话参与者列表

## 使用说明

Step1. 在./data文件夹下，解压LiHuaWorld.zip，获取原始聊天记录。

Step2. 使用./data下的所有聊天记录作为知识库。

Step3. 使用./qa下的query_set.csv或是query_set.json作为问题，进行RAG测试。