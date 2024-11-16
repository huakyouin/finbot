
import torch
import torch.nn as nn
from transformers import BertModel
from torch.utils.data import Dataset

class BertBinaryClassifier(nn.Module):
    def __init__(self, bert_path_or_name, num_labels=2):
        super(BertBinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path_or_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
        # 冻结 BERT 的所有参数
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token 的输出
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size[0], num_labels)
        # 通过分类头
        logits = self.classifier(cls_output)  # (batch_size, num_labels)
        return logits


class CLSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        初始化数据集
        Args:
            texts (list): 包含文本的列表
            labels (list): 文本对应的标签列表
            tokenizer (PreTrainedTokenizer): 分词器实例
            max_length (int): 文本最大长度，超过此长度将会截断
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取数据样本，并将文本转为 BERT 可用的输入格式
        Args:
            idx (int): 索引值
        Returns:
            dict: 包含 `input_ids`, `attention_mask`, `labels` 的字典
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 获取编码结果并移除不必要的维度
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train(model, data_loader, criterion, optimizer):
    """训练模型并返回平均训练损失"""
    model.train()  # 将模型设置为训练模式
    total_loss = 0.0  # 初始化总训练损失
    total_correct = 0  # 初始化正确预测的数量 

    for batch in data_loader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        # 前向传播和计算损失
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加当前 batch 的总损失
        total_loss += loss.item() * input_ids.size(0)

         # 计算预测正确的样本数量
        _, predicted = torch.max(logits, dim=1)  # 取 logits 中的最大值作为预测类别
        total_correct += (predicted == labels).sum().item()  # 统计正确预测数量

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)
    return avg_loss, accuracy

def eval(model, data_loader, criterion):
    """评估模型并返回平均验证损失"""
    model.eval()  # 将模型设置为评估模式
    total_loss = 0.0
    total_correct = 0  # 初始化正确预测的数量

    with torch.no_grad():  # 禁用梯度计算
        for batch in data_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            # 前向传播计算验证损失
            logits = model(input_ids, attention_mask)
            val_loss = criterion(logits, labels)

            # 累加当前 batch 的总验证损失
            total_loss += val_loss.item() * input_ids.size(0)

            # 计算预测正确的样本数量
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = total_correct / len(data_loader.dataset)

    return avg_loss, accuracy