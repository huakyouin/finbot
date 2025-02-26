import os
import lightgbm as lgb
import torch
import torch.nn as nn
import numpy as np
import json


class BaseModel():
    _registry = {}

    @classmethod
    def register(cls, model_name):
        def decorator(subclass):
            cls._registry[model_name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def use_subclass(cls, name,):
        subclass = cls._registry.get(name)
        assert subclass, f"No subclass registered for '{name}"
        return subclass

    def train(self,):
        raise NotImplementedError
    
    def pred(self,):
        raise NotImplementedError


@BaseModel.register("bert_classifier")
class BertClassifier(nn.Module, BaseModel):
    def __init__(self, bert_backbone, num_labels=2,device_map="cuda"):
        super().__init__()
        self.device = torch.device(device_map)
        self.bert = bert_backbone.to(self.device)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels).to(self.device)
        
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
    
    def train(self, data_loader, criterion, optimizer):
        """训练模型并返回平均训练损失"""
        self.classifier.train()
        total_loss = 0.0  
        total_correct = 0  

        for batch in data_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            logits = self(input_ids, attention_mask)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(logits, dim=1)  # 取 logits 中的最大值作为预测类别
            total_correct += (predicted == labels).sum().item()  # 统计正确预测数量

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)
        return avg_loss, accuracy
    
    def eval(self, data_loader, criterion):
        """评估模型并返回平均验证损失"""
        self.classifier.eval()
        total_loss = 0.0
        total_correct = 0  

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                logits = self(input_ids, attention_mask)
                val_loss = criterion(logits, labels)

                total_loss += val_loss.item() * input_ids.size(0)
                _, predicted = torch.max(logits, dim=1)
                total_correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)

        return avg_loss, accuracy
    
    def pred(self, input_tokens):
        pred_mapper = {
            0: "NEGATIVE",
            1: "POSITIVE"
        }
        outputs = self(input_tokens['input_ids'].to(self.device),input_tokens['attention_mask'].to(self.device)).squeeze(dim=0)
        prob_ls = torch.softmax(outputs, dim=-1).cpu().detach().numpy()
        prob_ls = np.expand_dims(prob_ls, axis=0) if prob_ls.ndim == 1 else prob_ls  # 保证形状为 [num_samples, num_labels]
        pred_labels = [pred_mapper[prob.argmax()] for prob in prob_ls]
        return prob_ls, pred_labels

    def save_classifier(self, to_dir):
        os.makedirs(to_dir, exist_ok=True)
        model_path = os.path.join(to_dir, "classifier_weights.pth")
        torch.save(self.classifier.state_dict(), model_path)
        print(f"Classifier weights saved to {model_path}")

    def load_classifier(self, from_dir):
        model_path = os.path.join(from_dir, "classifier_weights.pth")
        classifer_weights = torch.load(model_path, map_location=self.device, weights_only=True)
        self.classifier.load_state_dict(classifer_weights)
        print(f"Classifier weights loaded from {model_path}")


@BaseModel.register("lgb")
class LGBModel(BaseModel):
    def __init__(self,train_params=None):
        self.train_params = train_params
        self.early_stopping_callback = lgb.early_stopping(50)
        self.verbose_eval_callback = lgb.log_evaluation(period=20)
        self.evals_result = {}
        self.evals_result_callback = lgb.record_evaluation(self.evals_result)

    def train(self, train_df, val_df, feature_cols, label_cols):
        train_set = lgb.Dataset(train_df[feature_cols].values, label=train_df[label_cols].values, free_raw_data=False)
        valid_set = lgb.Dataset(val_df[feature_cols].values, label=val_df[label_cols].values, free_raw_data=False) 
        self.model = lgb.train(
            self.train_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[self.verbose_eval_callback, self.evals_result_callback],
        )

    def pred(self, df, feature_cols):
        df['predict'] = self.model.predict(df[feature_cols]).tolist()
        return df
    
    def load(self, from_dir):
        self.model = lgb.Booster(model_file=os.path.join(from_dir,'model.bin'))

    def save(self, to_dir):
        os.makedirs(to_dir,exist_ok=True)
        with open(os.path.join(to_dir,'train_params.json'), 'w', encoding='utf-8') as f:
            json.dump(self.train_params, f, ensure_ascii=False, indent=4)
        self.model.save_model(os.path.join(to_dir,'model.bin'))