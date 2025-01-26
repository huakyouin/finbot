import os
import lightgbm as lgb
import torch
import torch.nn as nn


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
    
    def load(self,):
        raise NotImplementedError

    def save(self,):
        raise NotImplementedError


@BaseModel.register("bert_classifier")
class BertClassifier(nn.Module, BaseModel):
    def __init__(self, bert_backbone, num_labels=2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.train()
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
        self.eval()  
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
        pred_labels = [pred_mapper[pred] for pred in outputs.argmax(dim=-1).cpu().detach().numpy()]
        return prob_ls, pred_labels
    
    def save(self, to_dir):
        os.makedirs(to_dir,exist_ok=True)
        model_path = os.path.join(to_dir,"model_weights.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

    def load(self, from_dir):
        model_path = os.path.join(from_dir,"model_weights.pth")
        weights = torch.load(model_path, map_location=self.device, weights_only=True)
        self.load_state_dict(weights)
        print(f"Model weights load from {model_path}")


@BaseModel.register("lgb")
class LGBModel(BaseModel):
    def __init__(self):
        self.model_params = dict(
            objective="mse", 
            colsample_bytree=0.8879,
            learning_rate=0.0421,
            subsample=0.8789,
            lambda_l1=2,  # 205.6999
            lambda_l2=5, # 正则超重 580.9768
            max_depth=8,
            num_leaves=210,
            num_threads=20,
            verbosity=-1,
        )
        self.early_stopping_callback = lgb.early_stopping(50)
        self.verbose_eval_callback = lgb.log_evaluation(period=20)
        self.evals_result = {}
        self.evals_result_callback = lgb.record_evaluation(self.evals_result)

    def train(self, train_df, val_df, feature_keys, label_key):
        train_set = lgb.Dataset(train_df[feature_keys].values, label=train_df[label_key].values, free_raw_data=False)
        valid_set = lgb.Dataset(val_df[feature_keys].values, label=val_df[label_key].values, free_raw_data=False) 
        self.model = lgb.train(
            self.model_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[train_set, valid_set],
            valid_names=['train', 'valid'],
            callbacks=[self.verbose_eval_callback, self.evals_result_callback],
        )

    def pred(self, df, feature_keys):
        df['predict'] = self.model.predict(df[feature_keys]).tolist()
        return df
    
    def load(self, from_dir):
        self.model = lgb.Booster(model_file=os.path.join(from_dir,'model.bin'))

    def save(self, to_dir):
        os.makedirs(to_dir,exist_ok=True)
        self.model.save_model(os.path.join(to_dir,'model.bin'), format='binary')