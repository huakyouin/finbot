import re
import numpy as np
import pandas as pd
import torch

class BaseSpliter():
    _registry = {}

    @classmethod
    def register(cls, model_name):
        def decorator(subclass):
            cls._registry[model_name] = subclass
            return subclass
        return decorator
    
    @classmethod
    def use_subclass(cls, name):
        subclass = cls._registry.get(name)
        assert subclass, f"No subclass registered for '{name}"
        return subclass
    
    def split_text_to_sentences(self, text, split_chars="?!。！…？"):
        """按分割符切割文本成句子，返回包含句子及位置信息的DataFrame。"""
        if pd.isna(text): return pd.DataFrame(columns=['sentence',"start_idx","end_idx"]) ## 入参检查
        pattern = re.compile(
            rf"[^{''.join(split_chars)}]+?(\.(?!\d|\w)|[{split_chars}]|$|(?<=。)[\"”])"  # 跳过小数点+捕获字符串的结尾+捕获句号后的引号
        )
        matches = [
            {"sentence": match.group(0), "start_idx": match.start(), "end_idx": match.end()}
            for i, match in enumerate(pattern.finditer(text))
        ]
        return pd.DataFrame(matches)
    
    def add_buffered_sentences(self, sentence_df, buffer_size=1):
        "为句子DataFrame添加缓冲列，缓冲长度为buffer_size，新列名为buffered_sentence"
        sentence_df['buffered_sentence'] = sentence_df['sentence']
        for i in range(1,buffer_size+1):
            sentence_df['buffered_sentence'] = sentence_df['sentence'].shift(i).fillna('') + sentence_df['buffered_sentence']
        return sentence_df
    
    
    def cluster(sentence_df,):
        raise NotImplementedError
    
@BaseSpliter.register("doc_seq_model_spliter")    
class DocSegModelSpliter(BaseSpliter):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_input_len = model.bert.embeddings.position_embeddings.num_embeddings
    
    def get_word_prediction(self, text, word = '[EOS]'):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        word_id = self.tokenizer.encode(word)[1]  # [[CLS] id,word id,[SEP] id]
        word_indices = [i for i, token_id in enumerate(inputs['input_ids'][0]) if token_id==word_id] 
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = outputs.logits.argmax(-1)  # 获取预测的最后 token
        return preds[0, word_indices]
        

    def cluster(self, sentence_df, max_sent_num = 5):
        s = sentence_df['sentence'] + '[EOS]'     # 加上分割标识
        
        cut_idx = 0
        cut_ids = [0]
        result = []
        
        while cut_idx < len(s):
            est_len = sentence_df.iloc[cut_idx:cut_idx+max_sent_num]['end_idx'] - sentence_df.iloc[cut_idx]['start_idx']
            i = len(est_len[est_len < self.max_input_len])

            if i==0 or i==1:  ## 只有一句，不用再切了
                cut_idx += 1
            else:
                ss = "".join(s.iloc[cut_idx:cut_idx+i])
                eos_pred = self.get_word_prediction(ss, word= '[EOS]')
                first_cut_idx = next((idx for idx, pred in enumerate(eos_pred) if pred == 0 and idx > 0), len(eos_pred))
                cut_idx += first_cut_idx

            result.append({
                "chunk": ''.join(sentence_df['sentence'].iloc[cut_ids[-1]:cut_idx].tolist()),
                "start_idx": sentence_df.iloc[cut_ids[-1]]['start_idx'],
                "end_idx": sentence_df.iloc[cut_idx-1]['end_idx'],
                "sentence_count": cut_idx-cut_ids[-1]
            })
            cut_ids.append(cut_idx)

        return pd.DataFrame(result)
    
@BaseSpliter.register("cos_sim_spliter")
class CosineSimilaritySpliter(BaseSpliter):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    
    def calculate_cosine_distances(self, vec1_list, vec2_list):
        """
        计算两组向量之间的余弦距离。
        
        Args:
            vec1_list (list of arrays): 第一组向量的列表。
            vec2_list (list of arrays): 第二组向量的列表，与第一组向量对应。

        Returns:
            distances (list): 每对向量之间的余弦距离的列表。
        """
        assert len(vec1_list) == len(vec2_list), "向量列表长度不一致"
        distances = []
        n = len(vec1_list)
        for i in range(n):
            # 计算余弦相似度
            similarity = self.cosine_similarity(vec1_list[i], vec2_list[i])
            # 转换为余弦距离
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def cluster(self, sentence_df, breakpoint_percentile_threshold=80):
        sentence_df = self.add_buffered_sentences(sentence_df)
        text_vecs = self.vectorizer.encode(sentence_df['buffered_sentence'].tolist())
        cosine_distances = self.calculate_cosine_distances(text_vecs[:-1], text_vecs[1:]) + [1] # 末句终止
        threshold = np.percentile(cosine_distances, breakpoint_percentile_threshold) 

        cut_idx = 0
        cut_ids = [0]
        result = []

        for i, cos_dist in enumerate(cosine_distances):
            if cos_dist > threshold or i==len(cosine_distances)-1:
                cut_idx = i+1  # dist(v1,v2)的v2索引
                result.append({
                    "chunk": ''.join(sentence_df['sentence'].iloc[cut_ids[-1]:cut_idx].tolist()),
                    "start_idx": sentence_df.iloc[cut_ids[-1]]['start_idx'],
                    "end_idx": sentence_df.iloc[cut_idx-1]['end_idx'],
                    "sentence_count": cut_idx-cut_ids[-1]
                })
                cut_ids.append(cut_idx)
        
            
        return pd.DataFrame(result)

def init_spliter(method, model_path):
    if method == "cos_sim_spliter":
        from FlagEmbedding import FlagModel
        model = FlagModel(model_path, query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：", use_fp16=True)
        spliter = BaseSpliter.use_subclass("cos_sim_spliter")(model)

    elif method == "doc_seq_model_spliter":
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        spliter = BaseSpliter.use_subclass("doc_seq_model_spliter")(model, tokenizer)
    return spliter

if __name__ == "__main__":
    spliter = BaseSpliter()
    text = "你好！!\n今天的天气怎么样？我们一起去散步吧。hhhhh, 不行！"
    df = spliter.split_text_to_sentences(text)
    print(df)
