'''
用于在RAG中对文本进行分片的函数，不包含embedding
'''

import re
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def calculate_cosine_distances(vec1_list, vec2_list):
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
        similarity = cosine_similarity(vec1_list[i], vec2_list[i])
        # 转换为余弦距离
        distance = 1 - similarity
        distances.append(distance)
    return distances


def build_chunks(text, sentences,distances,breakpoint_percentile_threshold=95,max_len=450):
    '''
    根据距离把文本分片
    Input:
        text:       原文
        sentences:  句子列表
        distances:  跟下句的相似距离
        breakpoint_percentile_threshold：   距离分割阈值分位数（0-100）
        max_len:    最大片段长度
    '''
    if len(sentences)<=1: # 全文被分为一句话
        return [{"start_idx":sentences[0]["start_idx"], "text": text}]

    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] + [len(sentences)-1] + ["end"]

    chunks = []
    start_sent_idx = 0
    pin = 0
    last_sent_idx = indices_above_thresh[pin]
    while last_sent_idx!="end":
        combined_text = text[sentences[start_sent_idx]["start_idx"]: sentences[last_sent_idx]["start_idx"]+sentences[last_sent_idx]["length"]]
        # print(len(combined_text))
        if len(combined_text) < max_len or last_sent_idx==start_sent_idx:
            chunk = {"text":combined_text,"sentence_range":[start_sent_idx,last_sent_idx]}
            chunks.append(chunk)
            # 更新循环变量 
            start_sent_idx = last_sent_idx + 1
            pin += indices_above_thresh[pin]==last_sent_idx  # 如果当前分裂原因不是max_len，就+1
            last_sent_idx = indices_above_thresh[pin]
        else:
            last_sent_idx = start_sent_idx + np.argmax(distances[start_sent_idx:last_sent_idx])  # 不含last_sent_idx

    return chunks


def find_emb_matched_chunks_ids(q_vec_ls, chunk_vec_ls, topn=10, min_similarity=0.51,relaxation_limit=2):
    """
    寻找匹配片段

    Args:
        q_vec_ls (list): 查询片段的嵌入列表
        chunk_vec_ls (list): 上下文片段的嵌入列表
        topn (int, optional): 返回每个查询片段的最大匹配数
        min_similarity (float, optional): 最小的余弦相似度阈值，用于确定匹配
        relaxation_limit (int, optional): 如果没有匹配的片段时的松弛限制数量

    Returns:
        list: 匹配片段的索引列表，每个查询片段对应一个子列表    (n,*)二维列表，n为查询数
    """
    matched_indices = []

    for q_vec in q_vec_ls:
        temp = []

        # 计算查询片段与所有上下文片段的余弦相似度
        similarities = [cosine_similarity(q_vec, chunk_vec) for chunk_vec in chunk_vec_ls]

        # 对相似度数组进行排序并获取排序后的索引
        sorted_indices = np.argsort(similarities)[::-1]

        # 选择符合最小相似度阈值且未超过匹配数的索引
        for idx in sorted_indices:
            if similarities[idx] > min_similarity:
                temp.append(idx)
        
        # 如果召回为0则根据relaxation_limit设定松弛
        if len(temp)==0:
            temp += sorted_indices[:relaxation_limit]

        matched_indices.append(temp[:topn])

    return matched_indices