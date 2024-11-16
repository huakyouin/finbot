def custom_Dataset_of_financial_news_sentiment_classification(from_dir, to_dir):
    import os
    import pandas as pd
    train_data = pd.read_csv(f"{from_dir}/train_data.csv")
    test_data = pd.read_csv(f"{from_dir}/test_data.csv")
    get_text = lambda x: x['正文']
    get_label = lambda x: x['正负面']
    # 合并数据集
    df = pd.concat([
        train_data.assign(split='train', text=get_text, label=get_label),
        test_data.assign(split='test', text=get_text, label=get_label)
    ], ignore_index=True)[['split', 'text', 'label']]
    # 清洗
    df = df[df['text'].notna() & (df['text'] != '')]
    # 保存清洗后的数据
    os.makedirs(to_dir, exist_ok=True)
    df.to_json(os.path.join(to_dir,'Dataset-of-financial-news-sentiment-classification.jsonl'), orient='records',force_ascii=False, lines=True)


def custom_qlib_stock_dataset(from_dir, to_dir):
    import qlib
    import os
    from qlib.data import D
    # 初始化 Qlib 的数据存储
    qlib.init(provider_uri = from_dir)
    fields = ['$open', '$high', '$low', '$close', '$volume', '$amount', '$vwap']
    df = D.features(D.instruments(market='csi300'), fields, start_time='20160101', end_time='20201231', freq='day')
    df.rename(columns=lambda x: x.replace('$', ''), inplace=True)
    os.makedirs(to_dir, exist_ok=True)
    df.to_csv(os.path.join(to_dir,"csi300_stock_feats.csv"))


def custom_FinCUGE(from_dir, to_dir):
    from datasets import load_dataset, concatenate_datasets
    import os
    dataset = load_dataset(from_dir)  
    for split_name in dataset:
        dataset[split_name] = dataset[split_name].map(lambda example: {"split": split_name})
    combined_data = concatenate_datasets([dataset[split_name] for split_name in dataset])
    os.makedirs(to_dir, exist_ok=True)
    df = combined_data.to_pandas()
    df.to_json(os.path.join(to_dir,"FinCUGE.jsonl"), orient='records',force_ascii=False, lines=True)