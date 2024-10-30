#!/bin/bash

# 设置脚本为严格模式，遇到错误时立即退出, 打印错误信息
set -ex

echo "开启代理, 如果没有需手动下载"
source /etc/profile.d/clash.sh
proxy_on

echo "股价数据"
wget https://github.com/chenditc/investment_data/releases/download/2024-08-09/qlib_bin.tar.gz
mkdir -p ./data/raw/qlib_data/cn_data
cd ./data/raw/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz --strip-components=1
rm -f qlib_bin.tar.gz
cd -

echo "金融新闻情感提取数据"
mkdir -p ./data/raw/Dataset-of-financial-news-sentiment-classification
cd ./data/raw/Dataset-of-financial-news-sentiment-classification
wget https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification/blob/master/train_data.csv
wget https://github.com/wwwxmu/Dataset-of-financial-news-sentiment-classification/blob/master/test_data.csv
cd -