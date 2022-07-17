# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: tfidf_classifier.py
@Time: 2022/7/16 23:23
"""

"""
tfidf 非词向量的方式
tfidf+lr 完成文本分类任务
"""
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_excel('./data/data_train.xlsx')
test_df = pd.read_excel('./data/data_test.xlsx')

# 拼接所有文本
train_text = train_df['diseaseName'] + ' ' + train_df['conditionDesc'] + ' ' + train_df['title'] + ' ' + train_df[
    'hopeHelp']
test_text = test_df['diseaseName'] + ' ' + test_df['conditionDesc'] + ' ' + test_df['title'] + ' ' + test_df['hopeHelp']
train_text = train_text.fillna('')
test_text = test_text.fillna('')

# 使用jieba分词进行分词
train_text = [' '.join(jieba.cut(x)) for x in train_text]
test_text = [' '.join(jieba.cut(x)) for x in test_text]

# 提取TFIDF特征
tfidf = TfidfVectorizer().fit(train_text)
train_tfidf = tfidf.fit_transform(train_text)
test_tfidf = tfidf.transform(test_text)

# 训练两个逻辑回归做分类
clf_i = LogisticRegression()
clf_i.fit(train_tfidf, train_df['label_i'])
