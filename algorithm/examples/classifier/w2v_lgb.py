# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: w2v_lgb.py
@Time: 2022/7/17 0:29
"""
import math
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from datetime import datetime
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder

# import xgboost as xgb
# from nullpom import run_null_importance
warnings.filterwarnings('ignore')

train = pd.read_csv('./algorithm/data/基于用户画像的商品推荐挑战赛_复赛数据集/train.txt', header=None,
                    names=['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
test = pd.read_csv('./algorithm/data/基于用户画像的商品推荐挑战赛_复赛数据集/test.txt', header=None,
                   names=['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'make', 'model'])
data = pd.concat([train, test], ignore_index=True)

data['tagid'] = data['tagid'].apply(lambda x: eval(x))
data['time'] = data['time'].apply(lambda x: eval(x))

# 通过w2v的词向量结果取mean得到句向量
data['tagid'] = data['tagid'].apply(lambda x: eval(x))
sentences = data['tagid'].values.tolist()
for i in range(len(sentences)):
    sentences[i] = [str(x) for x in sentences[i]]  # 将每个tagid转化成str格式

emb_size = 64
model = Word2Vec(sentences, vector_size=emb_size, window=10, min_count=1, sg=0, hs=0, seed=1, epochs=10)
# model = Word2Vec.load('./w2vmodel/w2vmodel.model')
emb_matrix = []
for seq in sentences:
    vec = []
    for w in seq:
        try:
            vec.append(model.wv[w])
        except KeyError:
            continue

    if len(vec) > 0:
        emb_matrix.append(np.mean(vec, axis=0))
    else:
        emb_matrix.append([0] * emb_size)

emb_matrix = np.array(emb_matrix)
for i in range(emb_size):
    data['tag_emb_{}'.format(i)] = emb_matrix[:, i]

# 简单构建其他特征；目标编码、时间统计等此处不再做演示
data['tagidLen'] = data['tagid'].apply(lambda x: len(x))
data['age'] = data['age'].fillna(0)

# 取数据开始训练模型
cat_cols = ['age', 'province', 'make']
features = [i for i in data.columns if
            i not in ['pid', 'gender', 'label', 'tagid', 'time', 'model', 'city']]  # 尝试加入用户pid，过拟合

for i in cat_cols:
    data[i] = LabelEncoder().fit_transform(data[i])

X_train = data[~data['label'].isna()]
X_test = data[data['label'].isna()]

y = X_train['label']
KF = StratifiedKFold(n_splits=10, random_state=2021, shuffle=True)
params = {
    'objective': 'binary',
    'metric': 'binary_error',
    'learning_rate': 0.05,
    'subsample': 0.8,
    'subsample_freq': 3,
    'colsample_btree': 0.8,
    'num_iterations': 10000,
    'verbose': -1
}
oof_lgb = np.zeros(len(X_train))
predictions_lgb = np.zeros((len(X_test)))
# 特征重要性
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})

model = lgb.LGBMClassifier(num_leaves=64,
                           max_depth=10,
                           learning_rate=0.08,
                           n_estimators=1000000,
                           subsample=0.8,
                           feature_fraction=0.8,
                           reg_alpha=0.5,
                           reg_lambda=0.5,
                           random_state=2021,
                           objective='binary',
                           metric='binary_error', )
# 十折交叉验证
for fold_, (trn_idx, val_idx) in enumerate(KF.split(X_train, y)):
    print("##########第{}折############".format(fold_ + 1))

    Xtrain = X_train.iloc[trn_idx][features]
    Ytrain = y.iloc[trn_idx,]

    X_val = X_train.iloc[val_idx,][features]
    Y_val = y.iloc[val_idx,]

    model = model.fit(Xtrain,
                      Ytrain,
                      # eval_metric="f1",
                      eval_set=[(X_val, Y_val)],
                      verbose=100,
                      early_stopping_rounds=100
                      )

    feat_imp_df['imp'] += model.feature_importances_
    oof_lgb[val_idx] = model.predict_proba(X_train.iloc[val_idx][features])[:, 1]
    predictions_lgb += model.predict_proba(X_test[features])[:, 1]
predictions_lgb = predictions_lgb / 10
