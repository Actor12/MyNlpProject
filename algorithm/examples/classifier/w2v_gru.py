# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: w2v_gru.py
@Time: 2022/7/16 13:58
"""
# coding:utf-8
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

# 读取数据，简单处理list数据
train = pd.read_csv('./algorithm/data/基于用户画像的商品推荐挑战赛_复赛数据集/train.txt', header=None)
test = pd.read_csv('./algorithm/data/基于用户画像的商品推荐挑战赛_复赛数据集/test.txt', header=None)

train.columns = ['pid', 'label', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']
test.columns = ['pid', 'gender', 'age', 'tagid', 'time', 'province', 'city', 'model', 'make']

train['label'] = train['label'].astype(int)

data = pd.concat([train, test])
data['label'] = data['label'].fillna(-1)

data['tagid'] = data['tagid'].apply(lambda x: eval(x))
data['tagid'] = data['tagid'].apply(lambda x: [str(i) for i in x])

# 超参数
embed_size = 256  # 输出词向量的长度
MAX_NB_WORDS = 230637  # tagid中的单词出现次数，可以通过w2v训练完后打印出来，或者人为设置
MAX_SEQUENCE_LENGTH = 300  # 输入的句子最大长度
# 训练word2vec，这里可以考虑elmo，bert等预训练
w2v_model = Word2Vec(sentences=data['tagid'].tolist(), vector_size=embed_size, window=5, min_count=1, epochs=7)
# w2v_model.save("./w2vmodel/w2vmodel.model")
# w2v_model = Word2Vec.load("./w2vmodel/w2vmodel.model")

# 这里是划分训练集和测试数据
X_train = data[:train.shape[0]]['tagid']
X_test = data[train.shape[0]:]['tagid']

# 创建词典，利用了tf.keras的API，其实就是编码一下，具体可以看看API的使用方法
# 建立词汇表，从数据集中出现的字符构建索引词汇表
tokenizer = text.Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
# 原始句子转为token_id
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# 统一句子的长度，一般取数据集中句子长度区间的80%左右，即最大长度为100，那么取80
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index
# 计算一共出现了多少个单词，其实MAX_NB_WORDS我直接就用了这个数据

nb_words = len(word_index) + 1
print('Total %s word vectors.' % nb_words)
# 构建一个embedding的矩阵，之后输入到模型使用
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv.get_vector(word)
        # print(np.shape(embedding_vector))
    except KeyError:
        continue
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

y_categorical = train['label'].values


def my_model():
    embedding_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # 词嵌入（使用预训练的词向量，也可以不使用，随机初始化）
    embedder = Embedding(nb_words,
                         embed_size,
                         input_length=MAX_SEQUENCE_LENGTH,
                         weights=[embedding_matrix],
                         trainable=False  # 不要解冻，很难调
                         )
    embed = embedder(embedding_input)
    g1 = GRU(units=512, return_sequences=True)(embed)
    bn1 = BatchNormalization()(g1)
    drop1 = Dropout(0.15)(bn1)
    g2 = GRU(units=512)(drop1)
    bn2 = BatchNormalization()(g2)
    drop2 = Dropout(0.15)(bn2)
    d2 = Dense(256, activation='relu')(drop2)
    # gap1 =AveragePooling1D()(drop2)
    main_output = Dense(1, activation='sigmoid')(d2)
    model = Model(inputs=embedding_input, outputs=main_output)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=Adam(lr=0.0009), metrics=['accuracy'])
    return model


# 五折交叉验证
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=2048)
oof = np.zeros([len(train), 1])
predictions = np.zeros([len(test), 1])

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print("fold n{}".format(fold_ + 1))
    model = my_model()
    # model.load_weights('./model/DGRU74341_73508/DGRU_5.h5')
    if fold_ == 0:
        model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=3, mode='auto', min_delta=0.0001,
                                  verbose=2)  # 改监控指标为val_acc
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=2)
    bst_model_path = "./model/DGRU_{}.h5".format(fold_ + 1)
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    X_tra, X_val = X_train[trn_idx], X_train[val_idx]
    y_tra, y_val = y_categorical[trn_idx], y_categorical[val_idx]

    model.fit(X_tra, y_tra,
              validation_data=(X_val, y_val),
              epochs=128, batch_size=300, shuffle=True,
              callbacks=[early_stopping, model_checkpoint, reduce_lr])

    model.load_weights(bst_model_path)

    oof[val_idx] = model.predict(X_val)

    predictions += model.predict(X_test) / folds.n_splits
    print(predictions)
    del model
