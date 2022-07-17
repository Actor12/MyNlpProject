# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: fasttext.py
@Time: 2022/7/16 23:44
"""

import os
import numpy as np
import fasttext
import jieba
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix, multilabel_confusion_matrix, \
    classification_report


def train_model(ipt=None, opt=None, model='', dim=100, epoch=100, lr=0.1, loss='softmax'):
    # suppress: bool, 科学记数法启用
    # True用固定点打印浮点数符号，当前精度中的数字等于零将打印为零。
    # False用科学记数法；最小数绝对值是<1e-4或比率最大绝对值> 1e3。默认值False
    np.set_printoptions(suppress=True)
    if os.path.isfile(model):
        classifier = fasttext.load_model(model)
    else:
        classifier = fasttext.train_supervised(ipt, label='"__label__', dim=dim, epoch=epoch,
                                               lr=lr, wordNgrams=2, loss=loss)
        """
          训练一个监督模型, 返回一个模型对象

          @param input:           训练数据文件路径
          @param lr:              学习率
          @param dim:             向量维度
          @param ws:              cbow模型时使用
          @param epoch:           次数
          @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
          @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
          @param minn:            构造subword时最小char个数
          @param maxn:            构造subword时最大char个数
          @param neg:             负采样
          @param wordNgrams:      n-gram个数
          @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
          @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
          @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
          @param lrUpdateRate:    学习率更新
          @param t:               负采样阈值
          @param label:           类别前缀
          @param verbose:         ??
          @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
          @return model object
        """
        classifier.save_model(opt)
    return classifier


def unified_format(data_lable, predict_lable):  # 多余
    lable_to_cate = {'萌宠': 0, '园艺': 1, '运动_户外': 2, '星座': 3, '影视': 4, '摄影': 5, '穿搭': 6, '游戏': 7, '数码_科技': 8, '家居': 9,
                     '汽车': 10,
                     '户外旅游': 11, '美食': 12, '母婴': 13, '美容妆发': 14, '二次元_动漫': 15}

    list_uni = []
    list_pre = []

    for i, y in zip(iter(data_lable), iter(predict_lable)):
        unified = [lable_to_cate[i.split('__label__')[-1]] for node_i in lable_to_cate.keys() if node_i in i]
        predict = [lable_to_cate[y[0].split('__label__')[-1]] for node_i in lable_to_cate.keys() if node_i in y[0]]
        list_pre.append(predict)
        list_uni.append(unified)

    return list_uni, list_pre


def get_stop_words(datapath):
    stopwords = pd.read_csv(datapath, index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
    stopwords = stopwords["stopword"].values
    return stopwords


def preprocess_lcut(content_line, stopwords):
    sentences = []
    for line in content_line:

        words = line.split('\t')[0]  # label_words
        try:
            # segs = jieba.lcut(words)  # 利用结巴分词进行中文分词  label_words
            segs = jieba.lcut(line)  # 利用结巴分词进行中文分词
            segs = filter(lambda x: len(x) > 1, segs)  # 去掉长度小于1的词
            segs = filter(lambda x: x not in stopwords, segs)  # 去掉停用词
            # sentences.append(label+'\t'+" ".join(segs))  # label_words
            sentences.append(" ".join(segs) + words)

        except Exception as e:
            print(line)
            continue

    return sentences


if __name__ == '__main__':
    dim = 300
    lr = 1e-3
    epoch = 1000
    # 模型存储路径
    # f'string' 相当于 format() 函数
    model = f'model/data_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.model'
    train_path = 'dataset/train_words.txt'  # 数据增强后的训练集，文章最后有提到
    val_path = 'dataset/val_label_sim.txt'
    test_path = 'dataset/test_label_sim.txt'
    # 输出原始标签与模型标签不匹配的文本
    unmatch_path = f'unmatch_classification/unmatch_classification_dim{str(dim)}_lr0{str(lr)}_iter{str(epoch)}.txt'

    # 模型训练
    classifier = train_model(ipt=train_path,
                             opt=model,
                             model=model,
                             dim=dim, epoch=epoch, lr=0.5
                             )

    print(classifier.words)  # list of words in dictionary
    print(classifier.labels)
    # # 模型测试
    result = classifier.test(val_path)
    print(result)

    data_words = [i.replace('\n', '').split('\t')[-1] for i in
                  open('./dataset/val_label_sim.txt', 'r', encoding='utf-8')]

    data_lable = [i.replace('\n', '').replace('"', '').split('\t')[0] for i in
                  open('./dataset/val_label_sim.txt', 'r', encoding='utf-8')]
    predict_lable = classifier.predict(data_words)[0]

    # 进行数据的编码
    list_uni, list_pre = unified_format(data_lable, predict_lable)

    # 降低维度
    list_pre = [i[0] for i in list_pre]
    list_uni = [i[0] for i in list_uni]

    print(recall_score(list_pre, list_uni, average='micro'))
    print(precision_score(list_uni, list_pre, average='micro'))

    '''
    average参数定义了该指标的计算方法，二分类时average参数默认是binary；多分类时，可选参数有micro、macro、weighted和samples。

    None：返回每个班级的分数。否则，这将确定对数据执行的平均类型。

    binary：仅报告由指定的类的结果pos_label。仅当targets（y_{true,pred}）是二进制时才适用。

    micro：通过计算总真阳性，假阴性和误报来全球计算指标。也就是把所有的类放在一起算（具体到precision），然后把所有类的TP加和，再除以所有类的TP和FN的加和。因此micro方法下的precision和recall都等于accuracy。

    macro：计算每个标签的指标，找出它们的未加权平均值。这不会考虑标签不平衡。也就是先分别求出每个类的precision再求其算术平均。

    weighted：计算每个标签的指标，并找到它们的平均值，按支持加权（每个标签的真实实例数）。这会改变“宏观”以解决标签不平衡问题; 它可能导致F分数不在精确度和召回之间。

    samples：计算每个实例的指标，并找出它们的平均值（仅对于不同的多标记分类有意义 accuracy_score）。
    ————————————————
    '''

    print(confusion_matrix(list_uni, list_pre))
    # print(multilabel_confusion_matrix(list_uni, list_pre))

    lable_to_cate = {'萌宠': 0, '园艺': 1, '运动_户外': 2, '星座': 3, '影视': 4, '摄影': 5, '穿搭': 6, '游戏': 7, '数码_科技': 8, '家居': 9,
                     '汽车': 10,
                     '户外旅游': 11, '美食': 12, '母婴': 13, '美容妆发': 14, '二次元_动漫': 15}

    categories = [i for i in lable_to_cate.keys()]
    print(classification_report(list_uni, list_pre, target_names=categories))

    test_demo = ['比比赞爆浆曲奇小丸子200g夹心饼干巧克力球网红小零食小吃休闲食品整箱',
                 '索讯科 GULIAN 实木床现代简约主卧双人出租房床架加宽床床经济型简易单人床 实木床40厘米高【满铺】 1.35米*2米',
                 '书架简约落地简易现代客厅置物架柜家用学生卧室储物收纳 【特惠款】53×150cm-暖白-无背板',
                 '【12期免息+碎屏险】vivo手机X80新品5G手机旗舰芯片蔡司闪充旗舰机vivox80  12GB+256GB',
                 '石砾水洗米石子胶粘石头水刷石水磨石砾石黑白灰色砾石米庭院透水路面 白色磨圆 5斤',
                 '红色高跟鞋']

    stopwordsFile = r"./data/stopwords.txt"
    stopwords = get_stop_words(stopwordsFile)
    sentence = preprocess_lcut(test_demo, stopwords)
    predict_lable = classifier.predict(sentence)
    print(predict_lable)
