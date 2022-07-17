# -*- coding = utf-8 -*-

"""
@Author: wufei
@File: bert_classifier.py
@Time: 2022/7/16 23:26
"""
import pandas as pd
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import BertPreTrainedModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, BertModel,BertConfig,BertTokenizer, AutoModel

''
train_df = pd.read_excel('./algorithm/data/xunfei_disease_classifier/data_train.xlsx')
test_df = pd.read_excel('./algorithm/data/xunfei_disease_classifier/data_test.xlsx')

tokenizer = AutoTokenizer.from_pretrained(r"F://数据集和模型//hf//chinese_roberta_wwm_ext//")
config = AutoConfig.from_pretrained(r"F://数据集和模型//hf//chinese_roberta_wwm_ext//")


class XunFeiModel(nn.Module):
    def __init__(self, num_labels_i, num_labels_j):
        super(XunFeiModel, self).__init__()

        # 加载模型
        self.model = model = AutoModel.from_pretrained("F://数据集和模型//hf//chinese_roberta_wwm_ext//")
        self.dropout = nn.Dropout(0.1)
        self.classifier_i = nn.Linear(768, num_labels_i)
        self.classifier_j = nn.Linear(768, num_labels_j)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])  # outputs[0]=last hidden state

        logits_i = self.classifier_i(sequence_output[:, 0, :].view(-1, 768))
        logits_j = self.classifier_j(sequence_output[:, 0, :].view(-1, 768))

        return logits_i, logits_j

train_text = train_df['diseaseName'] + ' ' + train_df['conditionDesc'] + ' ' + train_df['title'] + ' ' + train_df['hopeHelp']
test_text = test_df['diseaseName'] + ' ' + test_df['conditionDesc'] + ' ' + test_df['title'] + ' ' + test_df['hopeHelp']
train_text = train_text.fillna('')
test_text = test_text.fillna('')

#拼接文本与编码
train_encoding = tokenizer(train_text.tolist()[:-1000], truncation=True, padding=True, max_length=200)
val_encoding = tokenizer(train_text.tolist()[-1000:], truncation=True, padding=True, max_length=200)
test_encoding = tokenizer(test_text.tolist(), truncation=True, padding=True, max_length=200)


# 数据集读取
class XunFeiDataset(Dataset):
    def __init__(self, encodings, label_i, label_j):
        self.encodings = encodings
        self.label_i = label_i
        self.label_j = label_j

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['label_i'] = torch.tensor(int(self.label_i[idx]))
        item['label_j'] = torch.tensor(int(self.label_j[idx]))
        return item

    def __len__(self):
        return len(self.label_i)

train_dataset = XunFeiDataset(train_encoding,
                              train_df['label_i'].values[:-1000],
                              train_df['label_j'].values[:-1000])
val_dataset = XunFeiDataset(val_encoding,
                            train_df['label_i'].values[-1000:],
                            train_df['label_j'].values[-1000:])
test_dataset = XunFeiDataset(test_encoding, [0] * len(test_df), [0] * len(test_df))

# 单个读取到批量读取
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)# 在windows中，numworkers失效！num_workers=4,pin_memory=True
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = XunFeiModel(20, 61)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
model = model.to(device)



loss_fn = CrossEntropyLoss()
#optim = AdamW(model.parameters(), lr=5e-5)
new_lr = ['classifier_i','classifier_j']

# 分层学习率设置
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if 'embeddings' in n]},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in new_lr)], "lr" : 5e-4}
     ]
#bert推荐使用AdamW
optim = AdamW(optimizer_grouped_parameters,lr=5e-5)


def train():
    model.train()
    total_train_loss = 0
    iter_num = 0
    total_iter = len(train_loader)
    for batch in train_loader:
        # 正向传播
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        label_i = batch['label_i'].to(device)
        label_j = batch['label_j'].to(device)

        pred_i, pred_j = model(
            input_ids,
            attention_mask
        )

        # 疾病方向标签-1时，不记录损失
        valid = label_j != -1
        loss = loss_fn(pred_i, label_i) + loss_fn(pred_j[valid], label_j[valid])
        # 容易导致两个任务的loss不均衡，后面部分的样本较少，为前面的2/3,可以加系数。

        # 反向梯度信息
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 参数更新
        optim.step()

        iter_num += 1

        if (iter_num % 100 == 0):
            print("iter_num: %d, loss: %.4f, %.2f%% %.4f / %.4f" % (
                iter_num, loss.item(), iter_num / total_iter * 100,
                (pred_i.argmax(1) == label_i).float().data.cpu().numpy().mean(),
                (pred_j[valid].argmax(1) == label_j[valid]).float().data.cpu().numpy().mean()
            ))


def validation():
    model.eval()
    label_i_acc, label_j_acc = 0, 0
    for batch in val_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_i = batch['label_i'].to(device)
            label_j = batch['label_j'].to(device)

            pred_i, pred_j = model(
                input_ids,
                attention_mask
            )

            valid = label_j != -1
            label_i_acc += (pred_i.argmax(1) == label_i).float().sum().item()
            label_j_acc += (pred_j[valid].argmax(1) == label_j[valid]).float().sum().item()

    label_i_acc = label_i_acc / len(val_dataloader.dataset)
    label_j_acc = label_j_acc / len(val_dataloader.dataset)

    print("-------------------------------")
    print("Accuracy: %.4f / %.4f" % (label_i_acc, label_j_acc))
    print("-------------------------------")


def prediction():
    model.eval()
    test_label_i = []
    test_label_j = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label_i = batch['label_i'].to(device)
            label_j = batch['label_j'].to(device)

            pred_i, pred_j = model(input_ids, attention_mask)
            test_label_i += list(pred_i.argmax(1).data.cpu().numpy())
            test_label_j += list(pred_j.argmax(1).data.cpu().numpy())
    return test_label_i, test_label_j


for epoch in range(2):
    train()
    validation()
    # TODO
    # 添加早停、学习了下降、保存最优模型策略、自定义层高学习率
