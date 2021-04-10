# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/8 19:55
@Auth ： 想打球
@File ：main.py
@IDE ：PyCharm
@note:
"""
from data import *

from sklearn.model_selection import train_test_split, StratifiedKFold
from transformers import BertTokenizer, BertForMultipleChoice, AdamW, get_cosine_schedule_with_warmup
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
from pandas.core.frame import DataFrame

conf = {  # 训练的参数配置
    'fold_num': 5,  # 五折交叉验证
    'seed': 44,
    # 'model': '../../../archive/hfl/chinese_wwm_ext_L-12_H-768_A-12',  # 预训练模型
    'model': '../../../corpus/hfl/chinese_wwm_ext_L-12_H-768_A-12',  # 预训练模型
    'max_len': 256,  # 文本截断的最大长度
    'epochs': 8,
    'train_bs': 16,  # batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'lr': 2e-5,  # 学习率
    'num_workers': 16,
    'accum_iter': 2,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
}
conf['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def init_seeds(seed):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(
        seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(
        seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ## data [(content,question,choice),(),...]
        content = []
        question = []
        choice = []
        for item in self.data[idx]:
            content.append(item[0])
            question.append(item[1])
            choice.append(item[2])
        if len(choice) < 4:
            for i in range(4 - len(choice)):
                content.append(content[len(choice) - 1])
                question.append(question[len(choice) - 1])
                choice.append('不知道')
        question_choice = [q + ' ' + c for q, c in zip(question, choice)]

        return content, question_choice, self.label[idx]


def collate_fn(data):  # 将文章问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    input_ids, attention_mask, token_type_ids = [], [], []

    ##由MyDataset的__getitem__可以知道data中的数据内容：
    ## data  : (batch_size,2)  注:2中包含(content,label)
    for x in data:
        text = tokenizer(x[1], text_pair=x[0], padding='max_length', truncation=True,
                         max_length=conf['max_len'],
                         return_tensors='pt')
        input_ids.append(text['input_ids'].tolist())
        attention_mask.append(text['attention_mask'].tolist())
        token_type_ids.append(text['token_type_ids'].tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    label = torch.tensor([x[2] for x in data])
    return input_ids, attention_mask, token_type_ids, label


class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, train_loader, optimizer, scheduler, criterion, scaler):  # 训练一个epoch
    model.train()

    losses = AverageMeter()
    accs = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True, ncols=50)

    for step, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
        input_ids, attention_mask, token_type_ids, y = input_ids.to(conf['device']), attention_mask.to(
            conf['device']), token_type_ids.to(conf['device']), y.to(conf['device']).long()

        with autocast():  # 使用半精度训练
            output = model(input_ids, attention_mask, token_type_ids).logits

            loss = criterion(output, y) / conf['accum_iter']
            scaler.scale(loss).backward()

            if ((step + 1) % conf['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

        acc = (output.argmax(1) == y).sum().item() / y.size(0)

        losses.update(loss.item() * conf['accum_iter'], y.size(0))
        accs.update(acc, y.size(0))

        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def test_model(model, val_loader, criterion):  # 验证
    model.eval()

    losses = AverageMeter()
    accs = AverageMeter()

    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True, ncols=50)
        for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
            input_ids, attention_mask, token_type_ids, y = input_ids.to(conf['device']), attention_mask.to(
                conf['device']), token_type_ids.to(conf['device']), y.to(conf['device']).long()

            output = model(input_ids, attention_mask, token_type_ids).logits

            y_truth.extend(y.cpu().numpy())
            y_pred.extend(output.argmax(1).cpu().numpy())

            loss = criterion(output, y)

            acc = (output.argmax(1) == y).sum().item() / y.size(0)

            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))

            tk.set_postfix(loss=losses.avg, acc=accs.avg)

    return losses.avg, accs.avg


def train(folds, model, optimizer):
    cv = []  # 保存每折的最佳准确率

    for fold, (trn_idx, val_idx) in enumerate(folds):

        train_x = np.array(X)[trn_idx]
        train_y = np.array(y)[trn_idx]
        val_x = np.array(X)[val_idx]
        val_y = np.array(y)[val_idx]

        train_set = MyDataset(train_x, train_y)
        val_set = MyDataset(val_x, val_y)

        ## num_workers:这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
        train_loader = DataLoader(train_set, batch_size=conf['train_bs'], collate_fn=collate_fn, shuffle=True,
                                  num_workers=conf['num_workers'])
        val_loader = DataLoader(val_set, batch_size=conf['valid_bs'], collate_fn=collate_fn, shuffle=False,
                                num_workers=conf['num_workers'])

        best_acc = 0

        # model = BertForMultipleChoice.from_pretrained(conf['model']).to(conf['device'])  # 模型

        scaler = GradScaler()
        # optimizer = AdamW(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])  # AdamW优化器
        criterion = nn.CrossEntropyLoss()

        # warmup 需要在训练最初使用较小的学习率来启动，并很快切换到大学习率而后进行常见的 decay
        # get_cosine_schedule_with_warmup策略，学习率先warmup一个epoch，然后cos式下降
        scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_loader) // conf['accum_iter'],
                                                    conf['epochs'] * len(train_loader) // conf['accum_iter'])

        for epoch in range(conf['epochs']):

            print('epoch:', epoch)

            train_loss, train_acc = train_model(model, train_loader, optimizer, scheduler, criterion, scaler)
            val_loss, val_acc = test_model(model, val_loader, criterion)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), '../save/{}_fold_{}.pt'.format(conf['model'].split('/')[-1], fold))

        cv.append(best_acc)


def test(test_x, test_y):
    test_set = MyDataset(test_x, test_y)
    test_loader = DataLoader(test_set, batch_size=conf['valid_bs'], collate_fn=collate_fn, shuffle=False,
                             num_workers=conf['num_workers'])

    model = BertForMultipleChoice.from_pretrained(conf['model']).to(conf['device'])

    predictions = []

    for fold in [0, 1, 2, 3, 4]:  # 把训练后的五个模型挨个进行预测
        y_pred = []
        model.load_state_dict(torch.load('../save/{}_fold_{}.pt'.format(conf['model'].split('/')[-1], fold)))

        with torch.no_grad():
            tk = tqdm(test_loader, total=len(test_loader), position=0, leave=True, ncols=50)
            for idx, (input_ids, attention_mask, token_type_ids, y) in enumerate(tk):
                input_ids, attention_mask, token_type_ids, y = input_ids.to(conf['device']), attention_mask.to(
                    conf['device']), token_type_ids.to(conf['device']), y.to(conf['device']).long()

                output = model(input_ids, attention_mask, token_type_ids).logits.cpu().numpy()

                y_pred.extend(output)

        predictions += [y_pred]
    return predictions


if __name__ == '__main__':
    init_seeds(conf['seed'])
    model = BertForMultipleChoice.from_pretrained(conf['model']).to(conf['device'])  # 模型
    optimizer = AdamW(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])  # AdamW优化器
    if os.path.exists("../save/chinese_wwm_ext_L-12_H-768_A-12_fold_0.pt"):
        ## test_y全为0
        test_x, test_y, q_id = read_valid()
        predictions = test(test_x, test_y)
    else:
        X, y = read_data()
        # train_X, train_y, test_X, test_y = train_test_split(X, y, test_size=0.3, random_state=44)

        tokenizer = BertTokenizer.from_pretrained(conf['model'])  # 加载bert的分词器
        # 交叉验证
        folds = StratifiedKFold(n_splits=conf['fold_num'], shuffle=True, random_state=conf['seed']).split(np.arange(len(X)), y)
        train(folds, model, optimizer)

        ## test_y全为0
        test_x, test_y, q_id = read_valid()
        predictions = test(test_x, test_y)

    choice = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    predictions = np.mean(predictions, 0).argmax(1)  # 将结果按五折进行平均，然后argmax得到label
    for i, k in enumerate(predictions):
        predictions[i] = choice[k]
    result = DataFrame(
        {'q_id': q_id,
         'pred': predictions}
    )
    result.to_csv('./result.csv', index=False)
