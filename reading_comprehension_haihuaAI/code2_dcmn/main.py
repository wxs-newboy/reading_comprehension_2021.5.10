# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/8 19:55
@Auth ： 想打球
@File ：main.py
@IDE ：PyCharm
@note:
"""
from data import *
from dcmn import *

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert.optimization import warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from transformers import BertForMultipleChoice, AdamW, get_cosine_schedule_with_warmup
import torch
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
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
    'epochs': 100,
    'batch_size': 4,  # batch_size，可根据自己的显存调整
    'valid_bs': 16,
    'lr': 2e-5,  # 学习率
    'num_workers': 16,
    'accum_iter': 2,  # 梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4,  # 权重衰减，防止过拟合
    'warmup_proportion': 0.1,
    'ouput_eval_log': '../save/output_eval.txt',  # 保存评估结果
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


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def getDataLoader(features, is_training=True):
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_doc_len = torch.tensor(select_field(features, 'doc_len'), dtype=torch.long)
    all_ques_len = torch.tensor(select_field(features, 'ques_len'), dtype=torch.long)
    all_option_len = torch.tensor(select_field(features, 'option_len'), dtype=torch.long)

    if is_training:
        all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        all_label = torch.tensor([0 for _ in features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_doc_len, all_ques_len,
                               all_option_len)
    # sampler 定义取batch的方法，是一个迭代器， 每次生成一个key 用于读取dataset中的值
    train_sampler = RandomSampler(train_data)

    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=conf['batch_size'])
    return train_dataloader


def train(examples, model, optimizer):
    mid = int(len(examples) * 0.7)

    train_features = convert_examples_to_features(examples[:mid], tokenizer, conf['max_len'], True)
    dataloader = getDataLoader(train_features)

    num_train_steps = int(len(examples[:mid]) / conf['batch_size'] / conf['accum_iter'] * conf['epochs'])
    global_step = 0
    t_total = num_train_steps
    best_accuracy = 0
    for epoch in range(conf['epochs']):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(dataloader, desc="Iteration", ncols=50)):
            batch = tuple(t.to(conf['device']) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, doc_len, ques_len, option_len = batch
            loss = model(input_ids, segment_ids, input_mask, doc_len, ques_len, option_len, label_ids)
            if conf['accum_iter'] > 1:
                loss = loss / conf['accum_iter']
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            loss.backward()
            ## 自己动态改变学习率
            if (step + 1) % conf['accum_iter'] == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = conf['lr'] * warmup_linear(global_step / t_total, conf['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        # logger.info("lr = %f", lr_this_step)
        lr_pre = lr_this_step
        eval_features = convert_examples_to_features(examples[mid:], tokenizer, conf['max_len'], True)
        eval_dataloader = getDataLoader(eval_features)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
                eval_dataloader, desc="Evaluating", ncols=50):
            input_ids = input_ids.to(conf['device'])
            input_mask = input_mask.to(conf['device'])
            segment_ids = segment_ids.to(conf['device'])
            all_doc_len = all_doc_len.to(conf['device'])
            all_ques_len = all_ques_len.to(conf['device'])
            all_option_len = all_option_len.to(conf['device'])
            label_ids = label_ids.to(conf['device'])

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len,
                                      all_option_len, label_ids)
                logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        if eval_accuracy >= best_accuracy:
            logger.info("**** Saving model.... *****")
            best_accuracy = eval_accuracy
            torch.save(model.state_dict(), "../save/model.pkl")
            torch.save(optimizer.state_dict(), "../save/optimizer.pkl")

        result = {'epoch': epoch,
                  'eval_loss': eval_loss,
                  'best_accuracy': best_accuracy,
                  'eval_accuracy': eval_accuracy,
                  # 'global_step': global_step,
                  'lr_now': lr_pre,
                  'loss': tr_loss / nb_tr_steps}
        with open(conf['ouput_eval_log'], "a") as writer:
            logger.info("***** Eval results *****")
            writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
                epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\t" % (key, str(result[key])))
            writer.write("\t\n")


def test(test_examples, model):
    test_features = convert_examples_to_features(test_examples, tokenizer, conf['max_len'], is_training=False)
    dataloader = getDataLoader(test_features, is_training=False)

    predictions = []

    y_pred = []
    model.load_state_dict(torch.load('../save/model.pkl'))
    for input_ids, input_mask, segment_ids, label_ids, all_doc_len, all_ques_len, all_option_len in tqdm(
            dataloader, desc="prediction", ncols=50):
        input_ids = input_ids.to(conf['device'])
        input_mask = input_mask.to(conf['device'])
        segment_ids = segment_ids.to(conf['device'])
        all_doc_len = all_doc_len.to(conf['device'])
        all_ques_len = all_ques_len.to(conf['device'])
        all_option_len = all_option_len.to(conf['device'])
        label_ids = label_ids.to(conf['device'])

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len,
                                  all_option_len, label_ids)
            logits = model(input_ids, segment_ids, input_mask, all_doc_len, all_ques_len, all_option_len)

        output = logits.detach().cpu().numpy()

        y_pred.extend(output)

    predictions += [y_pred]
    return predictions


if __name__ == '__main__':
    init_seeds(conf['seed'])
    model = BertForMultipleChoiceWithMatch.from_pretrained(conf['model'], num_choices=4)
    model.to(conf['device'])
    optimizer = AdamW(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])  # AdamW优化器
    tokenizer = BertTokenizer.from_pretrained(conf['model'])  # 加载bert的分词器
    if os.path.exists('../save/model.pkl'):
        ## test_y全为0
        test_examples = read_swag_examples(is_training=False)
        q_id = []
        for example in test_examples:
            q_id.append(example.swag_id)

        predictions = test(test_examples, model)
    else:
        examples = read_swag_examples(is_training=True)
        train(examples, model, optimizer)

        # 交叉验证
        # folds = StratifiedKFold(n_splits=conf['fold_num'], shuffle=True, random_state=conf['seed']).split(
        #     np.arange(len(X)), y)
        # train(folds, model, optimizer)

        ## test_y全为0
        test_examples = read_swag_examples(is_training=False)
        q_id = []
        for example in test_examples:
            q_id.append(example.swag_id)
        predictions = test(test_examples, model)

    choice = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    print(predictions)
    predictions = np.mean(predictions, 0).argmax(1)
    pred = []
    print(predictions)
    for i, k in enumerate(predictions):
        pred.append(choice[k])
    result = DataFrame(
        {'q_id': q_id,
         'pred': pred}
    )
    result.to_csv('./result.csv', index=False)
