# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/8 19:15
@Auth ： 想打球
@File ：data.py
@IDE ：PyCharm
@note:
"""
import json
from torch.utils.data import Dataset,DataLoader


def read_data():
    train_path = '../data/train.json'
    train_data = []
    label_data = []
    answer = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    with open(train_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, item in enumerate(data):
            content = item['Content']
            questions = item['Questions']
            for line in questions:
                c_question_answer = []
                question = line['Question']
                for choice in line['Choices']:
                    c_question_answer.append((content,question,choice[2:-2]))
                train_data.append(c_question_answer)
                label_data.append(answer[line['Answer']])
    return train_data, label_data


def read_valid():
    valid_path = '../data/validation.json'
    valid_data = []
    label_data = []
    q_id = []
    with open(valid_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i, item in enumerate(data):
            content = item['Content']
            questions = item['Questions']
            for line in questions:
                c_question_answer = []
                question = line['Question']
                q_id.append(line['Q_id'])
                for choice in line['Choices']:
                    c_question_answer.append((content, question, choice[2:-2]))
                valid_data.append(c_question_answer)
                label_data.append(0)
    return valid_data,label_data,q_id






