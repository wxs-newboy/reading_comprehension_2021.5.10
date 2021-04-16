# -*- coding: utf-8 -*-
"""
@Time ： 2021/4/8 19:15
@Auth ： 想打球
@File ：data.py
@IDE ：PyCharm
@note:
"""
import json
import logging


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def read_data(path='../data/train.json'):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        answer = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        for i, item in enumerate(data_all):
            content = item['Content']
            questions = item['Questions']
            for line in questions:
                article.append(content)
                question.append(line['Question'])
                q_id.append(line['Q_id'])
                if len(line['Choices']) < 4:  # 如果选项不满四个，就补“不知道”
                    for _ in range(4 - len(line['Choices'])):
                        line['Choices'].append('D．不知道')
                ct1.append(line['Choices'][0])
                ct2.append(line['Choices'][1])
                ct3.append(line['Choices'][2])
                ct4.append(line['Choices'][3])
                y.append(answer[line['Answer']])
        return article, question, ct1, ct2, ct3, ct4, q_id, y


def read_valid(path='../data/validation.json'):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        for i, item in enumerate(data_all):
            content = item['Content']
            questions = item['Questions']
            for line in questions:
                article.append(content)
                question.append(line['Question'])
                q_id.append(line['Q_id'])
                if len(line['Choices']) < 4:  # 如果选项不满四个，就补“不知道”
                    for _ in range(4 - len(line['Choices'])):
                        line['Choices'].append('D．不知道')
                ct1.append(line['Choices'][0])
                ct2.append(line['Choices'][1])
                ct3.append(line['Choices'][2])
                ct4.append(line['Choices'][3])
                y.append(0)
        return article, question, ct1, ct2, ct3, ct4, q_id, y


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 label=None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


def read_swag_examples(is_training):
    if is_training:
        article, question, ct1, ct2, ct3, ct4, q_id, y = read_data()

    else:
        article, question, ct1, ct2, ct3, ct4, q_id, y = read_valid()


    examples = [
        SwagExample(
            swag_id=s8,
            context_sentence=s1,
            start_ending=s2,  # in the swag dataset, the
            # common beginning of each
            # choice is stored in "sent2".
            ending_0=s3,
            ending_1=s4,
            ending_2=s5,
            ending_3=s6,
            label=s7 if is_training else None
        ) for i, (s1, s2, s3, s4, s5, s6, s7, s8), in enumerate(zip(article, question, ct1, ct2, ct3, ct4, y, q_id))
    ]

    return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    pop_label = True
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(1)
        else:
            tokens_b.pop(1)

class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len in choices_features
        ]
        self.label = label

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    ool = 0
    for example_index, example in enumerate(examples):
        ## context
        context_tokens = tokenizer.tokenize(example.context_sentence)
        ## question
        start_ending_tokens = tokenizer.tokenize(example.start_ending)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens[:]  # + start_ending_tokens

            ending_token = tokenizer.tokenize(ending)
            option_len = len(ending_token)
            ques_len = len(start_ending_tokens)

            ending_tokens = start_ending_tokens + ending_token

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            doc_len = len(context_tokens_choice)
            if len(ending_tokens) + len(context_tokens_choice) >= max_seq_length - 3:
                ques_len = len(ending_tokens) - option_len

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            # assert (doc_len + ques_len + option_len) <= max_seq_length
            if (doc_len + ques_len + option_len) > max_seq_length:
                print(doc_len, ques_len, option_len, len(context_tokens_choice), len(ending_tokens))
                assert (doc_len + ques_len + option_len) <= max_seq_length
            choices_features.append((tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len))

        label = example.label
        if example_index < 5:
            logger.info("*** Example ***")
            logger.info(f"swag_id: {example.swag_id}")
            for choice_idx, (tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len) in enumerate(choices_features):
                logger.info(f"choice: {choice_idx}")
                logger.info(f"tokens: {' '.join(tokens)}")
                logger.info(f"input_ids: {' '.join(map(str, input_ids))}")
                logger.info(f"input_mask: {' '.join(map(str, input_mask))}")
                logger.info(f"segment_ids: {' '.join(map(str, segment_ids))}")
            if is_training:
                logger.info(f"label: {label}")

        features.append(
            InputFeatures(
                example_id=example.swag_id,
                choices_features=choices_features,
                label=label
            )
        )

    return features
