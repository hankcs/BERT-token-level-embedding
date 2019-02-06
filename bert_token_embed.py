# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-01-13 15:01
import pickle

import numpy as np
from bert_serving.client import BertClient

bc = BertClient(ip='127.0.0.1')  # ip address of the server


def embed_last_token(text):
    result = bc.encode(text, show_tokens=True)
    # print(result)
    batch = []
    for sent, tensor, tokens in zip(text, result[0], result[1]):
        valid = []
        tid = 0
        buffer = ''
        words = sent.lower().split()
        for i, t in enumerate(tokens):
            if t == '[CLS]' or t == '[SEP]':
                continue
            else:
                if t.startswith('##'):
                    t = t[2:]
                buffer += t
                if buffer == words[tid]:
                    valid.append(i)
                    buffer = ''
                    tid += 1
        # print(len(valid))
        # exit()
        if len(valid) != len(sent.split()) or tid != len(words):
            print(valid)
            print(sent.split())
            print(result[1])
        batch.append(tensor[valid, :])
    return batch


def embed_sum(text):
    result = bc.encode(text, show_tokens=True)
    # print(result)
    batch = []
    for sent, tensor, tokens in zip(text, result[0], result[1]):
        token_tensor = []
        sent_tensor = []
        tid = 0
        buffer = ''
        words = sent.lower().split()
        for i, t in enumerate(tokens):
            if t == '[CLS]' or t == '[SEP]':
                continue
            else:
                if t.startswith('##'):
                    t = t[2:]
                buffer += t
                token_tensor.append(tensor[i, :])
                if buffer == words[tid]:
                    sent_tensor.append(np.stack(token_tensor).mean(axis=0))
                    token_tensor = []
                    buffer = ''
                    tid += 1
        # print(len(valid))
        # exit()
        if tid != len(words) or len(sent_tensor) != len(words):
            print(sent.split())
            print(result[1])
        batch.append(np.stack(sent_tensor))
    return batch


def generate_bert(path, output, embed_fun=embed_sum):
    total = 0
    with open(path) as src:
        batch = []
        tensor = []
        for line in src:
            line = line.strip()
            if len(line) == 0:
                continue
            batch.append(line)
            if len(batch) and len(batch) % 100 == 0:
                tensor.extend(embed_fun(batch))
                total += len(batch)
                print(total)
                batch = []
        if len(batch):
            tensor.extend(embed_fun(batch))
            total += len(batch)
            print(total)
        with open(output, 'wb') as f:
            pickle.dump(tensor, f)


if __name__ == '__main__':
    generate_bert('input.txt', 'output.pkl', embed_fun=embed_last_token)
