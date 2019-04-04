#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:01:05 2019

@author: cxing95
"""

from bert_serving.client import BertClient
from tqdm import tqdm
import numpy as np
import csv, torch, pickle, json, random


bert = BertClient(check_length=False)
train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'

def select_neg_sent(p=0.06):
    
    return
    
def sent2vec(sent):
    emb= bert.encode([sent])
    return emb

def load_file(file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return file_stream.read().splitlines()

    elif file_type == "jsn":
        with open(file=file_path, mode="rt", encoding="utf-8") as file_stream:
            return json.load(file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="rb") as file_stream:
            return pickle.load(file_stream)

    else:
        pass

def dump_data(data_buffer, file_path, file_type):
    if file_type == "txt":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            file_stream.write("\n".join(data_buffer))

    elif file_type == "jsn":
        with open(file=file_path, mode="wt", encoding="utf-8") as file_stream:
            json.dump(obj=data_buffer, fp=file_stream)

    elif file_type == "obj":
        with open(file=file_path, mode="wb") as file_stream:
            pickle.dump(obj=data_buffer, file=file_stream)

    else:
        pass
    
def concat_q_sent(q_tensor, sent_tensor):
    assert q_tensor.dim() == 1
    assert sent_tensor.dim() == 1
    return  torch.cat((q_tensor, sent_tensor, q_tensor * sent_tensor, q_tensor - sent_tensor), dim=0)

def get_cat_emb(sent_emb):
    # element in para_emb: [question, para, label]
    ds = []
    for q,sent,label in tqdm(sent_emb):
        bert_emb = bert.encode([q, sent]) # 2*1024D np array
        bert_emb = torch.as_tensor(bert_emb)
        q_sent = concat_q_sent(bert_emb[0], bert_emb[1])
        ds.append([q_sent, label])
        
    return ds

def get_sent_emb(input_file, train=False):
    fp = load_file(input_file, 'jsn')
    ds = []
    #labels = []
    # one thrid of the training set
    for qdict in fp[:int(len(fp)/3)]:
        supports = qdict['supporting_facts'] # a list of 2-element lists of facts with form [title, sent_id]
        question = qdict['question']
        context = qdict['context'] # a list of 2-element list [title, paragraph]
        # some articles in the fullwiki dev/test sets have zero paragraphs
        if len(context) == 0:
            context = [['some random title', ['some random stuff']]]   
        for title, paragraph in context:
            for sent_index, sent in enumerate(paragraph):
                sent_sample = title + ": " + sent
                label = 1 if [title, sent_index] in supports else 0
                if label == 0 and train:
                    if random.random() <= 0.06:
                        ds.append([question, sent_sample, label])
                else:
                    ds.append([question, sent_sample, label])

    return ds

#__main()__
train_sent = get_sent_emb(train_file, train=True)
output_train_sent_cat = '/media/data1/hotpot/train_sent_cat_part.emb'
train_cat = get_cat_emb(train_sent)
dump_data(train_cat, output_train_sent_cat, 'obj')

dev_sent = get_sent_emb(dev_file, train=False)
output_dev_sent_cat = '/media/data1/hotpot/dev_sent_cat_part.emb'
dev_cat = get_cat_emb(dev_sent)
dump_data(dev_cat, output_dev_sent_cat, 'obj')