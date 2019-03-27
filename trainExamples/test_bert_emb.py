# -*- coding: utf-8 -*-
from bert_serving.client import BertClient
import numpy as np
import csv, torch

bert = BertClient(check_length=False)
train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'

def sent2vec(sent):
    emb= bert.encode([sent])
    return emb

#[a, b, a.*b, a.-b]
def concat_q_para(q_tensor, para_tensor):
    
    
    return 
    