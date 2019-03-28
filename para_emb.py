# -*- coding: utf-8 -*-
from bert_serving.client import BertClient
from tqdm import tqdm
import numpy as np
import csv, torch, pickle, json


bert = BertClient(check_length=False)
train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'

output_train_para = '/media/data1/hotpot/train_paras.emb'
output_dev_para = '/media/data1/hotpot/dev_paras.emb'

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


def get_para_emb(input_file):
    fp = load_file(input_file, 'jsn')
    ds = []
    #labels = []
    for qdict in fp:
        supports = qdict['supporting_facts'] # a list of 2-element lists of facts with form [title, sent_id]
        fact_titles = [fact[0] for fact in supports] # only the titles of supporting paragraphs         
        question = qdict['question']
        context = qdict['context'] # a list of 2-element list [title, paragraph]
        # some articles in the fullwiki dev/test sets have zero paragraphs
        if len(context) == 0:
            context = [['some random title', ['some random stuff']]]   
        for title, paragraph in context:
            label = 1 if title in fact_titles else 0
            #labels.append(label)
            para_str = ''.join(paragraph)
            ds.append([question, para_str, label])

    return ds

#[a, b, a.*b, a.-b]
def concat_q_para(q_tensor, para_tensor):
    assert q_tensor.dim() == 0
    assert para_tensor.dim() == 0
    return  torch.cat((q_tensor, para_tensor, q_tensor * para_tensor, q_tensor - para_tensor), dim=0)

def get_cat_emb(para_emb):
    # element in para_emb: [question, para, label]
    ds = []
    for q,para,label in tqdm(para_emb):
        bert_emb = bert.encode([q, para]) # 2*1024D np array
        q_para = concat_q_para(bert_emb[0], bert_emb[1])
        ds.append([q_para, label])
        
    return ds

#__main()__
train_para = get_para_emb(train_file)
#dump_data(train_para, output_train_para, 'obj')
dev_para = get_para_emb(dev_file)
#dump_data(dev_para, output_dev_para, 'obj')
output_train_cat = '/media/data1/hotpot/train_cat.emb'
output_dev_cat = '/media/data1/hotpot/dev_cat.emb'
train_cat = get_cat_emb(train_para)
dev_cat = get_cat_emb(dev_para)
dump_data(train_cat, output_train_cat, 'obj')
dump_data(dev_cat, output_dev_cat, 'obj')


    
