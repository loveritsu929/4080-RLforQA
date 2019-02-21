#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:28:52 2019

@author: cxing95
"""

from utility import *
#import pickle
import json
import torch
import torch.nn as nn
#fp0 = load_file('./data/word_vocabulary','obj')
#fp1 = load_file('./data/word_embedding','obj')
#fp2 = load_file('/media/data1/hotpot/train_composite','obj')   # change data path

# issue: some sent in para is empty
test = load_file('/media/data1/hotpot/hotpot_dev_fullwiki_v1.json', 'jsn')
qtitle = 'The Hilltop (newspaper)'#'East Tennessee Natural Gas Pipeline'
print('start')
for q in test:
    for title, para in q['context']:
        if title == qtitle:
            print(para)
            with open('./issue.json', mode = 'w') as js:
                json.dump(q, js, indent = 4)
            break

#training set
#dict_keys(['fact_handles', 'sentence_source_array', 'sentence_length_array', 'sentence_symbols_array', 
#           'sentence_numbers_array', 'fact_labels', 'answer_class', 'answer_spans'])

'''
with open('./data/word_embedding', mode="rb") as file_stream:
    fp = pickle.load(file_stream)

eg_num = 2
with open('./trainExamples/trainEg' + str(eg_num) +'.json', mode = 'w') as js:
    json.dump(hpqa[eg_num], js, indent = 4)
'''
#out = torch.as_tensor([0.333,0.454,0.676])
#t = torch.as_tensor([1,0,1])
#for i in range(out.size()[0]):
#    if t[i] == 0:
#        print(out[i])

#def para_padding(para_list, maxLen):
#    para = para_list
#    paraLen = len(para_list)
#    diff = maxLen - paraLen
#    times = int(maxLen/paraLen)
#    if diff > 0:
#        para = (para * (times+1))[:maxLen]
#    return para
#
#a = [1,2,3]
#b = para_padding(a, 10)
#print(b)