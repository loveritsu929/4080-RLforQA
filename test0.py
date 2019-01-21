#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:28:52 2019

@author: cxing95
"""

from utility import *
import pickle
import json

#fp0 = load_file('./data/word_vocabulary','obj')
#fp1 = load_file('./data/word_embedding','obj')
#fp2 = load_file('/media/data1/hotpot/train_composite','obj')   # change data path

hpqa = load_file('/media/data1/hotpot/hotpot_train_v1.1.json', 'jsn')

#training set
#dict_keys(['fact_handles', 'sentence_source_array', 'sentence_length_array', 'sentence_symbols_array', 
#           'sentence_numbers_array', 'fact_labels', 'answer_class', 'answer_spans'])

'''
with open('./data/word_embedding', mode="rb") as file_stream:
    fp = pickle.load(file_stream)
'''
eg_num = 2
with open('./trainExamples/trainEg' + str(eg_num) +'.json', mode = 'w') as js:
    json.dump(hpqa[eg_num], js, indent = 4)



