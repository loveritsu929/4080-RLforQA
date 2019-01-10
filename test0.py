#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:28:52 2019

@author: cxing95
"""

from utility import *
import pickle

#fp0 = load_file('./data/word_vocabulary','obj')
#fp1 = load_file('./data/word_embedding','obj')
#fp2 = load_file('./data/train_composite','obj')

# training set
#dict_keys(['fact_handles', 'sentence_source_array', 'sentence_length_array', 'sentence_symbols_array', 
#           'sentence_numbers_array', 'fact_labels', 'answer_class', 'answer_spans'])

with open('./data/word_embedding', mode="rb") as file_stream:
    fp = pickle.load(file_stream)