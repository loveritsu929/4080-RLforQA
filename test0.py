#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:28:52 2019

@author: cxing95
"""
import torch
import torch.nn as nn
# init CE Loss function
criterion = nn.CrossEntropyLoss()

# sequence of length 1
output = torch.rand(1, 5)
# in this case the 1th class is our target, index of 1th class is 0
target = torch.LongTensor([0])
loss = criterion(output, target)
print('Sequence of length 1:')
print('Output:', output, 'shape:', output.shape)
print('Target:', target, 'shape:', target.shape)
print('Loss:', loss)

# sequence of length 2
output = torch.rand(2, 5)
# targets are here 1th class for the first element and 2th class for the second element
target = torch.LongTensor([0, 1])
loss = criterion(output, target)
print('\nSequence of length 2:')
print('Output:', output, 'shape:', output.shape)
print('Target:', target, 'shape:', target.shape)
print('Loss:', loss)

# sequence of length 3
output = torch.rand(2, 3, 5)
# targets here 1th class, 2th class and 2th class again for the last element of the sequence
target = torch.LongTensor([[0, 1, 1],[0,1,1]])
loss = 0
for i, item in enumerate(output):
    loss += criterion(item, target[i])
print('\nSequence of length 3:')
print('Output:', output, 'shape:', output.shape)
print('Target:', target, 'shape:', target.shape)
print('Loss:', loss)

#nums = [i for i in range(1000)]
#selected = []
#for i in nums:
#    if rd.random() <= 0.05:
#        selected.append(i)
#        
#print(len(selected))
#fp0 = load_file('./data/word_vocabulary','obj')
#fp1 = load_file('./data/word_embedding','obj')
#fp2 = load_file('/media/data1/hotpot/train_composite','obj')   # change data path

# issue: some sent in para is empty
#test = load_file('/media/data1/hotpot/hotpot_dev_fullwiki_v1.json', 'jsn')
#qtitle = 'The Hilltop (newspaper)'#'East Tennessee Natural Gas Pipeline'
#print('start')
#for q in test:
#    for title, para in q['context']:
#        if title == qtitle:
#            print(para)
#            with open('./issue.json', mode = 'w') as js:
#                json.dump(q, js, indent = 4)
#            break

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