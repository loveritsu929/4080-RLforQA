# -*- coding: utf-8 -*-
import time
import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.cuda as cuda

import NNetworks
import dlDataset

from bert_serving.client import BertClient

#===================== Hyper Parameters
device = torch.device('cuda:0')

batchSize = 1024

modelSaveDir = './exps'

testDataset = dlDataset.sentDataset(mode='test')
testLoader = data.DataLoader(testDataset, batch_size = batchSize, shuffle = False, num_workers = 0)

model = NNetworks.SimpleNet()
model = model.to(device) # or model = model.to(device)
lossFunction = nn.CrossEntropyLoss()

bert = BertClient(check_length=False)
#===================== testing
# TODO: kaiming init
print("Ready for testing")
for ep in range(1):
    print('Start Test Epoch {}'.format(ep))
    #t_0 = time.time()
    running_loss = 0.0
    #running_corrects = 0
    num_true = 0
    true_positive = 0
    for i, (q, sent, label) in enumerate(testLoader, 0):
        q_t = torch.as_tensor(bert.encode(list(q)))
        sent_t = torch.as_tensor(bert.encode(list(sent)))
        sample = torch.cat([q_t, sent_t], dim=1) # size: 1 * 2048
        sample = sample.to(device)
        
        label = label.to(device)
        
        # turn off gradients
        with torch.no_grad():    
            out = model(sample)
            loss = lossFunction(out, label)
        
        # statistics
        _, preds = torch.max(out,1)  # The second return value is the index location of each maximum value found (argmax).
        running_loss += loss.item() * sample.size(0)
        #running_corrects += torch.sum(preds == label.data)
        true_positive += torch.sum(preds * label.data)
        num_true += torch.sum(label.data)
        true_pos_acc = true_positive.double() / num_true.double()
        
        # batchSize samples in each iteration 
        if i % 10 == 0:
            print('Test Iteration {}: true_positive: {} Acc: {:.4f}'.format(i,true_positive, true_pos_acc))
             
    epoch_loss = running_loss / len(testDataset)
    #epoch_acc = running_corrects.double() / len(testDataset)
    print('Finish test epLoss: {:.4f} true_positie_acc: {:.4f}'.format(epoch_loss, true_positive.double() / testDataset.num_positive))
    
