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

batchSize = 256
lnR = 1e-2
maxEpoches = 20

modelSaveDir = './exps'

trainDataset = dlDataset.sentDataset(mode='train')
trainLoader = data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True, num_workers = 0)

model = NNetworks.SimpleNet()
model = model.to(device) # or model = model.to(device)
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lnR, amsgrad = True)

bert = BertClient(check_length=False)
#===================== training
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
print("Ready for trainning")
for ep in range(maxEpoches):
    print('Start Epoch {}'.format(ep))
    #t_0 = time.time()
    running_loss = 0.0
    running_corrects = 0
    for i, (q, sent, label) in enumerate(trainLoader, 0):
        q_t = torch.as_tensor(bert.encode(list(q)))
        sent_t = torch.as_tensor(bert.encode(list(sent)))
        sample = torch.cat([q_t, sent_t], dim=1) # size: 1 * 2048
        sample = sample.to(device)
        
        label = label.to(device)
        
        out = model(sample)
        loss = lossFunction(out, label)
        
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        # statistics
        _, preds = torch.max(out,1)  # The second return value is the index location of each maximum value found (argmax).
        running_loss += loss.item() * sample.size(0)
        running_corrects += torch.sum(preds == label.data)
        
        # batchSize samples in each iteration 
        if i % 200 == 0:
            print('Epoch {} Iteration {}: running_corrects: {} running loss = {}'.format(ep,i,running_corrects,running_loss))
             
    epoch_loss = running_loss / len(trainDataset)
    epoch_acc = running_corrects.double() / len(trainDataset)
    print('Finish epoch {} epLoss: {:.4f} epAcc: {:.4f}'.format(ep, epoch_loss, epoch_acc))
    
    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), os.path.join(modelSaveDir, 'exp1_best_model.mdl'))
        
print('Finish Training, Best val Acc: {:4f}'.format(best_acc))
    
