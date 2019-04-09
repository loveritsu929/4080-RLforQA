#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:43:40 2019

@author: cxing95
"""

import csv, torch, sys, copy, os
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from bert_serving.client import BertClient
from para_emb import get_para_emb, concat_q_para, get_cat_emb

bert = BertClient(ip='130.63.94.249', check_length=False)
train_file = '/media/data1/hotpot/hotpot_train_v1.1.json'
dev_file = '/media/data1/hotpot/hotpot_dev_fullwiki_v1.json'
#hyperparameters
lnR = 0.001
numEpoch = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CatDataset(data.Dataset):
    ''' a dataset for my sentence embedding
    '''
    
    def __init__(self, file):
        self.dataset = get_para_emb(file)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        q, para, label = self.dataset[index]
        bert_emb = torch.as_tensor(bert.encode([q, para]))
        q_para = concat_q_para(bert_emb[0], bert_emb[1])
        return q_para, label
    
class MyNN(nn.Module):
    
    def __init__(self):
        super(MyNN, self).__init__()
        
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 2)
        
        self.dropout = nn.Dropout(p = 0.5)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        h = self.fc1(x)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fc2(h)
        h = self.relu(h)
        
        h = self.fc3(h)
        h = self.relu(h)
        
        return h

class MyCNN(nn.Module):
    
    def __init__(self):
        super(MyCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=128, stride=128)
        #self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=4)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        
        self.dropout = nn.Dropout(p = 0.2)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        
        h = self.conv1(x)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.fc2(h)
        h = self.relu(h)
        
        h = self.fc3(h)
        h = self.relu(h)
        
        return h
datasets = {}
train_ds = CatDataset(train_file)
dev_ds = CatDataset(dev_file)
datasets['train'] = train_ds
datasets['dev'] = dev_ds
dataloaders = {}
dataloaders['train'] = data.DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=0)
dataloaders['dev'] = data.DataLoader(dev_ds, batch_size=1024, shuffle=False, num_workers=0)

model = MyNN()
#model = MyCNN()
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lnR, amsgrad = True)

#train and validata
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
best_tp = 0.0
best_f1 = 0.0
loss_dict = {}
loss_dict['train'] = []
loss_dict['dev'] = []
acc_dict = {}
acc_dict['train'] = []
acc_dict['dev'] = []
tp_dict = {}
tp_dict['train'] = []
tp_dict['dev'] = []
fp_dict = {}
fp_dict['train'] = []
fp_dict['dev'] = []

print('NN structure: ')
print(model)
for epoch in range(numEpoch):
    print('Epoch {}/{}'.format(epoch+1, numEpoch))
    print('-' * 20)
    
    for phase in ['train','dev']:
        if phase  == 'train':
            model.train()
        else: 
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0
        running_pos = 0
        running_tp = 0
        running_fp = 0
        running_preds = 0
        labels = []
        outputs = []
        #[question, para_str, label]
        for i, (sample,label) in enumerate(dataloaders[phase]):
            #move to gpu
            sample = sample.to(device)
            #if model.conv1 != None:
            #    sample.unsqueeze_(dim=1)
            label = label.to(device)
            
            #clear gradients
            optimizer.zero_grad()
            
            #forward
            with torch.set_grad_enabled(phase == 'train'):
                output = model(sample)
                _, preds = torch.max(output, 1)
                loss = loss_func(output, label)
                
                #backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * sample.size(0)
            running_corrects += torch.sum(preds == label.data)
            running_pos += torch.sum(label.data)
            running_tp += torch.sum(label*preds)
            running_preds += torch.sum(preds)
            running_fp += np.sum(np.logical_and(preds.cpu().numpy(), (label==0).cpu().numpy()))
            labels.append(label)
            outputs.append(preds)
            if i%10==0:     
            	print('Epoch {} Iteration {}: running_corrects: {} running loss = {:4f}'.format(epoch+1,i,running_corrects,running_loss))
        
            
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets[phase])
        epoch_tp = running_tp.double() / running_pos.double()
        epoch_fp = running_fp / running_preds.double()
        loss_dict[phase].append(epoch_loss)
        acc_dict[phase].append(epoch_acc)
        tp_dict[phase].append(epoch_tp)
        fp_dict[phase].append(epoch_fp)
        f1 = f1_score(torch.cat(labels), torch.cat(outputs))
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print('{} Loss: {:.4f} TP%: {:.4f} FP%: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_tp, epoch_fp,epoch_acc, f1))
        print(' ')
        
        #save the best model
        if phase == 'dev':
            if epoch_tp > best_tp:
                best_tp = epoch_tp
            if f1 > best_f1:
                best_f1 = f1
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            #best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model.state_dict(), os.path.join('./my-models', 'bestmodel_{}.mdl'.format(file_abbr)))
print('Finish Training, Best val TP%:{:4f} Best Acc:{:.4f} Best F1:{:.4f}'.format(best_tp, best_acc,best_f1))

#plot
x = list(range(1, numEpoch+1))
plt.figure("Q_Para_Cat training")
plt.subplot(2, 3, 1).set_title('train loss')
plt.plot(x, loss_dict['train'])
plt.subplot(2, 3, 2).set_title('train TP%')
plt.plot(x, tp_dict['train'], color = 'red',marker='+', linestyle='dashed')
plt.subplot(2, 3, 3).set_title('train FP%')
plt.plot(x, fp_dict['train'], color = 'black',marker='+', linestyle='dashed')


plt.subplot(2, 3, 4).set_title('dev loss')
plt.plot(x, loss_dict['dev'])
plt.subplot(2, 3, 5).set_title('dev TP%')
plt.plot(x, tp_dict['dev'], color = 'red',marker='+', linestyle='dashed')
plt.subplot(2, 3, 6).set_title('dev FP%')
plt.plot(x, fp_dict['dev'], color = 'black',marker='+', linestyle='dashed')
plt.show()