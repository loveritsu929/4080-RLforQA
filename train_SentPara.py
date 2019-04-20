# -*- coding: utf-8 -*-
import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.cuda as cuda
import numpy as np
import sklearn
import NNetworks
import dlDataset
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pad_sequence
from bert_serving.client import BertClient
bert = BertClient(ip='130.63.94.250', check_length=False) # on server: voice

#===================== Hyper Parameters
device = torch.device('cuda:0')

batchSize = 32
lnR = 1e-3
numEpoch = 20
currentEpoch = 0

modelSaveDir = './exps/SentPara'
model_weight = ' '
if not os.path.exists(modelSaveDir):
    os.makedirs(modelSaveDir)
model = NNetworks.SentParaNN()
if os.path.isfile(os.path.join(modelSaveDir, model_weight)):
	print('Continue training: ', model_weight)
	model.load_state_dict(torch.load(os.path.join(modelSaveDir, model_weight)))
	currentEpoch = int(model_weight.split('_')[0][2:]) + 1
	print('Start from epoch {}'.format(currentEpoch))
model = model.to(device)

def my_collate(batch):
    #batch: a list of items
    #item: (sents, para, sent_labels, para_label)
    #q = [item[0] for item in batch]
    sents = [item[0] for item in batch]
    para = [item[1] for item in batch]
    
    sl = []
    sent_labels = [item[2] for item in batch]
    for i, sent_l in enumerate(sent_labels):
        sl.append(torch.LongTensor(sent_l))
    sent_labels = pad_sequence(sl, batch_first=True)
    para_label = torch.LongTensor([item[3] for item in batch])
    
    s_t = []
    for i, sents_list in enumerate(sents):
        #sents_list = list(filter(lambda x: x != '' and x != ' ' and x != '  ', sents_list))
        s_t.append(torch.as_tensor(bert.encode(sents_list)))
    s_t = pad_sequence(s_t, batch_first=True)
    p_t = torch.as_tensor(bert.encode(para))
    
    
    return [s_t, p_t, sent_labels, para_label]

datasets = {}
dataloaders = {}
trainDataset = dlDataset.SentParaDataset(mode='train')
trainLoader = data.DataLoader(trainDataset, batch_size = batchSize, collate_fn=my_collate, shuffle = True, num_workers = 0)
devDataset = dlDataset.SentParaDataset(mode='test')
devLoader = data.DataLoader(devDataset, batch_size = batchSize, collate_fn=my_collate, shuffle = False, num_workers = 0)
datasets['train'] = trainDataset
datasets['dev'] = devDataset
dataloaders['train'] = trainLoader
dataloaders['dev'] = devLoader

optimizer = optim.Adam(model.parameters(), lr = lnR, amsgrad = True)  
loss_func = nn.CrossEntropyLoss()
#debugging
#trainiter = iter(trainLoader)
#s,p,sl,pl = trainiter.next()
#model = model.to('cpu')
#sout, pout = model(s,p)
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
best_f1 = 0.0

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

        labels = []
        outputs = []
        #[question, para_str, label]
        for i, (sents, para, sents_label, para_label) in enumerate(dataloaders[phase]):
            #move to gpu
            sents = sents.to(device)
            para = para.to(device)
            sents_label = sents_label.to(device)
            para_label = para_label.to(device)
            
            #clear gradients
            optimizer.zero_grad()
            
            #forward
            with torch.set_grad_enabled(phase == 'train'):
                s_out, p_out = model(sents, para)
                _, s_preds = torch.max(s_out, dim=2)
                loss = 0
                for j, item in enumerate(s_out): # batch*seq*2
                    loss += loss_func(item, sents_label[j])
                loss += loss_func(p_out, para_label)
                
                #backprop
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
            running_loss += loss.item() * batchSize
            running_corrects += torch.sum(s_preds == sents_label.data)
            
            labels.append(sents_label)
            outputs.append(s_preds)
            if i%50==0:     
                print('Epoch {} Iteration {}: running_corrects: {} running loss = {:4f}'.format(epoch+1,i,running_corrects,running_loss))
        
            
        epoch_loss = running_loss / len(datasets[phase])
        epoch_acc = running_corrects.double() / len(datasets[phase])
        f1 = f1_score(torch.cat(labels), torch.cat(outputs))
        #print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        print('{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}'.format(phase, epoch_loss, epoch_acc, f1))
        print(' ')
        
        #save the best model
        if phase == 'dev':
            if f1 > best_f1:
                best_f1 = f1
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            #best_model_wts = copy.deepcopy(model.state_dict())
            #torch.save(model.state_dict(), os.path.join('./my-models', 'bestmodel_{}.mdl'.format(file_abbr)))
print('Finish Training, Best Acc:{:.4f} Best F1:{:.4f}'.format(best_acc,best_f1))
