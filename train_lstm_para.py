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
#import testExp
from bert_serving.client import BertClient

#===================== Hyper Parameters
device = torch.device('cuda:0')

batchSize = 1024
lnR = 1e-3
maxEpoches = 20
currentEpoch = 0

modelSaveDir = './exps/biLSTM'
model_weight = ' '
if not os.path.exists(modelSaveDir):
    os.makedirs(modelSaveDir)
model = NNetworks.MyLSTM()
if os.path.isfile(os.path.join(modelSaveDir, model_weight)):
	print('Continue training: ', model_weight)
	model.load_state_dict(torch.load(os.path.join(modelSaveDir, model_weight)))
	currentEpoch = int(model_weight.split('_')[0][2:]) + 1
	print('Start from epoch {}'.format(currentEpoch))
model = model.to(device)

subrange = 204800
trainDataset = dlDataset.ParaDataset(mode='train')
#loss_weights = torch.as_tensor(sklearn.utils.class_weight.compute_class_weight('balanced', [0,1], trainDataset.label_array[:subrange])).float()
#loss_weights = loss_weights.to(device)
trainDataset = data.Subset(trainDataset, [i for i in range(subrange)])
trainLoader = data.DataLoader(trainDataset, batch_size = batchSize, shuffle = True, num_workers = 0)
optimizer = optim.Adam(model.parameters(), lr = lnR, amsgrad = True)
bert = BertClient(check_length=False)
#=====================
class MyLoss(nn.Module):
    def __init__(self, weights):
        super(MyLoss, self).__init__()
        self.loss_weights = weights
        
    def forward(self, out, label):
        loss = 0
        assert out.size() == label.size()
        out = torch.log(out)
        out = torch.mul(out, -1)
        for i in range(out.size()[0]):
            if label[i] == 1:
                loss += out[i] * self.loss_weights[1]
            elif label[i] == 0:
                loss += (1 - out[i]) * self.loss_weights[0]
        return loss
    
lossFunction = nn.CrossEntropyLoss()
#===================== training
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
print("Ready for trainning")
for ep in range(currentEpoch, maxEpoches):
    print('Epoch {}/{}'.format(ep, maxEpoches))
    print('=' * 20)
    model.set_mode('train')
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    # ds: q, title+para , label
    for i, (q, para, label) in enumerate(trainLoader, 0):
        q_t = torch.as_tensor(bert.encode(list(q))).unsqueeze_(dim = 1)
        para = [list(item) for item in para] # list of batchSize list, len = smallest # of sents in paras
        para = np.array(para) 
        (num_sent, w) = para.shape
        assert w == batchSize
        
        para_t = torch.zeros((batchSize, num_sent, 1024))
        for k in range(batchSize):
            temp = torch.as_tensor(bert.encode(list(para[:, k]))) # para[:,k] == the kth paragraph, a list of n sents
            para_t[k] = temp
        
        sample = torch.cat([q_t, para_t], dim=1) # size: b seq f
        sample = sample.to(device)
        label = label.to(device)
        
        out = model(sample) #.squeeze() # trainning: fc output  ##1d sigmoid score
        #_, preds = torch.max(out, dim=1)
        loss = lossFunction(out, label)
        
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
         
        # statistics
        running_loss += loss.item() * sample.size(0)
        
        # batchSize samples in each iteration 
        if i % 10 == 0:
            print('Epoch {} Iteration {}: loss = {:.6f}'.format(ep,i,loss.item()))
             # deep copy the model
#    if epoch_acc > best_acc:
#        best_acc = epoch_acc
#        best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), os.path.join(modelSaveDir, 'ep{}_loss={:.4f}.mdl'.format(ep,running_loss)))
    epoch_loss = running_loss / len(trainDataset)
    
    print('Finish epoch {} epLoss: {:.4f}\n'.format(ep, epoch_loss))
    
