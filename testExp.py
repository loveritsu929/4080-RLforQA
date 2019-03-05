# -*- coding: utf-8 -*-
import os, copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.cuda as cuda
import numpy as np

import NNetworks
import dlDataset

from bert_serving.client import BertClient

#===================== Hyper Parameters
device = torch.device('cuda:0')

batchSize = 2048
threshold = 0.001
modelSaveDir = './exps/lstm_para_2'

testDataset = dlDataset.ParaDataset(mode='test')
#loss_weights = torch.as_tensor(sklearn.utils.class_weight.compute_class_weight('balanced', [0,1], testDataset.label_array[:10240]))
testLoader = data.DataLoader(testDataset, batch_size = batchSize, drop_last = True, shuffle = False, num_workers = 0)

bert = BertClient(check_length=False)

#===================== testing
def test_model(model, current_acc):
    model.set_mode('test')
    model.eval()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    print("Testing")
    print('#' * 20)

    running_corrects = 0
    true_positive = 0
    false_positive = 0
    num_true = 0
    num_pred_true = 0
    
    # ds: q, title+para , label
    for i, (q, para, label) in enumerate(testLoader, 0):
        q_t = torch.as_tensor(bert.encode(list(q))).unsqueeze_(dim = 1)
        para = [list(item) for item in para] # list of batchSize list, len = smallest # of sents in paras
        para = np.array(para) 
        (num_sent, w) = para.shape
        #assert w == batchSize
        
        para_t = torch.zeros((batchSize, num_sent, 1024))
        for k in range(batchSize):
            #print(para[:, k])
            temp = torch.as_tensor(bert.encode(list(para[:, k]))) # para[:,k] == the kth paragraph, a list of n sents
            para_t[k] = temp
        
        sample = torch.cat([q_t, para_t], dim=1) # size: b seq f
        sample = sample.to(device)
        label = label.to(device)
        
        out = model(sample) #.squeeze() # testing: fc output, batchSize*1 
        
        #TODO
        out = nn.functional.softmax(out[:,1]) #scores for label '1'
        maxScore = torch.max(out)
        preds = (out > maxScore - threshold).long()
        
        running_corrects += torch.sum(preds == label.data)
        true_positive += torch.sum(preds * label.data)
        false_positive += np.sum(np.logical_and(preds.cpu().numpy(), (label==0).cpu().numpy()))
        num_true += torch.sum(label.data)
        num_pred_true += torch.sum(preds)
        
        true_pos_acc = true_positive.double() / num_true.double()
        false_pos_rate = false_positive / num_pred_true.double()
        
        
        if i % 10 == 0:
            print('Iter {}: true_pos = {} TP_acc = {:.4f} FP% = {:.4f}'.format(i, true_positive, true_pos_acc, false_pos_rate))
        
    
    acc = running_corrects.double() / len(testDataset)  
    if acc > current_acc:
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(modelSaveDir, 'lstm1_best_model.mdl'))
    
    
    print('Finish testing: acc = {:.4f}'.format(acc))
    print('#' * 20)
    return acc

if __name__ == '__main__':
    model_w = './exps/lstm_para_2/ep5_loss=134393.3278.mdl'
    print('To test: ', model_w)
    model = NNetworks.MyLSTM()
    model.load_state_dict(torch.load(model_w))
    model = model.to(device)
    test_model(model, 100)
