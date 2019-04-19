# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(2048,2)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.softmax(x)
        
        return out
    
class MyLSTM(nn.Module):
    def __init__(self, mode = 'train', input_size = 1024, hidden_size = 1024, num_layers = 1):
        super(MyLSTM, self).__init__()
        self.mode = mode
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.0, batch_first=True) 
        # (seq_len, batch, hidden_size * num_directions)
        
        #initial weights
        #all_weights first dimension == all hidden layers
        nn.init.kaiming_normal_(self.lstm.all_weights[0][0], a = 0.01, nonlinearity='leaky_relu') #weights
        nn.init.kaiming_normal_(self.lstm.all_weights[0][1], a = 0.01, nonlinearity='leaky_relu') #bias
        
        self.fc = nn.Linear(hidden_size, 2)
        #self.fc = nn.Linear(hidden_size, 1)  #for My loss 
        #nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        
    def set_mode(self, mode):
        self.mode = mode    
        
    def forward(self, x):
        if self.mode == 'train':
            # Forward propagate RNN
            out, _ = self.lstm(x)
            out = self.leaky_relu(x)
            # Decode hidden state of last time step
            out = self.fc(out[:, -1, :]) #  False: (seq_len, batch, hidden_size * num_directions)
            #out = self.sigmoid(out)
            return out
        
        if self.mode == 'test':
            # Forward propagate RNN
            out, _ = self.lstm(x)
            out = self.leaky_relu(x)
            # Decode hidden state of last time step
            out = self.fc(out[:, -1, :]) #   batch,seq,feature         False: (seq_len, batch, hidden_size * num_directions)
            #out = self.softmax(out)
            return out


        
class SentParaNN(nn.Module):
    def __init__(self):
        super(SentParaNN, self).__init__()
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=2, dropout = 0.0, batch_first=True, bidirectional = True)
        
        self.paraNN = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
                )
        
        self.sentNN = nn.Sequential(
                nn.Linear(2048, 1024), #bidirection
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 2)
                )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.para_softmax = nn.Softmax(dim=1)
        self.sent_softmax = nn.Softmax(dim=2)
    
    def forward(self, sents, para):
        para_out = self.paraNN(para)
        #para_out = self.para_softmax(para_out)
        
        sents_out, _ = self.lstm(sents)
        sents_out = self.relu(sents_out)
        sents_out = self.sentNN(sents_out[:, 1:, :])
        #sents_out = self.sent_softmax(sents_out)
        
        return sents_out, para_out
        
if __name__ == '__main__':
    net = SentParaNN()
    s_t = torch.rand(1,10,1024)
    p_t = torch.rand(1,1024)
    s_out, p_out = net(s_t, p_t)