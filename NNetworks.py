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
    def __init__(self, input_size = 1024, hidden_size = 1024, num_layers = 1):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = 0.0, batch_first=False) 
        # (seq_len, batch, hidden_size * num_directions)
        
        #initial weights
        #all_weights first dimension == all hidden layers
        nn.init.kaiming_normal_(self.lstm.all_weights[0][0], nonlinearity='leaky_relu') #weights
        nn.init.kaiming_normal_(self.lstm.all_weights[0][1], nonlinearity='leaky_relu') #bias
        
        self.fc = nn.Linear(hidden_size, 2)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):

        # Forward propagate RNN
        out, _ = self.lstm(x)
        out = self.leaky_relu(x)
        # Decode hidden state of last time step
        out = self.fc(out[-1, :, :]) #  (seq_len, batch, hidden_size * num_directions)
        out = self.softmax(out)
        return out