# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(2048,2)
        
        self.softmax = nn.Softmax()
        
    def foward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        out = self.softmax(x)
        
        return out
    
    