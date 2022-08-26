#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class MLP(torch.nn.Module):
    
    def __init__(self, seq_len, channels):
        super(MLP, self).__init__()
        self.architecture = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(seq_len*channels,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,8),
            torch.nn.ReLU(),
            torch.nn.Linear(8,1),
        )
        self.architecture.apply(init_weights)
        
        
    def forward(self, x):
        logits = self.architecture(x)
        return logits