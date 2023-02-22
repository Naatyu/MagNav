#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Optuna_CNN(torch.nn.Module):
    """
    Class to create CNN model.
    """
    def __init__(self,seq_length,n_features,n_convblock=2,filters=[16,32], num_neurons=[64,8]):
        """
        Initialize the model.

        Arguments:
        - `seq_length`  : number of time steps in an input sequence
        - `n_features`  : number of input features in the model
        - `n_convblock` : number of desired convolutional blocks
        - `filters`     : filters by convblocks (must be a list)
        - `num_neurons` : number of neurons for linear layers (must be a list)

        Returns:
        - None
        """
        super(Optuna_CNN,self).__init__()
        
        # Raise Error if number of conv blocks don't match number of filters
        if type(filters) != list:
            raise TypeError ("Filter should be a list (if only 1 filter put in 1 element list. Ex : filter = [8])")

        if len(filters) != n_convblock:
            raise ValueError ("Number of convolutional blocks and number of filters doesn't match")
        
        self.architecture = torch.nn.Sequential()
        
        for i in range(n_convblock):

            if i == 0:
                self.architecture.add_module(f'conv_{i+1}',
                                            torch.nn.Conv1d(in_channels  = n_features,
                                                            out_channels = filters[i],
                                                            kernel_size  = 3,
                                                            stride       = 1,
                                                            padding      = 1,
                                                            padding_mode = 'zeros'))
                self.architecture.add_module(f'relu_{i+1}',torch.nn.ReLU())
                self.architecture.add_module(f'maxpool_{i+1}',torch.nn.MaxPool1d(kernel_size = 2,stride = 2))
                                             
            else:
                self.architecture.add_module(f'conv_{i+1}',
                        torch.nn.Conv1d(in_channels  = filters[i-1],
                                        out_channels = filters[i],
                                        kernel_size  = 3,
                                        stride       = 1,
                                        padding      = 1,
                                        padding_mode = 'zeros'))
                self.architecture.add_module(f'relu_{i+1}',torch.nn.ReLU())
                self.architecture.add_module(f'maxpool_{i+1}',torch.nn.MaxPool1d(kernel_size = 2,stride = 2))
            

        self.architecture.add_module('flatten',torch.nn.Flatten())
        self.architecture.add_module('linear_1',torch.nn.Linear(filters[-1]*math.floor(seq_length/(2**n_convblock)),num_neurons[0]))
        # self.architecture.add_module('linear_1',torch.nn.Linear(n_features*seq_length/2,num_neurons[0]))
        self.architecture.add_module(f'relu_{n_convblock+1}',torch.nn.ReLU())
        self.architecture.add_module('linear_2',torch.nn.Linear(num_neurons[0],num_neurons[1]))
        self.architecture.add_module(f'relu_{n_convblock+2}',torch.nn.ReLU())
        self.architecture.add_module('linear_3',torch.nn.Linear(num_neurons[1],1))

    def forward(self, x):
        """
        Forward input sequence in the model.

        Arguments:
        - `x`  : input sequence

        Returns:
        - `logits` : model prediction
        """
        logits = self.architecture(x)
        return logits

# ResNet
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.expansion*planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        return out
    
class ResNet(nn.Module):
    
    def __init__(self,block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(11, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(2*512*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0),-1)
        out = self.linear(out)

        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])