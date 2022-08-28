#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(torch.nn.Module):

    def __init__(self,seq_len,in_channels):
        super(CNN,self).__init__()

        self.layers = torch.nn.Sequential(

            # Conv layers
            nn.Conv1d(in_channels  = in_channels,
                      out_channels = 8,
                      kernel_size  = 3,
                      padding = 1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels  = 8,
                      out_channels = 16,
                      kernel_size  = 3,
                      padding = 1),
            nn.LeakyReLU(),

            # Linear layers
            nn.Flatten(),
            nn.Linear(16*seq_len,16),
            nn.LeakyReLU(),
            nn.Linear(16,4),
            nn.LeakyReLU(),
            nn.Linear(4,1))
        
        nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='relu')
        nn.init.constant_(self.layers[0].bias, 0)
        nn.init.kaiming_normal_(self.layers[2].weight, nonlinearity='relu')
        nn.init.constant_(self.layers[2].bias, 0)



    def forward(self, x):
        logits = self.layers(x)
        return logits
    
# class CNN(nn.Module):

    
#     def __init__(self, num_conv_layers, num_filters, num_neurons, drop_conv1, drop_fc1, seq_len):

#         super(CNN, self).__init__()                                                   # Initialize parent class
#         in_size = 17                                                                  # Input features size
#         kernel_size = 2                                                               # Conv filter size
        
#         # Define conv layers
#         self.convs = nn.ModuleList([nn.Conv1d(in_size, num_filters[0], kernel_size)]) # List with the Conv layers
#         out_size = seq_len - kernel_size + 1                                          # Size after conv layer
#         out_size = int(out_size / 2)                                                  # Size after pooling
        
#         for i in range(1, num_conv_layers):
#             self.convs.append(nn.Conv1d(num_filters[i-1], num_filters[i], kernel_size))
#             out_size = out_size - kernel_size + 1                                     # Size after conv layer
#             out_size = int(out_size / 2)                                              # Size after pooling
        
#         self.conv1_drop = nn.Dropout2d(p=drop_conv1)                                  # Dropout for conv1d
#         self.out_feature = num_filters[num_conv_layers-1] * out_size                  # Size after flattened features
        
#         self.fc1 = nn.Linear(self.out_feature, num_neurons[0])                        # Fully connected layer 1
#         self.fc2 = nn.Linear(num_neurons[0], num_neurons[-1])                         # Fully connected layer 2
#         self.fc3 = nn.Linear(num_neurons[-1], 1)                                      # Fully connected layer 3
        
#         self.p1 = drop_fc1                                                            # Dropout ratio for FC1
        
#         # Initialize weights with He initialization
#         for i in range(1, num_conv_layers):
#             nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
#             if self.convs[i].bias is not None:
#                 nn.init.constant_(self.convs[i].bias, 0)
#         nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
#         nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
    
    
#     def forward(self, x):
        
#         for i, conv_i in enumerate(self.convs):
#             if i == 2:                                                                # Apply dropout at 2nd layer
#                 x = F.relu(F.max_pool1d(self.conv1_drop(conv_i(x)),2))                # Apply conv_i, Dropout, max-pooling(kernel_size =2), ReLU
#             else:
#                 x = F.relu(F.max_pool1d(conv_i(x),2))                                 # Apply conv_i, max-pooling(kernel_size=2), ReLU
        
#         x = x.view(-1, self.out_feature)                                              # Flatten tensor
#         x = F.relu(self.fc1(x))                                                       # Apply FC1, ReLU
#         x = F.dropout(x, p=self.p1, training=self.training)                           # Apply Dropout after FC1 only when training
#         x = F.relu(self.fc2(x))                                                       # Apply FC2, ReLU
#         x = self.fc3(x)                                                               # Apply FC3
        
#         return x
    
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


class Optuna_CNN(torch.nn.Module):

    def __init__(self,seq_length,n_features,n_convblock=2,filters=[16,32], num_neurons=[64,8]):
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
        logits = self.architecture(x)
        return logits