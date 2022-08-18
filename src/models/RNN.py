#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(torch.nn.Module):
    
    def __init__(self, seq_len, drop_lstm1, hidden_size, num_layers, num_LSTM, num_linear, num_neurons):
        
        super(LSTM, self).__init__()
        self.num_LSTM = num_LSTM
        self.num_linear = num_linear
        self.hidden_size = hidden_size
        self.drop_lstm1 = drop_lstm1
        self.num_layers = num_layers
        self.lstms = nn.ModuleList()
        self.linears = nn.ModuleList()
        
        for k in range(num_LSTM):
            if k == 0:
                self.lstms.append(nn.LSTM(seq_len, hidden_size[0], num_layers[0], batch_first=True))
                self.lstms.append(nn.Dropout(self.drop_lstm1))
                continue
                
            self.lstms.append(nn.LSTM(hidden_size[k-1], hidden_size[k], num_layers[k], batch_first=True))
            
        for n in range(num_linear):
            if n == 0:
                self.linears.append(nn.Linear(hidden_size[-1], num_neurons[0]))
                continue
            
            self.linears.append(nn.Linear(num_neurons[n-1], num_neurons[n]))
        
        self.linears.append(nn.Linear(num_neurons[-1],1))
            
        for k in range(len(self.lstms)):
            if k == 1:
                continue
            
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_ih_l0'])
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_hh_l0'])
            if self.lstms[k].bias is not None:
                nn.init.constant_(self.lstms[k]._parameters['bias_ih_l0'], 0)
                nn.init.constant_(self.lstms[k]._parameters['bias_hh_l0'], 0)
        
        for k in range(num_linear):
            nn.init.kaiming_normal_(self.linears[k].weight)
            if self.linears[k].bias is not None:
                nn.init.constant_(self.linears[k].bias, 0)

            
    def forward(self, x):
        
        for k, lstm_k in enumerate(self.lstms):
            if k == 0:
                h = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')
                c = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')

                out, _ = lstm_k(x, (h,c))
#                 out = F.dropout(out,self.drop_lstm1)
                continue
            
            if k == 1:
                out = lstm_k(out)
                continue
                                   
            h = torch.zeros(self.num_layers[k-1], x.size(0), self.hidden_size[k-1]).to('cuda')
            c = torch.zeros(self.num_layers[k-1], x.size(0), self.hidden_size[k-1]).to('cuda')

            out, _ = lstm_k(out, (h,c))
        
        out = out[:, -1, :]
        
        for k, linear_k in enumerate(self.linears):
            if k == self.num_linear:
                out = linear_k(out)
                return out
            
            out = F.relu(linear_k(out))
            
# class LSTM(torch.nn.Module):
#     def __init__(self, seq_len):
#         super(LSTM, self).__init__()
#         self.lstm1 = torch.nn.LSTM(seq_len, 64, 1, batch_first=True)
#         self.dropout = torch.nn.Dropout(0.2)
#         self.lstm2 = torch.nn.LSTM(64, 32, 2, batch_first=True)
#         self.lstm3 = torch.nn.LSTM(32, 16, 2, batch_first=True)
#         self.fc = torch.nn.Linear(16,1)
    
#     def forward(self, x):
#         h1 = torch.zeros(1, x.size(0), 64).to('cuda')
#         c1 = torch.zeros(1, x.size(0), 64).to('cuda')
        
#         h2 = torch.zeros(2, x.size(0), 32).to('cuda')
#         c2 = torch.zeros(2, x.size(0), 32).to('cuda')
    
#         h3 = torch.zeros(2, x.size(0), 16).to('cuda')
#         c3 = torch.zeros(2, x.size(0), 16).to('cuda')
        
#         out, hidden = self.lstm1(x, (h1,c1))
#         out = self.dropout(out)
#         out, hidden = self.lstm2(out, (h2,c2))
#         out, hidden = self.lstm3(out, (h3,c3))
#         out = out[:, -1, :]
#         out = self.fc(out)
        
#         return out

class Optuna_LSTM(torch.nn.Module):
    
    def __init__(self, seq_len, drop_lstm1, hidden_size, num_layers, num_LSTM, num_linear, num_neurons):
        
        super(Optuna_LSTM, self).__init__()
        self.num_LSTM = num_LSTM
        self.num_linear = num_linear
        self.hidden_size = hidden_size
        self.drop_lstm1 = drop_lstm1
        self.num_layers = num_layers
        self.lstms = nn.ModuleList()
        self.linears = nn.ModuleList()
        
        for k in range(num_LSTM):
            if k == 0:
                self.lstms.append(nn.LSTM(seq_len, hidden_size[0], num_layers[0], batch_first=True))
                continue
                
            self.lstms.append(nn.LSTM(hidden_size[k-1], hidden_size[k], num_layers[k], batch_first=True))
            
        for n in range(num_linear):
            if n == 0:
                self.linears.append(nn.Linear(hidden_size[-1], num_neurons[0]))
                continue
            
            self.linears.append(nn.Linear(num_neurons[n-1], num_neurons[n]))
        
        self.linears.append(nn.Linear(num_neurons[-1],1))
            
        for k in range(num_LSTM):
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_ih_l0'])
            nn.init.kaiming_normal_(self.lstms[k]._parameters['weight_hh_l0'])
            if self.lstms[k].bias is not None:
                nn.init.constant_(self.lstms[k]._parameters['bias_ih_l0'], 0)
                nn.init.constant_(self.lstms[k]._parameters['bias_hh_l0'], 0)
        
        for k in range(num_linear):
            nn.init.kaiming_normal_(self.linears[k].weight)
            if self.linears[k].bias is not None:
                nn.init.constant_(self.linears[k].bias, 0)

            
    def forward(self, x):
        
        for k, lstm_k in enumerate(self.lstms):
            if k == 0:
                h = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')
                c = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')

                out, _ = lstm_k(x, (h,c))
                out = F.dropout(out)
                continue
                                   
            h = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')
            c = torch.zeros(self.num_layers[k], x.size(0), self.hidden_size[k]).to('cuda')

            out, _ = lstm_k(out, (h,c))
        
        out = out[:, -1, :]
        
        for k, linear_k in enumerate(self.linears):
            if k == self.num_linear:
                out = linear_k(out)
                return out
            
            out = F.relu(linear_k(out))
            
class GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_size,1)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        
        out, _ = self.gru(x,h0)
        
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        
        return out