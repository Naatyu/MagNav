#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from tqdm import tqdm
import random
from sklearn.model_selection import KFold

import argparse
import warnings
import os
from datetime import datetime
import time
import math


#--- Functions ---#
def reset_weights(m):
    '''
    Reset weights to avoid weight leakage
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
    

def trim_data(data,seq_length):
    # Remove excessive data that cannot be in a full sequence
    if (len(data)%seq_length) != 0:
        data = data[:-(len(data)%seq_length)]
    else:
        pass
        
    return data


class MagNavDataset(Dataset):
    # split can be 'Train', 'Val', 'Test'
    def __init__(self, df, seq_length, n_fold, split):
        
        self.seq_length = seq_length
        
        # Get list of features
        self.features = df.drop(columns=['LINE','IGRFMAG1']).columns.to_list()
        
        # Get train sections for fold n
        train_fold_0 = np.concatenate([df2.LINE.unique(),df3.LINE.unique(),df4.LINE.unique(),df6.LINE.unique()]).tolist()
        test_fold_0  = df7.LINE.unique().tolist()
        
        train_fold_1 = np.concatenate([df3.LINE.unique(),df4.LINE.unique(),df6.LINE.unique(),df7.LINE.unique()]).tolist()
        test_fold_1  = df2.LINE.unique().tolist()
        
        train_fold_2 = np.concatenate([df4.LINE.unique(),df6.LINE.unique(),df7.LINE.unique(),df2.LINE.unique()]).tolist()
        test_fold_2  = df3.LINE.unique().tolist()
        
        if n_fold == 0:
            self.train_sections = train_fold_0
            self.test_sections = test_fold_0
        elif n_fold == 1:
            self.train_sections = train_fold_1
            self.test_sections = test_fold_1
        elif n_fold == 2:
            self.train_sections = train_fold_2
            self.test_sections = test_fold_2
        
        
        if split == 'train':
            
            mask_train = pd.Series(dtype=bool)
            for line in self.train_sections:
                mask  = (df.LINE == line)
                mask_train = mask|mask_train
            
            # Split in X, y for training
            X_train    = df.loc[mask_train,self.features]
            y_train    = df.loc[mask_train,'IGRFMAG1']
            
            # Removing data that can't fit in full sequence and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(),dtype=torch.float32),seq_length))
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(),[-1,1]),dtype=torch.float32),seq_length)
            
        elif split == 'test':
            
            mask_test = pd.Series(dtype=bool)
            for line in self.test_sections:
                mask  = (df.LINE == line)
                mask_test = mask|mask_test
            
            # Split in X, y for test
            X_test      = df.loc[mask_test,self.features]
            y_test      = df.loc[mask_test,'IGRFMAG1']
            
            # Removing data that can't fit in full sequence and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_test.to_numpy(),dtype=torch.float32),seq_length))
            self.y = trim_data(torch.tensor(np.reshape(y_test.to_numpy(),[-1,1]),dtype=torch.float32),seq_length)

    def __getitem__(self, index):
        X = self.X[:,index:(index+self.seq_length)]
        y = self.y[index+self.seq_length-1]
        return X, y
    
    def __len__(self):
        return len(torch.t(self.X))-self.seq_length


class RMSELoss(torch.nn.Module):
    
    def __init__(self):
        super(RMSELoss,self).__init__()
        
    def forward(self,yhat,y):
        criterion = torch.nn.MSELoss()
        loss = torch.sqrt(criterion(yhat,y)+1e-6)
        return loss 


# class CNN(torch.nn.Module):

#     def __init__(self,seq_length):
#         super(CNN,self).__init__()

#         self.layers = torch.nn.Sequential(

#             torch.nn.Conv1d(in_channels  = 11,
#                             out_channels = 16,
#                             kernel_size  = 2),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool1d(kernel_size = 2,
#                                stride      = 2),
#             torch.nn.Conv1d(in_channels  = 16,
#                             out_channels = 32,
#                             kernel_size  = 2),
#             torch.nn.ReLU(),
#             torch.nn.MaxPool1d(kernel_size = 2,
#                               stride       = 2),

#             torch.nn.Flatten(),
#             torch.nn.Linear(768,90),
#             torch.nn.ReLU(),
#             torch.nn.Linear(90,20),
#             torch.nn.ReLU(),
#             torch.nn.Linear(20,1))
        
#         torch.nn.init.kaiming_normal_(self.layers[0].weight, nonlinearity='relu')
#         torch.nn.init.constant_(self.layers[0].bias, 0)
#         torch.nn.init.kaiming_normal_(self.layers[3].weight, nonlinearity='relu')
#         torch.nn.init.constant_(self.layers[3].bias, 0)
#         torch.nn.init.kaiming_normal_(self.layers[7].weight, nonlinearity='relu')
# #         torch.nn.init.constant_(self.layers[8].bias, 0)
#         torch.nn.init.kaiming_normal_(self.layers[9].weight, nonlinearity='relu')
# #         torch.nn.init.constant_(self.layers[10].bias, 0)
# #         torch.nn.init.kaiming_normal_(self.layers[12].weight)
# #         torch.nn.init.constant_(self.layers[12].bias, 0)

#     def forward(self, x):
#         logits = self.layers(x)
#         return logits
    
class CNN(nn.Module):

    
    def __init__(self, num_conv_layers, num_filters, num_neurons, drop_conv1, drop_fc1, seq_len):

        super(CNN, self).__init__()                                                   # Initialize parent class
        in_size = 11                                                                  # Input features size
        kernel_size = 2                                                               # Conv filter size
        
        # Define conv layers
        self.convs = nn.ModuleList([nn.Conv1d(in_size, num_filters[0], kernel_size)]) # List with the Conv layers
        out_size = seq_len - kernel_size + 1                                          # Size after conv layer
        out_size = int(out_size / 2)                                                  # Size after pooling
        
        for i in range(1, num_conv_layers):
            self.convs.append(nn.Conv1d(num_filters[i-1], num_filters[i], kernel_size))
            out_size = out_size - kernel_size + 1                                     # Size after conv layer
            out_size = int(out_size / 2)                                              # Size after pooling
        
        self.conv1_drop = nn.Dropout2d(p=drop_conv1)                                  # Dropout for conv1d
        self.out_feature = num_filters[num_conv_layers-1] * out_size                  # Size after flattened features
        
        self.fc1 = nn.Linear(self.out_feature, num_neurons[0])                        # Fully connected layer 1
        self.fc2 = nn.Linear(num_neurons[0], num_neurons[-1])                         # Fully connected layer 2
        self.fc3 = nn.Linear(num_neurons[-1], 1)                                      # Fully connected layer 3
        
        self.p1 = drop_fc1                                                            # Dropout ratio for FC1
        
        # Initialize weights with He initialization
        for i in range(1, num_conv_layers):
            nn.init.kaiming_normal_(self.convs[i].weight, nonlinearity='relu')
            if self.convs[i].bias is not None:
                nn.init.constant_(self.convs[i].bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
    
    
    def forward(self, x):
        
        for i, conv_i in enumerate(self.convs):
            if i == 2:                                                                # Apply dropout at 2nd layer
                x = F.relu(F.max_pool1d(self.conv1_drop(conv_i(x)),2))                # Apply conv_i, Dropout, max-pooling(kernel_size =2), ReLU
            else:
                x = F.relu(F.max_pool1d(conv_i(x),2))                                 # Apply conv_i, max-pooling(kernel_size=2), ReLU
        
        x = x.view(-1, self.out_feature)                                              # Flatten tensor
        x = F.relu(self.fc1(x))                                                       # Apply FC1, ReLU
        x = F.dropout(x, p=self.p1, training=self.training)                           # Apply Dropout after FC1 only when training
        x = F.relu(self.fc2(x))                                                       # Apply FC2, ReLU
        x = self.fc3(x)                                                               # Apply FC3
        
        return x

    
def make_training(model,EPOCHS):

    optimizer = torch.optim.Adam(model.parameters(),lr=1.5e-4) 
    
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False, position=0, ncols=150)
    epoch_bar = tqdm(total=EPOCHS,unit="epoch",desc='Training',leave=False, position=1, ncols=150)

    train_loss_history = []
    test_loss_history = []

    for epoch in range(EPOCHS):

        #---TRAIN---#

        train_running_loss = 0.

        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        
        batch_bar.reset()
        # Enumerate allow to track batch index and intra-epoch reporting 
        for batch_index, (inputs, labels) in enumerate(train_loader):

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Make prediction for this batch
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions, labels)

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_running_loss += loss.item()
            
            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1),lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()

        train_loss = train_running_loss / batch_index
        train_loss_history.append(train_loss)

        #---Test---#

        test_running_loss = 0.
        
        model.eval()
        
        with torch.no_grad():
            
            for batch_index, (inputs, labels) in enumerate(test_loader):

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Make prediction
                predictions = model(inputs)

                # Compute the loss
                loss = criterion(predictions, labels)

                # Gather data and report
                test_running_loss += loss.item()

            test_loss = test_running_loss / batch_index
            test_loss_history.append(test_loss)
        
        epoch_bar.set_postfix(train_loss=train_loss,test_loss=test_loss,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    print('\n')
    return train_loss_history, test_loss_history

    
#--- Main ---#

if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    # Remove UserWarnings from Python
    warnings.filterwarnings("ignore", category=UserWarning) 
    
    # Reproducibility
    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d","--device", type=str, required=True, help="Which GPU to use (cuda or cpu). Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "-e","--epochs", type=int, required=True, help="Number of epochs to train the model. Ex : --epochs 200", metavar=""
    )
    parser.add_argument(
        "-b","--batch", type=int, required=True, help="Batch size for training. Ex : --batch 64", metavar=""
    )
#     parser.add_argument(
#         "-s","--seq", type=int, required=True, help="Length sequence of data. Ex : --seq 15", metavar=""
#     )
    parser.add_argument(
        "--shut", action="store_true",required=False, help="Shutdown pc after training is done."
    )
    
    args = parser.parse_args()
    
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    DEVICE     = args.device
    SEQ_LEN    = 100
    
    # Import Data
    
    df2 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1002')
    df3 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1003')
    df4 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1004')
    df6 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1006')
    df7 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1007')
    
    scaling = 'none'
    
    # Train, Validation, Test set
    df_concat = pd.concat([df2,df3,df4,df6,df7],ignore_index=True,axis=0)
    
    # 
    train_loss_history = []
    test_loss_history = []
    last_test = []
    
    for n_fold in range(3):
        
        print('\n--------------------')
        print(f'Fold number {n_fold}')
        print('--------------------\n')
        
        
        train = MagNavDataset(df_concat, seq_length=SEQ_LEN, n_fold=n_fold, split='train')
        test  = MagNavDataset(df_concat, seq_length=SEQ_LEN, n_fold=n_fold, split='test')

        # Dataloaders
        train_loader  = DataLoader(train,
                               batch_size=BATCH_SIZE,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False)

        test_loader    = DataLoader(test,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False,
                                   num_workers=0,
                                   pin_memory=False)

        # Model
        model = CNN(2,[48,120], [20,90], 0.3, 0.015,SEQ_LEN).to(DEVICE)
#         model = CNN(SEQ_LEN).to(DEVICE)
        model.name = 'CNN'

        # Loss
        criterion = torch.nn.MSELoss()

        # Training
        train_loss, test_loss = make_training(model,EPOCHS)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        last_test.append(test_loss[-1])
        
    perf_folds = sum(last_test)/3
        
    print('\n-------------------------')
    print('Performance for all folds')
    print('-------------------------')
    print(f'Fold 1 | RMSE = {np.sqrt(last_test[0]):.2f} nT')
    print(f'Fold 2 | RMSE = {np.sqrt(last_test[1]):.2f} nT')
    print(f'Fold 3 | RMSE = {np.sqrt(last_test[2]):.2f} nT')
    print(f'Total  | RMSE = {np.sqrt(perf_folds):.2f} nT\n')
    
    # Saving the model and metrics
    date = datetime.strftime(datetime.now(),'%y%m%d_%H%M')
    folder_path = f'models/CNN_runs/{model.name}_{date}'
    os.mkdir(folder_path)
    
    torch.save(model,folder_path+f'/{model.name}.pt')
    
    with open(folder_path+'/train_loss.txt','w') as f:
        for item in train_loss_history:
            f.write('%s\n' % item)
        
    with open(folder_path+'/test_loss.txt','w') as f:
        for item in test_loss_history:
            f.write('%s\n' % item)
            
    # Saving parameters in txt 
    end_time = time.time()-start_time

    with open(folder_path+'/parameters.txt','w') as f:
        f.write(f'Epochs :\n{EPOCHS}\n\n')
        f.write(f'Batch_size :\n{BATCH_SIZE}\n\n')
        f.write(f'Loss :\n{criterion}\n\n')
        f.write(f'Scaling :\n{scaling}\n\n')
        f.write(f'Input_shape :\n{[train.__getitem__(0)[0].size()[0],train.__getitem__(0)[0].size()[1]]}\n\n')
        f.write(f'Sequence_length :\n{SEQ_LEN}\n\n')
        f.write(f'Training_device :\n{DEVICE}\n\n')
        f.write(f'Execution_time :\n{end_time:.2f}s\n\n')
        f.write(f'Architecture :\n{model}\n\n')
        f.write(f'Features :\n{train.features}\n\n')
        
    # Empty GPU ram and shutdown computer
    torch.cuda.empty_cache()
    
    if args.shut == True:
        os.system("shutdown")
    
