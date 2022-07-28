#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from tabulate import tabulate
from tqdm import tqdm
import random

import argparse
import warnings
import os
from datetime import datetime
import time
import math


#--- Functions ---#

def trim_data(data,seq_length):
    # Remove excessive data that cannot be in a full sequence
    if (len(data)%seq_length) != 0:
        data = data[:-(len(data)%seq_length)]
    else:
        pass
        
    return data


class MagNavDataset(Dataset):
    # split can be 'Train', 'Val', 'Test'
    def __init__(self, df, seq_length, split):
        
        self.seq_length = seq_length
        
        # Get list of features
        self.features   = df.drop(columns=['LINE','IGRFMAG1']).columns.to_list()
        
        if split == 'train':
            
            # Keeping only 1003, 1002 and 1004 flight sections for training except 1002.14
            sections = np.concatenate([df2.LINE.unique(),df3.LINE.unique(),df4.LINE.unique()]).tolist()
            sections.remove(1002.14)
            self.sections = sections
            
            mask_train = pd.Series(dtype=bool)
            for line in sections:
                mask  = (df.LINE == line)
                mask_train = mask|mask_train
            
            # Split in X, y for training
            X_train    = df.loc[mask_train,self.features]
            y_train    = df.loc[mask_train,'IGRFMAG1']
            
            # Removing data that can't fit in full sequence and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(),dtype=torch.float32),seq_length))
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(),[-1,1]),dtype=torch.float32),seq_length)
            
        elif split == 'val':
            
            # Selecting 1002.14 for validation
            mask_val   = (df.LINE == 1002.14)
            self.sections = 1002.14
            
            # Split in X, y for validation
            X_val      = df.loc[mask_val,self.features]
            y_val      = df.loc[mask_val,'IGRFMAG1']
            
            # Removing data that can't fit in full sequence and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_val.to_numpy(),dtype=torch.float32),seq_length))
            self.y = trim_data(torch.tensor(np.reshape(y_val.to_numpy(),[-1,1]),dtype=torch.float32),seq_length)

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



class CNN(torch.nn.Module):

    def __init__(self,seq_length,n_features,n_convblock=2,filters=[32,64]):
        super(CNN,self).__init__()
        
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
        self.architecture.add_module('linear_1',torch.nn.Linear(filters[-1]*math.floor(seq_length/(2**n_convblock)),256))
        self.architecture.add_module(f'relu_{n_convblock+1}',torch.nn.ReLU())
        self.architecture.add_module('linear_2',torch.nn.Linear(256,128))
        self.architecture.add_module(f'relu_{n_convblock+2}',torch.nn.ReLU())
        self.architecture.add_module('linear_3',torch.nn.Linear(128,1))

    def forward(self, x):
        logits = self.architecture(x)
        return logits

    
def make_training(model,EPOCHS):

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4) 
    
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False)
    epoch_bar = tqdm(total=EPOCHS,unit="epoch",desc='Training')

    train_loss_history = []
    val_loss_history = []

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

        #---VALIDATION---#

        val_running_loss = 0.
        
        model.eval()
        
        with torch.no_grad():
            
            for batch_index, (inputs, labels) in enumerate(val_loader):

                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # Make prediction
                predictions = model(inputs)

                # Compute the loss
                loss = criterion(predictions, labels)

                # Gather data and report
                val_running_loss += loss.item()

            val_loss = val_running_loss / batch_index
            val_loss_history.append(val_loss)
        
        epoch_bar.set_postfix(train_loss=train_loss,val_loss=val_loss,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()

    return train_loss_history, val_loss_history

    
#--- Main ---#

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d","--device", type=str, required=True, help="Which GPU to use ('cuda' or 'cpu'). Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true",required=False, help="Shutdown pc after training is done."
    )

    args = parser.parse_args()
    
    # a rajouter avec epochs
    
    batch_sizes = [32]#[16,32,64,128,256,512,1024]
    seqs_len    = [100]#[5,10,15,20,30,50,100,200,500]
    convblocks = {3:[[8,16,32]]}
                  #{1:[[8],[16],[32],[64],[128]],
                  #2:[[8,16],[16,32],[32,64],[64,128]],
                  #3:[[8,16,32],[16,32,64],[32,64,128]],
                  #4:[[8,16,32,64],[16,32,64,128],[32,64,128,256]]}
    
    for batch in batch_sizes:
        for seq_len in seqs_len:
            for n_conv in convblocks:
                for filters in convblocks[n_conv]:

                    EPOCHS     = 25
                    BATCH_SIZE = batch
                    DEVICE     = args.device
                    SEQ_LEN    = seq_len
                    
                    if (n_conv == 3 or n_conv == 4) and seq_len == 5:
                        print('Output of Max pooling is too small, move to next parameters.')
                        continue
                    if n_conv == 4 and (seq_len == 10 or seq_len == 15):
                        print('Output of Max pooling is too small, move to next parameters.')
                        continue

                    # Start timer
                    start_time = time.time()

                    # Remove UserWarnings from Python
                    warnings.filterwarnings("ignore", category=UserWarning) 

                    # Reproducibility
                    torch.manual_seed(27)
                    random.seed(27)
                    np.random.seed(27)

                    # Import Data
                    df2 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1002')
                    df3 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1003')
                    df4 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1004')
                #     df2 = pd.read_hdf('./data/interim/dataset.h5', key=f'Flt1002')
                #     df3 = pd.read_hdf('./data/interim/dataset.h5', key=f'Flt1003')
                #     df4 = pd.read_hdf('./data/interim/dataset.h5', key=f'Flt1004')
                    print('\nImport Done')

                    # Data scaling
                    scaling = 'none' # none or minmax or standard
                #     scaling_range = [-1,1]
                #     MinMaxScaler_2 = MinMaxScaler(scaling_range)
                #     MinMaxScaler_3 = MinMaxScaler(scaling_range)
                #     MinMaxScaler_4 = MinMaxScaler(scaling_range)

                #     df2_scaled = pd.DataFrame()
                #     df3_scaled = pd.DataFrame()
                #     df4_scaled = pd.DataFrame()


                #     df2_scaled[df2.drop(columns=['LINE','IGRFMAG1']).columns] = MinMaxScaler_2.fit_transform(df2.drop(columns=['LINE','IGRFMAG1']))
                #     df3_scaled[df3.drop(columns=['LINE','IGRFMAG1']).columns] = MinMaxScaler_3.fit_transform(df3.drop(columns=['LINE','IGRFMAG1']))
                #     df4_scaled[df4.drop(columns=['LINE','IGRFMAG1']).columns] = MinMaxScaler_4.fit_transform(df4.drop(columns=['LINE','IGRFMAG1']))

                #     df2_scaled.index = df2.index
                #     df3_scaled.index = df3.index
                #     df4_scaled.index = df4.index

                #     df2_scaled[['LINE','IGRFMAG1']] = df2[['LINE','IGRFMAG1']]
                #     df3_scaled[['LINE','IGRFMAG1']] = df3[['LINE','IGRFMAG1']]
                #     df4_scaled[['LINE','IGRFMAG1']] = df4[['LINE','IGRFMAG1']]

                #     print('Data scaling Done\n')

                    # Train, Validation, Test set
                    df_concat = pd.concat([df2,df3,df4],ignore_index=True,axis=0)

                    train = MagNavDataset(df_concat,seq_length=SEQ_LEN,split='train')
                    val   = MagNavDataset(df_concat,seq_length=SEQ_LEN,split='val')

                    # Dataloaders
                    train_loader  = DataLoader(train,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=False)

                    val_loader    = DataLoader(val,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=False)

                    print('Data split Done\n')

                    # Model
                    n_features = train.__getitem__(0)[0].size()[0]
                    
                    model = CNN(seq_length  = SEQ_LEN,
                                n_features  = n_features,
                                n_convblock = n_conv,
                                filters     = filters
                               ).to(DEVICE)
                    model.name = 'CNN'

                    # Loss
                    criterion = RMSELoss()

                    # Training
                    train_loss_history, val_loss_history = make_training(model,EPOCHS)

                    # Saving the model and metrics
                    date = datetime.strftime(datetime.now(),'%y%m%d_%H%M')
                    folder_path = f'models/CNN_runs/{model.name}_{date}'
                    os.mkdir(folder_path)

                    torch.save(model,folder_path+f'/{model.name}.pt')

                    with open(folder_path+'/train_loss.txt','w') as f:
                        for item in train_loss_history:
                            f.write('%s\n' % item)

                    with open(folder_path+'/val_loss.txt','w') as f:
                        for item in val_loss_history:
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
                        f.write(f'Train sections :\n{train.sections}\n\n')
                        f.write(f'Validation sections :\n{val.sections}\n')

                    # Empty GPU ram and shutdown computer
                    torch.cuda.empty_cache()
                    
                    print('\n')
    
    if args.shut == True:
        os.system("shutdown")
    
