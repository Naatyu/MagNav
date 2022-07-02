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
            
            # Keeping only 1003, 1002, 1006 and 1004 flight sections for training except 1002.14
            sections = np.concatenate([df2.LINE.unique(),df3.LINE.unique(),df4.LINE.unique(),df6.LINE.unique()]).tolist()
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
            
            # Selecting 1007.02 for validation
            mask_val   = (df.LINE == 1007.02)
            self.sections = 1007.02
            
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

    def __init__(self,seq_length):
        super(CNN,self).__init__()

        self.architecture = torch.nn.Sequential(

            torch.nn.Conv1d(in_channels  = 11,
                            out_channels = 128,
                            kernel_size  = 2,
                            stride       = 1,
                            padding      = 0,
                            padding_mode = 'zeros'),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = 2,
                               stride      = 2),
            torch.nn.Conv1d(in_channels  = 128,
                            out_channels = 128,
                            kernel_size  = 2,
                            stride       = 1,
                            padding      = 0,
                            padding_mode = 'zeros'),
            torch.nn.Dropout2d(p=0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = 2,
                              stride       = 2),

            torch.nn.Flatten(),
            torch.nn.Linear(49*128,512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(),
            torch.nn.Linear(512,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,1))

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
    parser.add_argument(
        "-s","--seq", type=int, required=True, help="Length sequence of data. Ex : --seq 15", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true",required=False, help="Shutdown pc after training is done."
    )
    
    args = parser.parse_args()
    
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    DEVICE     = args.device
    SEQ_LEN    = args.seq
    
    # Import Data
    
    df2 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1002')
    df3 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1003')
    df4 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1004')
    df6 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1006')
    df7 = pd.read_hdf('./data/interim/Chall_dataset.h5', key=f'Flt1007')
    
    scaling = 'None'
#     scaling_range = [-1,1]

#     df2_scaled = pd.DataFrame()
#     df3_scaled = pd.DataFrame()
#     df4_scaled = pd.DataFrame()

#     min_df2 = df2['TL_comp_mag5_cl'].min()
#     max_df2 = df2['TL_comp_mag5_cl'].max()

#     min_df3 = df3['TL_comp_mag5_cl'].min()
#     max_df3 = df3['TL_comp_mag5_cl'].max()

#     min_df4 = df4['TL_comp_mag5_cl'].min()
#     max_df4 = df4['TL_comp_mag5_cl'].max()
    
#     min_df6 = df6['TL_comp_mag5_cl'].min()
#     max_df6 = df6['TL_comp_mag5_cl'].max()
    
#     min_df7 = df7['TL_comp_mag5_cl'].min()
#     max_df7 = df7['TL_comp_mag5_cl'].max()

#     df2_scaled = -1+((df2 - min_df2)*(2))/(max_df2-min_df2)
#     df3_scaled = -1+((df3 - min_df3)*(2))/(max_df3-min_df3)
#     df4_scaled = -1+((df4 - min_df4)*(2))/(max_df4-min_df4)
#     df6_scaled = -1+((df6 - min_df6)*(2))/(max_df6-min_df6)
#     df7_scaled = -1+((df7 - min_df7)*(2))/(max_df7-min_df7)

#     df2_scaled[['LINE','IGRFMAG1']] = df2[['LINE','IGRFMAG1']]
#     df3_scaled[['LINE','IGRFMAG1']] = df3[['LINE','IGRFMAG1']]
#     df4_scaled[['LINE','IGRFMAG1']] = df4[['LINE','IGRFMAG1']]
#     df6_scaled[['LINE','IGRFMAG1']] = df6[['LINE','IGRFMAG1']]
#     df7_scaled[['LINE','IGRFMAG1']] = df7[['LINE','IGRFMAG1']]

                #     print('Data scaling Done\n')
    
    # Train, Validation, Test set
    df_concat = pd.concat([df2,df3,df4,df6,df7],ignore_index=True,axis=0)

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
    
    print('Data split Done')
    
    # Model
    model = CNN(SEQ_LEN).to(DEVICE)
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
    
    if args.shut == True:
        os.system("shutdown")
    
