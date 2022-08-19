#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import argparse
import warnings
import os
from datetime import datetime
import time
import math

from models.CNN import CNN
from models.RNN import Optuna_LSTM
import magnav


#-----------------#
#----Functions----#
#-----------------#
    
    
def trim_data(data, seq_len):
    '''
    Delete part of the training data so that the remainder of the Euclidean division between the length of the data and the size of a sequence is 0. This ensures that all sequences are complete.
    
    Arguments:
    - `data` : data that needs to be trimmed
    - `seq_len` : lenght of a sequence
    
    Returns:
    - `data` : trimmed data
    '''
    if (len(data)%seq_len) != 0:
        data = data[:-(len(data)%seq_len)]
    else:
        pass
    
    return data
    
    
class MagNavDataset(Dataset):
    '''
    Transform Pandas dataframe of flights data into a custom PyTorch dataset that returns the data into sequences of a desired length.
    '''
    def __init__(self, df, seq_len, n_fold, split, truth='IGRFMAG1'):
        '''
        Initialization of the dataset.
        
        Arguments:
        - `df` : dataframe to transform in a custom PyTorch dataset
        - `seq_len` : length of a sequence
        - `n_fold` : fold number
        - `split` : data split ('train' or 'test')
        - `truth` : ground truth used as a reference for training the model ('IGRFMAG1' or 'COMPMAG1')
        
        Returns:
        - None
        '''
        self.seq_len  = seq_len
        self.features = df.drop(columns=['LINE',truth]).columns.to_list()
        
        # Fold 0 - flights 1002, 1003, 1004 and 1006 for training and flight 1007 for testing
        train_fold_0 = np.concatenate([df2.LINE.unique(),df3.LINE.unique(),df4.LINE.unique(),df6.LINE.unique()]).tolist()
        test_fold_0  = df7.LINE.unique().tolist()
        
        # Fold 1 - flights 1002, 1004, 1006 and 1007 for training and flight 1003 for testing
        train_fold_1 = np.concatenate([df4.LINE.unique(),df6.LINE.unique(),df7.LINE.unique(),df2.LINE.unique()]).tolist()
        test_fold_1  = df3.LINE.unique().tolist()
        
        if n_fold == 0:
            self.train_sections = train_fold_0
            self.test_sections  = test_fold_0
        elif n_fold == 1:
            self.train_sections = train_fold_1
            self.test_sections  = test_fold_1
        
        if split == 'train':
            
            # Create a mask to keep only training data
            mask_train = pd.Series(dtype=bool)
            for line in self.train_sections:
                mask = (df.LINE == line)
                mask_train = mask|mask_train
            
            # Split in X, y for training
            X_train = df.loc[mask_train,self.features]
            y_train = df.loc[mask_train,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_train.to_numpy(),dtype=torch.float32),seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_train.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)
            
        elif split == 'test':
            
            # Create a mask to keep only testing data
            mask_test = pd.Series(dtype=bool)
            for line in self.test_sections:
                mask = (df.LINE == line)
                mask_test = mask|mask_test
            
            # Split in X, y for testing
            X_test = df.loc[mask_test,self.features]
            y_test = df.loc[mask_test,truth]
            
            # Trim data and convert it to torch tensor
            self.X = torch.t(trim_data(torch.tensor(X_test.to_numpy(),dtype=torch.float32),seq_len))
            self.y = trim_data(torch.tensor(np.reshape(y_test.to_numpy(),[-1,1]),dtype=torch.float32),seq_len)

    def __getitem__(self, idx):
        '''
        Return a sequence for a given index.
        
        Arguments:
        - `idx` : index of a sequence
        
        Returns:
        - `X` : sequence of features
        - `y` : ground truth corresponding to the sequence
        '''
        X = self.X[:,idx:(idx+self.seq_len)]
        y = self.y[idx+self.seq_len-1]
        return X, y
    
    def __len__(self):
        '''
        Return the numbers of sequences in the dataset.
        
        Arguments:
        -None
        
        -Returns:
        -number of sequences in the dataset
        '''
        return len(torch.t(self.X))-self.seq_len


def make_training(model, epochs, train_loader, test_loader, scaling=['None']):
    '''
    PyTorch training loop with testing.
    
    Arguments:
    - `model` : model to train
    - `epochs` : number of epochs to train the model
    - `train_loader` : PyTorch dataloader for training
    - `test_loader` : PyTorch dataloader for testing
    - `scaling` : (optional) scaling parameters
    
    Returns:
    - `train_loss_history` : history of loss values during training
    - `test_loss_history` : history of loss values during testing
    '''
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=1.5e-4)
    
    # Create batch and epoch progress bar
    batch_bar = tqdm(total=len(train)//BATCH_SIZE,unit="batch",desc='Training',leave=False, position=0, ncols=150)
    epoch_bar = tqdm(total=epochs,unit="epoch",desc='Training',leave=False, position=1, ncols=150)
    
    train_loss_history = []
    test_loss_history = []

    for epoch in range(epochs):

        #----Train----#

        train_running_loss = 0.

        # Turn on gradients computation
        model.train()
        
        batch_bar.reset()
        
        # Enumerate allow to track batch index and intra-epoch reporting 
        for batch_index, (inputs, labels) in enumerate(train_loader):
            
            # Put data to the desired device (CPU or GPU)
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Make predictions for this batch
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions, labels)

            # Zero gradients of optimizer for every batch
            optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            train_running_loss += loss.item()
            
            # Update batch progess bar
            batch_bar.set_postfix(train_loss=train_running_loss/(batch_index+1),lr=optimizer.param_groups[0]['lr'])
            batch_bar.update()

        # Compute the loss of the batch and save it
        train_loss = train_running_loss / batch_index
        train_loss_history.append(train_loss)

        #----Test----#

        test_running_loss = 0.
        preds = []
        
        # Disable layers specific to training such as Dropout/BatchNorm
        model.eval()
        
        # Turn off gradients computation
        with torch.no_grad():
            
            # Enumerate allow to track batch index and intra-epoch reporting
            for batch_index, (inputs, labels) in enumerate(test_loader):

                # Put data to the desired device (CPU or GPU)
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                preds.append(model(inputs).cpu())

                # Make prediction for this batch
                predictions = model(inputs)

                # Compute the loss
                loss = criterion(predictions, labels)

                # Gather data and report
                test_running_loss += loss.item()

            # Compute the loss of the batch and save it
            preds = np.concatenate(preds)
            RMSE = magnav.rmse(preds,test.y[SEQ_LEN:],False)
            test_loss = test_running_loss / batch_index
            test_loss_history.append(test_loss)

        # Update epoch progress bar
        epoch_bar.set_postfix(train_loss=train_loss,test_loss=test_loss,RMSE=RMSE,lr=optimizer.param_groups[0]['lr'])
        epoch_bar.update()
    print('\n')
    
    return train_loss_history, test_loss_history


def Standard_scaling(df):
    '''
    Apply standardization (Z-score normalization) to a pandas dataframe except for the 'LINE' feature.
    
    Arguments:
    - `df` : dataframe to standardize
    
    Returns:
    - `df_scaled` : standardized dataframe
    '''
    df_scaled = (df-df.mean())/df.std()
    df_scaled['LINE'] = df['LINE']

    return df_scaled
    
    
def MinMax_scaling(df, bound=[-1,1]):
    '''
    Apply min-max scaling to a pandas dataframe except for the 'LINE' feature.

    Arguments:
    - `df` : dataframe to standardize
    - `bound` : (optional) upper and lower bound for min-max scaling
    
    Returns:
    - `df_scaled` : scaled dataframe
    '''
    df_scaled = bound[0] + ((bound[1]-bound[0])*(df-df.min())/(df.max()-df.min()))
    df_scaled['LINE'] = df['LINE']
    
    return df_scaled


def apply_corrections(df,mags_to_cor,diurnal=True,igrf=True):
    '''
    Apply IGRF and/or diurnal corrections on data.
    
    Arguments:
    - `df` : dataframe to correct
    - `mags_to_cor` : list of string of magnetometers to be corrected
    - `diurnal` : (optional) apply diunal correction (True or False)
    - `igrf` : (optional) apply IGRF correction (True or False)
    
    Returns:
    - `df_cor` : corrected dataframe
    '''
    mag_measurements = np.array(mags_to_cor)
    df_cor = df.copy()
    
    # Diurnal cor
    if diurnal == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(df_cor['DIURNAL'].values,[-1,1])
    
    # IGRF cor
    lat  = df_cor['LAT']
    lon  = df_cor['LONG']
    h    = df_cor['BARO']*1e-3
    date = datetime.datetime(2020, 6, 29) # Date on which the flights were made
    Be, Bn, Bu = magnav.igrf(lon,lat,h,date)
    
    if igrf == True:
        df_cor[mag_measurements] = df_cor[mag_measurements]-np.reshape(np.sqrt(Be**2+Bn**2+Bu**2)[0],[-1,1])

    return df_cor

    
#------------#
#----Main----#
#------------#


if __name__ == "__main__":
    
    # Start timer
    start_time = time.time()
    
    # set seed for reproducibility
    torch.manual_seed(27)
    random.seed(27)
    np.random.seed(27)
    
    #----User arguments----#
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d","--device", type=str, required=False, default='cuda', help="Which GPU to use (cuda or cpu), default='cuda'. Ex : --device 'cuda' ", metavar=""
    )
    parser.add_argument(
        "-e","--epochs", type=int, required=False, default=35, help="Number of epochs to train the model, default=35. Ex : --epochs 200", metavar=""
    )
    parser.add_argument(
        "-b","--batch", type=int, required=False, default=256, help="Batch size for training, default=256. Ex : --batch 64", metavar=""
    )
    parser.add_argument(
        "-sq","--seq", type=int, required=False, default=20, help="Length sequence of data, default=20. Ex : --seq 15", metavar=""
    )
    parser.add_argument(
        "--shut", action="store_true", required=False, help="Shutdown pc after training is done."
    )
    parser.add_argument(
        "-sc", "--scaling", type=int, required=False, default=0, help="Data scaling, 1 for standardization, 2 for MinMax scaling, 0 for no scaling, default=0. Ex : --scaling 0", metavar=''
    )
    parser.add_argument(
        "-cor", "--corrections", type=int, required=False, default=3, help="Data correction, 0 for no corrections, 1 for IGRF correction, 2 for diurnal correction, 3 for IGRF+diurnal correction. Ex : --corrections 3", metavar=''
    )
    
    args = parser.parse_args()
    
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch
    DEVICE     = args.device
    SEQ_LEN    = args.seq
    SCALING    = args.scaling
    COR        = agrs.corrections
    
    print(f'\nCurrently training on {torch.cuda.get_device_name(DEVICE)}')

    #----Import data----#

    # df2 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1002')
    # df3 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1003')
    # df4 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1004')
    # df6 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1006')
    # df7 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1007')
    # df2 = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt1002')
    # df3 = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt1003')
    # df4 = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt1004')
    # df6 = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt1006')
    # df7 = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt1007')
    
    flights = {}
    flights_num = [2,3,4,6,7]
    
    for n in flights_num:
        df = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt100{n}')
        flights[n] = df
    
    #----Apply Tolles-Lawson----#
    
    # Get cloverleaf pattern data
    mask = (flights[2].LINE == 1002.20)
    tl_pattern = flights[2][mask]

    # filter parameters
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]
    
    for n in tqdm(flights_num)
    
        # A matrix of Tolles-Lawson
        A = magnav.create_TL_A(flights[n]['FLUXB_X'],flights[n]['FLUXB_Y'],flights[n]['FLUXB_Z'])

        # Tolles Lawson coefficients computation
        TL_coef_4 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG4'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)
        TL_coef_5 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG5'],
                                      lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt)

        # Magnetometers correction
        flights[n]['TL_comp_mag4_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG4'].tolist(),(-1,1)), TL_coef_4, A)
        flights[n]['TL_comp_mag5_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG5'].tolist(),(-1,1)), TL_coef_5, A)
        
    #----Apply IGRF and diurnal corrections----#
    
    flights_cor = {}
    mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']
    
    if COR == 0:
        flights_cor = flights.copy()
        truth = 'COMPMAG1'
    if COR == 1:
        for n in flights_num:
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=False, igrf=True)
        truth = 'IGRFMAG1'
    if COR == 2: 
        for n in flights_num:
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=False)
        truth = 'DCMAG1'
    if COR == 3:
        for n in flights_num:
            flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
        truth = 'IGRFMAG1'
        
    #----Select features----#
    
    # Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
    features = ['TL_comp_mag3_cl','TL_comp_mag5_cl','V_BAT1','V_BAT2',
                    'INS_ACC_X','INS_ACC_Y','INS_ACC_Z','CUR_IHTR','PITCH','ROLL','AZIMUTH','LINE',truth]
    
    dataset = {}
    
    for n in flights_num:
        dataset[n] = flights_cor[n][features]
    
    #----Data scaling----#
    
    # if SCALING == 0:
    #     df = pd.concat([df2,df3,df4,df6,df7],ignore_index=True,axis=0)
    #     scaling = ['None']
    # elif SCALING == 2:
    #     bound = [-1,1]
    #     df2_scaled = MinMax_scaling(df2, bound=bound)
    #     df3_scaled = MinMax_scaling(df3, bound=bound)
    #     df4_scaled = MinMax_scaling(df4, bound=bound)
    #     df6_scaled = MinMax_scaling(df6, bound=bound)
    #     df7_scaled = MinMax_scaling(df7, bound=bound)
    df = pd.concat([dataset[2],dataset[3],dataset[4],dataset[6],dataset[7]],ignore_index=True,axis=0)
#         scaling = ['minmax',bound[0],bound[1],df3[truth].min(),df3[truth].max()]
    # elif SCALING == 1:
    #     df2_scaled = Standard_scaling(df2)
    #     df3_scaled = Standard_scaling(df3)
    #     df4_scaled = Standard_scaling(df4)
    #     df6_scaled = Standard_scaling(df6)
    #     df7_scaled = Standard_scaling(df7)
    #     df = pd.concat([df2_scaled,df3_scaled,df4_scaled,df6_scaled,df7_scaled],ignore_index=True,axis=0)
    #     scaling = ['std']
    
    #----Training----#
    
    train_loss_history = []
    test_loss_history = []
    last_test = []
    
    for n_fold in range(2):
        
        print('\n--------------------')
        print(f'Fold number {n_fold}')
        print('--------------------\n')
        
        
        train = MagNavDataset(df, seq_len=SEQ_LEN, n_fold=n_fold, split='train', truth='IGRFMAG1')
        test  = MagNavDataset(df, seq_len=SEQ_LEN, n_fold=n_fold, split='test', truth='IGRFMAG1')

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
#         model = CNN(3,[16,32,64],[512,64],0.15,0.15,SEQ_LEN).to(DEVICE)
#         model = CNN(SEQ_LEN,11).to(DEVICE)
        num_LSTM    = 2                                                                   # Number of LSTM layers
        hidden_size = [16,16]          # Hidden size by lstm layers
        num_layers  = [15,5]           # Layers by lstm layers
        num_linear  = 2                          # Number of fully connected layers
        num_neurons = [64,12]       # Number of neurons for the FC layers
        drop_lstm1  = 0                             # Drop for 1st LSTM layer
        model = Optuna_LSTM(SEQ_LEN, drop_lstm1, hidden_size, num_layers, 
                     num_LSTM, num_linear, num_neurons).to(DEVICE)
#         model = ResNet18().to(DEVICE)
        model.name = 'LSTM'

        # Loss
        criterion = torch.nn.MSELoss()

        # Training
        train_loss, test_loss = make_training(model, EPOCHS, train_loader, test_loader)
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        last_test.append(test_loss[-1])
        
    perf_folds = sum(last_test)/2
        
    print('\n-------------------------')
    print('Performance for all folds')
    print('-------------------------')
    print(f'Fold 1 | RMSE = {np.sqrt(last_test[0]):.2f} nT')
    print(f'Fold 2 | RMSE = {np.sqrt(last_test[1]):.2f} nT')
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
    