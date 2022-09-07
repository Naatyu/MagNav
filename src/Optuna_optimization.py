#!/usr/bin/env python3

#####################
#  NOT WORKING NOW  #
#####################

import os
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
from optuna.trial import TrialState
from datetime import datetime
import joblib
import math
from tqdm import tqdm

from models.CNN import Optuna_CNN


def train(network, optimizer, seq_len, n_fold, batch_size):
    
    train_data = MagNavDataset(df_concat, seq_length=seq_len, n_fold=n_fold,split='train') # Train data
    train_loader  = DataLoader(train_data,                                            # Train data loader
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False)
    
    network.train()                                                                   # Set the module in training mode
    
    for batch_i, (data, target) in enumerate(train_loader):
        
        if batch_i * batch_size > number_of_train_examples:                     # Limit training data for faster computation
            break

        optimizer.zero_grad()                                                         # Clear gradients
        output = network(data.to(device))                                             # Forward propagration
        loss = F.mse_loss(output, target.to(device))                                  # Compute loss (Mean Squared Error)
        loss.backward()                                                               # Compute gradients
        optimizer.step()                                                              # Update weights


def validate(network,seq_len, n_fold, batch_size):
    
    val_data   = MagNavDataset(df_concat, seq_length=seq_len, n_fold=n_fold,split='test') # Validation data
    


    val_loader    = DataLoader(val_data,                                              # Validation data loader
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=False,
                               drop_last=True)
    
    network.eval()                                                                    # Set module in evaluation mode
    preds = []
    truth = []
    
    with torch.no_grad():                                                             # Disable gradient calculation
        for batch_i, (data, target) in enumerate(val_loader):
            
            if batch_i * batch_size > number_of_val_examples:                     # Limit validation data for faster computation
                break
            
            preds.append(network(data.to(device)))                                    # Forward propagation
            truth.append(target.to(device))                                           # Collecting truth data
            
    preds = torch.cat(preds,dim=1)                                                    # Unification of sequences
    truth = torch.cat(truth,dim=1)
    validation_RMSE = torch.sqrt(F.mse_loss(preds,truth))                             # Compute RMSE
    validation_SNR = compute_SNR(truth.cpu().numpy(),preds.cpu().numpy())
    
    return validation_RMSE


def objective(trial):
    
    num_conv_layers = trial.suggest_int("num_conv_layers",1,4)                        # Number of convolutional layers
    num_filters     = [int(trial.suggest_discrete_uniform(
                      f"num_filter_{i}",4,128,4)) for i in range(num_conv_layers)]    # Number of filters for the conv layers
    num_neurons     = [int(trial.suggest_discrete_uniform(
                      f"num_neurons_{i}",4,512,4)) for i in range(2)]                # Number of neurons for the FC layers
    seq_len         = int(trial.suggest_discrete_uniform("seq_len",15,300,5))         # Length of a sequence
    
    
    
    optimizer_name = trial.suggest_categorical("optimizer",["Adam","RMSprop","SGD"])  # Optimizers
    lr             = trial.suggest_float("lr", 1e-6, 1e-1, log=True)                  # Learning rates
    
    n_epochs = trial.suggest_int("n_epochs",2,50)
    batch_size = int(trial.suggest_discrete_uniform("batch_size",32,2048,32))
    
    fold_RMSE = []
    
    for n_fold in range(3):
        model = Optuna_CNN(seq_len, 11, num_conv_layers, num_filters,num_neurons).to(device) # Generate the model
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)             # Optimizer set up
        for epoch in range(n_epochs):
            train(model, optimizer, seq_len, n_fold, batch_size)                         # Training of the model
            RMSE = validate(model, seq_len, n_fold, batch_size)                           # Evaluate the model
        
        fold_RMSE.append(RMSE)
        trial.report(RMSE, epoch)                                                     # Report values
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()                                     # Prune training (if it is not promising)
            
    total_RMSE = sum(fold_RMSE)/2

    return total_RMSE


def trim_data(data,seq_length):
    # Remove excessive data that cannot be in a full sequence
    if (len(data)%seq_length) != 0:
        data = data[:-(len(data)%seq_length)]
    else:
        pass
        
    return data


class MagNavDataset(Dataset):
    '''
    Transform Pandas dataframe of flights data into a custom PyTorch dataset that returns the data into sequences of a desired length.
    '''
    def __init__(self, df, seq_len, split, train_lines, test_lines,truth='IGRFMAG1'):
        '''
        Initialization of the dataset.
        
        Arguments:
        - `df` : dataframe to transform in a custom PyTorch dataset
        - `seq_len` : length of a sequence
        - `split` : data split ('train' or 'test')
        - `train_lines` : flight lines used for training
        - `test_lines` : flight lines used for testing
        - `truth` : ground truth used as a reference for training the model ('IGRFMAG1' or 'COMPMAG1')
        
        Returns:
        - None
        '''
        self.seq_len  = seq_len
        self.features = df.drop(columns=['LINE',truth]).columns.to_list()
        self.train_sections = train_lines
        self.test_sections = test_lines
        
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


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             # Use GPU if available for faster computation
    
    number_of_trials = 200                                                            # Number of Optuna trials
    limit_obs = False                                                                 # Limit number of observations for faster computation
    
    
    if limit_obs:
        number_of_train_examples = 1500 * batch_size                            # Max of train observations
        number_of_val_examples = 5 * batch_size                                   # Max of validation observations
    else:
        number_of_train_examples = 9e6
        number_of_val_examples = 9e6
    
    
    random_seed = 27                                                                  # Make runs repeatable
    torch.backends.cudnn.enable = False                                               # Disable cuDNN use of nondeterministic algorithms (for repeatability)
    torch.manual_seed(random_seed)                                                    # Set torch seed

    #----Import data----#
    
    flights = {}
    
    # Flights to import
    flights_num = [2,3,4,6,7]
    for n in flights_num:
        df = pd.read_hdf('./data/processed/Flt_data.h5', key=f'Flt100{n}')
    flights[n] = df

    train_lines = [np.concatenate([flights[2].LINE.unique(),flights[3].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique()]).tolist(),
                   np.concatenate([flights[2].LINE.unique(),flights[4].LINE.unique(),flights[6].LINE.unique(),flights[7].LINE.unique()]).tolist()]
    test_lines  = [flights[7].LINE.unique().tolist(),
                   flights[3].LINE.unique().tolist()]

    #----Apply Tolles-Lawson----#
    
    # Get cloverleaf pattern data
    mask = (flights[2].LINE == 1002.20)
    tl_pattern = flights[2][mask]

    # filter parameters
    fs      = 10.0
    lowcut  = 0.1
    highcut = 0.9
    filt    = ['Butterworth',4]
    
    ridge = 0.025

    for n in tqdm(flights_num):

        # A matrix of Tolles-Lawson
        A = magnav.create_TL_A(flights[n]['FLUXB_X'],flights[n]['FLUXB_Y'],flights[n]['FLUXB_Z'])

        # Tolles Lawson coefficients computation
        TL_coef_2 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG2'],
                                        lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_3 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG3'],
                                        lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_4 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG4'],
                                        lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)
        TL_coef_5 = magnav.create_TL_coef(tl_pattern['FLUXB_X'],tl_pattern['FLUXB_Y'],tl_pattern['FLUXB_Z'],tl_pattern['UNCOMPMAG5'],
                                        lowcut=lowcut,highcut=highcut,fs=fs,filter_params=filt,ridge=ridge)

        # Magnetometers correction
        flights[n]['TL_comp_mag2_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG2'].tolist(),(-1,1)), TL_coef_2, A)
        flights[n]['TL_comp_mag3_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG3'].tolist(),(-1,1)), TL_coef_3, A)
        flights[n]['TL_comp_mag4_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG4'].tolist(),(-1,1)), TL_coef_4, A)
        flights[n]['TL_comp_mag5_cl'] = magnav.apply_TL(np.reshape(flights[n]['UNCOMPMAG5'].tolist(),(-1,1)), TL_coef_5, A)

    #----Apply IGRF and diurnal corrections----#

    flights_cor = {}
    mags_to_cor = ['TL_comp_mag4_cl', 'TL_comp_mag5_cl']

    for n in tqdm(flights_num):
        flights_cor[n] = apply_corrections(flights[n], mags_to_cor, diurnal=True, igrf=True)
    
    #----Select features----#
    
    # Always keep the 'LINE' feature in the feature list so that the MagNavDataset function can split the flight data
    features = [mags_to_cor[0],mags_to_cor[1],'V_BAT1','V_BAT2',
                    'INS_VEL_N','INS_VEL_V','INS_VEL_W','CUR_IHTR','CUR_FLAP','CUR_ACLo','CUR_TANK','PITCH','ROLL','AZIMUTH','BARO','LINE',TRUTH]
    
    dataset = {}
    
    for n in flights_num:
        dataset[n] = flights_cor[n][features]
    
    del flights_cor
    print(f'Feature selection done. Memory used {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.2f} Mb')
    
    #----Data scaling----#

    # Save scaling parameters
    scaling = {}
    df = pd.DataFrame()
    for flight in flights_num:
        df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)
    for n in range(len(test_lines)):
        mask = pd.Series(dtype=bool)
        for line in test_lines[n]:
            temp_mask = (df.LINE == line)
            mask = temp_mask|mask
        scaling[n] = ['std', df.loc[mask,TRUTH].mean(), df.loc[mask,TRUTH].std()]
    del mask, temp_mask, df
    
    # Apply Standard scaling to the dataset
    for n in tqdm(flights_num):
        dataset[n] = Standard_scaling(dataset[n])
    df = pd.DataFrame()
    for flight in flights_num:
        df = pd.concat([df,dataset[flight]], ignore_index=True, axis=0)

    
    date = datetime.strftime(datetime.now(),'%y%m%d_%H%M%S')
    study_name=f'study_{date}'
    study = optuna.create_study(direction='minimize')                                 # Create Optuna study to minimize RMSE
    study.optimize(objective, n_trials=number_of_trials)
    
    joblib.dump(study,f'./models/{study_name}.pkl')                                   # Save study for visualization
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])      # Number of pruned trials
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])  # Number of completed trials
    
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    trial = study.best_trial                                                          # Get best trial
    print("Best trial:")                                                              # Display best trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
        
    most_important_parameters = optuna.importance.get_param_importances(study, target=None) # Most important hyperparameters
    
    print("\nMost important parameters:")                                             # Display most important hyperparameters
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))