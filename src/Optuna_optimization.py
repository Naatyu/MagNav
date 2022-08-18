#!/usr/bin/env python3

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
        
        
def compute_SNR(truth_mag,pred_mag):
    
    error = pred_mag - truth_mag
    std_truth = np.std(truth_mag)
    std_error = np.std(error)
    
    SNR = std_truth / std_error
    
    return SNR  


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
            
    total_RMSE = sum(fold_RMSE)/3

    return total_RMSE


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
        
        train_fold_2 = np.concatenate([df4.LINE.unique(),df6.LINE.unique(),df7.LINE.unique(),df2.LINE.unique()]).tolist()
        test_fold_2  = df3.LINE.unique().tolist()
        
        if n_fold == 0:
            self.train_sections = train_fold_0
            self.test_sections = test_fold_0
        elif n_fold == 1:
            self.train_sections = train_fold_1
            self.test_sections = test_fold_1
        
        
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
    
    
    df2 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1002')              # Import flight 1002
    df3 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1003')              # Import flight 1003
    df4 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1004')              # Import flight 1004
    df6 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1006')              # Import flight 1006
    df7 = pd.read_hdf('./data/processed/Chall_dataset.h5', key=f'Flt1007')              # Import flight 1007
    
    df_concat = pd.concat([df2,df3,df4,df6,df7], ignore_index=True, axis=0)           # Concatenate data
    
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