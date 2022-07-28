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
            if k == self.num_linear-1:
                out = linear_k(out)
                return out
            
            out = F.relu(linear_k(out))
    

def train(network, optimizer, seq_len):
    
    train_data = MagNavDataset(df_concat, seq_length=seq_len, split='train')          # Train data
    train_loader  = DataLoader(train_data,                                            # Train data loader
                               batch_size=batch_size_train,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=False)
    
    network.train()                                                                   # Set the module in training mode
    
    for batch_i, (data, target) in enumerate(train_loader):
        
        if batch_i * batch_size_train > number_of_train_examples:                     # Limit training data for faster computation
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


def validate(network,seq_len):
    
    val_data   = MagNavDataset(df_concat, seq_length=seq_len, split='val')            # Validation data
    


    val_loader    = DataLoader(val_data,                                              # Validation data loader
                               batch_size=batch_size_val,
                               shuffle=False,
                               num_workers=0,
                               pin_memory=False,
                               drop_last=True)
    
    network.eval()                                                                    # Set module in evaluation mode
    preds = []
    truth = []
    
    with torch.no_grad():                                                             # Disable gradient calculation
        for batch_i, (data, target) in enumerate(val_loader):
            
            if batch_i * batch_size_val > number_of_val_examples:                     # Limit validation data for faster computation
                break
            
            preds.append(network(data.to(device)))                                    # Forward propagation
            truth.append(target.to(device))                                           # Collecting truth data
            
    preds = torch.cat(preds,dim=1)                                                    # Unification of sequences
    truth = torch.cat(truth,dim=1)
    validation_RMSE = torch.sqrt(F.mse_loss(preds,truth))                             # Compute RMSE
    validation_SNR = compute_SNR(truth.cpu().numpy(),preds.cpu().numpy())
    
    return validation_RMSE

# (self, seq_len, drop_lstm1, hidden_size, num_layers, num_LSTM, num_linear, num_neurons):
def objective(trial):
    
    num_LSTM    = trial.suggest_int('num_lstm_layers',1,10)                           # Number of LSTM layers
    hidden_size = [int(trial.suggest_discrete_uniform(
                      f"hidden_size_{i}",4,512,4)) for i in range(num_LSTM)]          # Hidden size by lstm layers
    num_layers  = [int(trial.suggest_discrete_uniform(
                      f"layers_lstm_{i}",1,20,1)) for i in range(num_LSTM)]           # Layers by lstm layers
    num_linear  = trial.suggest_int("num_linear_layers",1,3)                          # Number of fully connected layers
    num_neurons = [int(trial.suggest_discrete_uniform(
                      f"num_neurons_{i}",4,1024,4)) for i in range(num_linear)]       # Number of neurons for the FC layers
    drop_lstm1  = trial.suggest_float("drop_lstm1",0,0.5)                             # Drop for 1st LSTM layer
    seq_len     = int(trial.suggest_discrete_uniform("seq_len",5,400,5))              # Length of a sequence
    
    model = LSTM(seq_len, drop_lstm1, hidden_size, num_layers, 
                 num_LSTM, num_linear, num_neurons).to(device)                        # Generate the model
    
    optimizer_name = trial.suggest_categorical("optimizer",["Adam","RMSprop","SGD"])  # Optimizers
    lr             = trial.suggest_float("lr", 1e-6, 1e-1, log=True)                  # Learning rates
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)             # Optimizer set up
    
    for epoch in range(n_epochs):
        train(model, optimizer, seq_len)                                              # Training of the model
        RMSE = validate(model, seq_len)                                               # Evaluate the model
        
        trial.report(RMSE, epoch)                                                     # Report values
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()                                     # Prune training (if it is not promising)
            
    return RMSE


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
            
            # Keeping only 1003, 1002, 1004 and 1006 flight sections for training
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
            
            # Selecting 1007 for validation except 1007.06
            val_sections = df7.LINE.unique().tolist()
            val_sections.remove(1007.06)
            self.sections = val_sections
            
            mask_val = pd.Series(dtype=bool)
            for line in val_sections:
                mask  = (df.LINE == line)
                mask_val = mask|mask_val
            
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


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             # Use GPU if available for faster computation
    
    n_epochs = 30                                                                     # Number of training epochs
    batch_size_train = 64                                                             # batch size for training
    batch_size_val = 64                                                              # batch size for validation
    number_of_trials = 200                                                            # Number of Optuna trials
    limit_obs = False                                                                 # Limit number of observations for faster computation
    
    
    if limit_obs:
        number_of_train_examples = 1500 * batch_size_train                            # Max of train observations
        number_of_val_examples = 5 * batch_size_val                                   # Max of validation observations
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