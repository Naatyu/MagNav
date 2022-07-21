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


class CNN(nn.Module):

    
    def __init__(self, trial, num_conv_layers, num_filters, num_neurons, drop_conv1, drop_fc1, seq_len):

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


def objective(trial):
    
    num_conv_layers = trial.suggest_int("num_conv_layers",2,3)                        # Number of convolutional layers
    num_filters     = [int(trial.suggest_discrete_uniform(
                      f"num_filter_{i}",8,128,4)) for i in range(num_conv_layers)]    # Number of filters for the conv layers
    num_neurons     = [int(trial.suggest_discrete_uniform(
                      f"num_neurons_{i}",4,512,4)) for i in range(2)]                # Number of neurons for the FC layers
    drop_conv1      = trial.suggest_float("drop_conv1",0,0.5)                         # Drop for 2nd conv layer
    drop_fc1        = trial.suggest_float("drop_fc1",0,0.5)                           # Drop for 1st FC layer
    seq_len         = int(trial.suggest_discrete_uniform("seq_len",15,300,5))         # Length of a sequence
    
    model = CNN(trial, num_conv_layers, num_filters,
                num_neurons, drop_conv1, drop_fc1, seq_len).to(device)                # Generate the model
    
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
    
    n_epochs = 25                                                                     # Number of training epochs
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