import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset


'''
def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device
'''

def try_gpu(i=0):
    return torch.device(f'cuda:{i}' if torch.cuda.device_count() >= i + 1 else "cpu")

def update_loss_params(params):
    loss_params = {}
    items_required = ['edl_used', 'class_n']
    if params['edl_used'] != 0:
        items_required.extend(['kl', 'edl_fun', 'evi_fun'])
        if params['edl_used'] == 2:
            items_required.append('annealing_step')
        elif params['edl_used'] == 3:
            items_required.append('l')
    loss_params.update(
            {item: params.get(item) for item in items_required})
    return loss_params


def load_data(data_path, sb_n, day_list, time_list, trial_list, **kwargs):
    arr_default = {'tcn_used': False, 'batch_size': 128, 'shuffle': True, 'drop_last': True, 'num_workers': 2, 'pin_memory': True}

    for key, item in kwargs.items():
        if key in arr_default:
            arr_default[key] = item
    
    # Specify for Ninapro DB6
    X = []  # L*1*14(channels)*500(samples)
    Y = []
    for day_n in day_list:
        for time_n in time_list:
            for trial_n in trial_list:
                temp = pd.read_pickle(
                    os.getcwd() + data_path + f"S{sb_n}_D{day_n}_T{time_n}_t{trial_n}.pkl")
                X.extend(temp['x'])
                Y.extend(temp['y'])
    X_torch = torch.from_numpy(np.array(X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_torch = torch.from_numpy(np.array(Y, dtype=np.int64))
    if arr_default['tcn_used']:
        X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
    data = TensorDataset(X_torch, Y_torch)

    if arr_default['batch_size'] > 1:  # For training and validation
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=arr_default['batch_size'], shuffle=arr_default['shuffle'], drop_last=arr_default['drop_last'], num_workers=arr_default['num_workers'], pin_memory=arr_default['pin_memory'])#, prefetch_factor=128, persistent_workers=True)
    elif arr_default['batch_size'] == 1:  # For testing
        # default DataLoader: batch_size = 1, shuffle = False, drop_last =False
        data_loader = torch.utils.data.DataLoader(data)
    return data_loader


def load_data_test(data_path, sb_n, day_n, time_n, trial_n, tcn_used=False):
    # Specify for Ninapro DB6
    
    #X = [] # tensor
    #Y = [] # numpy

    temp = pd.read_pickle(
        os.getcwd() + data_path + f"S{sb_n}_D{day_n}_T{time_n}_t{trial_n}.pkl")

    X_torch = torch.from_numpy(np.array(temp['x'], dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    if tcn_used:
        X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
    Y_numpy = np.array(temp['y'], dtype=np.int64)
    
    return X_torch, Y_numpy
