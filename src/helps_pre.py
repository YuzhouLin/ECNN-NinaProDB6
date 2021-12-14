import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


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


def load_data_cnn(data_path, sb_n, trial_list, batch_size):
    X = []  # L*1*16(channels)*50(samples)
    Y = []
    for trial_n in trial_list:
        temp = pd.read_pickle(
            os.getcwd() + data_path + f"sb{sb_n}_trial{trial_n}.pkl")
        X.extend(temp['x'])
        Y.extend(temp['y'])
    data = TensorDataset(
        torch.from_numpy(np.array(X, dtype=np.float32)).permute(0, 1, 3, 2),
        torch.from_numpy(np.array(Y, dtype=np.int64)))
    if batch_size > 1:  # For training and validation
        data_loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=True, drop_last=True)
    elif batch_size == 1:  # For testing
        # default DataLoader: batch_size = 1, shuffle = False, drop_last =False
        data_loader = torch.utils.data.DataLoader(data)
    return data_loader


def load_data_test_cnn(data_path, sb_n, trial_n):
    X = []  # L*1*16(channels)*50(samples)
    Y = []

    temp = pd.read_pickle(
        os.getcwd() + data_path + f'sb{sb_n}_trial{trial_n}.pkl')
    X.extend(temp['x'])
    Y.extend(temp['y'])

    X = torch.as_tensor(
        torch.from_numpy(np.array(X))).permute(0, 1, 3, 2)
    # L*1*16*50
    # Y = torch.from_numpy(np.array(Y, dtype=np.int64))
    Y = np.array(Y, dtype=np.int64)
    # X: tensor; Y: numpy
    return X, Y
