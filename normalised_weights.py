import argparse
import pickle
import scipy.io as io
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--sb', type=int)

args=parser.parse_args()


if __name__ == "__main__":
    sb_n = args.sb
    DATA_PATH = f'../../hy-nas/Data6/s{sb_n}'

    train_folder = DATA_PATH+'/train'
    W_train = np.load(train_folder+'/W.npy')
    Y_train = np.load(train_folder+'/Y.npy')

    val_folder = DATA_PATH+'/val'
    W_val = np.load(val_folder+'/W.npy')
    Y_val=np.load(val_folder+'/Y.npy')

    for i in range(8):
        W_train[Y_train==i] = (W_train[Y_train==i] - 1.8)/(W_train[Y_train==i].max()-1.8)
        W_val[Y_val==i] = (W_val[Y_val==i] - 1.8)/(W_val[Y_val==i].max()-1.8)

    np.save(train_folder+'W_n.npy')
    np.save(val_folder+'W_n.npy')