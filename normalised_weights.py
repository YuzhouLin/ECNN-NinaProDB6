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

def seg_emg(samples, wl=400, ratio_non_overlap=0.1):
    segmented = []
    for n in range(0, samples.shape[0]-wl, round(wl*ratio_non_overlap)):
        segdata = samples[n:n+wl,:] # 400*14
        segmented.append(np.expand_dims(segdata, axis=0))
    return segmented

if __name__ == "__main__":
    sb_n = args.sb
    DATA_PATH = f'../../hy-nas/Data6/s{sb_n}'
    
    train_folder = DATA_PATH+'/train'
    W_train = np.load(train_folder+'/W.npy')
    Y_train = np.load(val_folder+'/Y.npy')
    
    val_folder = DATA_PATH+'/val'
    W_val = np.load(val_folder+'/W.npy')
    Y_val=np.load(val_folder+'/Y.npy')
    

    
