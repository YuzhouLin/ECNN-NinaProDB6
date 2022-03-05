import pickle
import scipy.io as io
import numpy as np
import os
import argparse


# DATA_PATH = '../../hy-tmp/Data6' # modify it for your own
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data_path', type=str, 
    help='the place to save data')


args = parser.parse_args()
DATA_PATH = args.data_path


# to segment the data with a sliding window
def prepare_data(data_params):
    fs = data_params['fs']
    wl = round(data_params['ws']/1000*fs)
    # window length (ms)/1000 * fs = samples (int)
    # ratio_non_overlap [0-1]
    # if ratio_non_overlap = 0.1, size_non_overlap = wl*0.1
    for sb_n in data_params['sb_list']:
        for day_n in data_params['day_list']:
            for time_n in data_params['time_list']:
                data_path = DATA_PATH + f'/s{sb_n}/S{sb_n}_D{day_n}_T{time_n}.mat'
                emg = io.loadmat(data_path)['emg']
                label = io.loadmat(data_path)['restimulus']
                cycle = io.loadmat(data_path)['rerepetition'] # cycle(trial)
                valid_channel = np.where(emg.any(axis=0))[0] # only 14 valid channels in this case
                for k in data_params['trial_list']:
                    X = []
                    Y = []
                    temp_file = DATA_PATH + f'{data_params["folder"]}S{sb_n}_D{day_n}_T{time_n}_t{k}.pkl'
                    if os.path.isfile(temp_file):
                        continue
                    # S:subject; D: day; T: time; t: trial
                    # add relative path later
                    new_class_label = 0
                    for m in data_params['class_list']:
                        if m == 0:
                            samples = emg[np.nonzero(np.logical_and(cycle==k, label==m))[0]][:6*fs,valid_channel] # take only first 6s data for the 'rest'
                        else:
                            samples = emg[np.nonzero(np.logical_and(cycle==k, label==m))[0]][:,valid_channel]
                        temp = []
                        for n in range(0,samples.shape[0]-wl, round(wl*data_params['ratio_non_overlap'])):
                            segdata = samples[n:n+wl,:] # 500*14
                            temp.append(np.expand_dims(segdata, axis=0))
                        X.extend(temp)
                        Y.extend(new_class_label+np.zeros(len(temp)))
                        new_class_label+=1
                    temp_dict = {'x': X, 'y':Y}
                    f = open(temp_file,"wb")
                    pickle.dump(temp_dict,f)
                    f.close()
    return

if __name__ == "__main__":
    data_params = {}
    data_params['data_num'] = 6 # dataset number
    data_params['day_list'] = list(range(1,6)) # day list, 5 days
    data_params['sb_list'] = [2] # subject list; 
    # data_params['sb_list'] = list(range(1, 11)) change it later
    data_params['class_list'] = [0, 1, 3, 4, 6, 9, 10, 11] # class list
    data_params['channel_list'] = list(range(1,16)) # channel list
    data_params['trial_list'] = list(range(1, 13)) # trial list
    data_params['time_list'] = [1, 2] # 1: morning; 2: afternoon  
    data_params['fs'] = 2000 # sampling frequency (Hz)
    data_params['ws'] = 200 # 200 ms 
    data_params['ratio_non_overlap'] = 0.1
    data_params['folder'] = '/Processed/'

    # sliding window length: wl (Unit: samples)
    # example:
    # if taking 250ms as a window and sampling frequency of data is 200Hz,
    # wl = (200Hz/1000ms) * (250ms) = 50 samples
    # ratio_non_overlap [0-1]
    # if ratio_non_overlap = 0.1, size_non_overlap = wl*0.1
    # Save the data used for deep learning
    prepare_data(data_params)
