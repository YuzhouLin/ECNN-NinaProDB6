import scipy.io as io
import numpy as np
import os
import yaml


DATA_READ='D://NinaPro/DB6/Data6/'
DATA_SAVE='D://NinaPro/DB6/data/'

def seg_emg(samples, wl=400, ratio_non_overlap=0.1):
    segmented = []
    for n in range(0, samples.shape[0]-wl, round(wl*ratio_non_overlap)):
        segdata = samples[n:n+wl,:] # 400*14
        segmented.append(np.expand_dims(segdata, axis=0))
    return segmented

def data_prepared(sb_n):
    class_list = [0, 1, 3, 4, 6, 9, 10, 11]

    train_folder = DATA_SAVE+f'{sb_n}/train'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    val_folder = DATA_SAVE+f'{sb_n}/val'
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    test_folder = DATA_SAVE+f'{sb_n}/test'
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


    with open(f'./data_distill/sb_{sb_n}', 'r') as f:
        clean_label_list = yaml.load(f, Loader=yaml.SafeLoader)

    min_len = 6000 # ignore trials which are less than 3s recording
    rest_len = 1000 # starts taking rest samples after 0.5s of each grasp
    time_list = [1,2]

    for day_n in [1,2]:
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        for time_n in time_list:
            time = 'AM' if time_n==1 else 'PM'
            data_path = DATA_READ + f's{sb_n}/S{sb_n}_D{day_n}_T{time_n}.mat'
            emg = io.loadmat(data_path)['emg']
            label = io.loadmat(data_path)['restimulus']
            cycle = io.loadmat(data_path)['rerepetition'] # cycle(trial)
            valid_channel = np.where(emg.any(axis=0))[0] # only 14 valid channels in this case
            for g_n in range(1,8):
                sorted_trial_list = np.array(sorted(clean_label_list[f'd{day_n}'][time][f'g{g_n}'], key=lambda x: x[0]))

                #sorted_snr_list = sorted(clean_label_list['d1'][time][f'g{g_n}'], key=lambda x: x[1])
                #val_cycle = [clean_label_list['d1'][time][f'g{g_n}'][0][0], clean_label_list['d1'][time][f'g{g_n}'][-1][0]]
                #m_snr_trial = sorted_snr_list[int(len(sorted_snr_list)/2)][0]
                #val_cycle = [sorted_snr_list[0][0], m_snr_trial, sorted_snr_list[-1][0]]
                #val_cycle = [sorted_snr_list[0][0], sorted_snr_list[-1][0]]
                # for each gesture, the trials with highest snr and lowest snr are used for validation

                # for each gesture, the last two trials are used for validation, one odd and one even for different objects
                val_cycle=[sorted_trial_list[sorted_trial_list%2==1][-1], sorted_trial_list[sorted_trial_list%2==0][-1]]

                for c_n, _ in clean_label_list[f'd{day_n}'][time][f'g{g_n}']:
                    selected_index = np.nonzero(np.logical_and(cycle==c_n, label==class_list[g_n]))[0]
                    samples = emg[selected_index][:,valid_channel]
                    if len(samples) < min_len:
                        print(f'warning trial detected on sb{sb_n}, day{day_n}, {time}, g{g_n}, t{c_n}')
                        continue
                    segment = seg_emg(samples)
                    rest_samples = emg[selected_index[-1]+rest_len:selected_index[-1]+rest_len+round(len(samples)/6)][:,valid_channel]
                    segment_rest = seg_emg(rest_samples)
                    n_segments = [len(segment), len(segment_rest)]
                    if c_n in val_cycle:
                        X_val.extend(segment)
                        Y_val.extend(g_n+np.zeros(n_segments[0]))
                        X_val.extend(segment_rest)
                        Y_val.extend(np.zeros(n_segments[1]))
                    else:
                        X_train.extend(segment)
                        Y_train.extend(g_n+np.zeros(n_segments[0]))
                        X_train.extend(segment_rest)
                        Y_train.extend(np.zeros(n_segments[1]))

            np.save(train_folder + f'/X_d{day_n}_t{time_n}.npy', np.array(X_train, dtype=np.float32))
            np.save(train_folder + f'/Y_d{day_n}_t{time_n}.npy', np.array(Y_train, dtype=np.uint8))

            np.save(val_folder + f'/X_d{day_n}_t{time_n}.npy', np.array(X_val, dtype=np.float32))
            np.save(val_folder + f'/Y_d{day_n}_t{time_n}.npy', np.array(Y_val, dtype=np.uint8))


    for day_n in [3,4,5]:
        for time_n in time_list:
            X_test = []
            Y_test = []
            W_test = []
            time = 'AM' if time_n==1 else 'PM'
            data_path = DATA_READ + f's{sb_n}/S{sb_n}_D{day_n}_T{time_n}.mat'
            emg = io.loadmat(data_path)['emg']
            label = io.loadmat(data_path)['restimulus']
            cycle = io.loadmat(data_path)['rerepetition'] # cycle(trial)
            valid_channel = np.where(emg.any(axis=0))[0] # only 14 valid channels in this case
            for g_n in range(1,8):
                # leave it later for checking if there is enough trials for each gesture
                for c_n, snr in clean_label_list[f'd{day_n}'][time][f'g{g_n}']:
                    selected_index = np.nonzero(np.logical_and(cycle==c_n, label==class_list[g_n]))[0]
                    samples = emg[selected_index][:,valid_channel]
                    if len(samples) < min_len:
                        print(f'warning trial detected on sb{sb_n}, day{day_n}, {time}, g{g_n}, t{c_n}')
                        continue

                    segment = seg_emg(samples)
                    rest_samples = emg[selected_index[-1]+rest_len:selected_index[-1]+rest_len+round(len(samples)/6)][:,valid_channel]
                    #segment = seg_emg(samples[3500:-3500])
                    #rest_samples = emg[selected_index[-1]+3500:selected_index[-1]+round(len(samples[3500:-3500])/7)][:,valid_channel]

                    segment_rest = seg_emg(rest_samples)
                    n_segments = [len(segment), len(segment_rest)]
                    X_test.extend(segment)
                    Y_test.extend(g_n+np.zeros(n_segments[0]))
                    W_test.extend(snr+np.zeros(n_segments[0]))
                    X_test.extend(segment_rest)
                    Y_test.extend(np.zeros(n_segments[1]))
                    W_test.extend(snr+np.zeros(n_segments[1]))

            np.save(test_folder+f'/X_d{day_n}_t{time_n}.npy', np.array(X_test, dtype=np.float32))
            np.save(test_folder+f'/Y_d{day_n}_t{time_n}.npy', np.array(Y_test, dtype=np.uint8))


if __name__ == "__main__":
    for sb_n in [1,3,4,5,6,7,8,10]:
        data_prepared(sb_n)


