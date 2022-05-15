import argparse
from sklearn.metrics import label_ranking_loss
import torch
import utils
import helps_pre as pre
import numpy as np
#import pandas as pd
import os
import yaml
from easydict import EasyDict as edict
import scipy.io as io

parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl (annealing); \
        3: edl with kl (trade-off)')
parser.add_argument('--tcn', action='store_true', default=False,
                    help='to use tcn if it activates, cnn otherwise')

args = parser.parse_args()

EDL_USED = args.edl
TCN_USED = args.tcn
# DEVICE = torch.device('cpu')
DEVICE = pre.try_gpu()


def get_weights(cfg):
    un = "entropy"
    sb_n = cfg.DATA_CONFIG.sb_n
    print('Getting weights for sb: ', sb_n)
    # Load trained model
    checkpoint = torch.load(cfg.test_model)#, map_location=torch.device('cpu'))
    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)

    #model.load_state_dict(
    #    torch.load(checkpoint['model_state_dict'], map_location=DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    train_day_list = list(set(range(1,6))-set(cfg.DATA_CONFIG.day_list))

    folder="train"
    weights = []
    labels = []
    for day_n in train_day_list:
        print('on day: ', day_n)

        for time_n in cfg.DATA_CONFIG.time_list:
            print('on time: ', time_n)

            #for folder in ["train", "val"]:
            test_X = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/X_d{day_n}_t{time_n}.npy')
            Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/Y_d{day_n}_t{time_n}.npy')
            labels.extend(Y_numpy)
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            Y_torch = torch.from_numpy(Y_numpy)
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            test_data = torch.utils.data.TensorDataset(X_torch, Y_torch)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=2048, shuffle=False, drop_last=False, num_workers=4)
            with torch.no_grad():
                for _, (inputs,targets) in enumerate(test_loader):
                    inputs=inputs.to(DEVICE)
                    targets=targets.detach().cpu()
                    #print(targets.size())
                    outputs = model(inputs).detach().cpu()
                    # get results

                    eng = utils.EngineTest(outputs, targets)
                    predict = np.squeeze(eng.get_pred_labels())
                    acti_fun = cfg.HP.evi_fun if EDL_USED else 'softmax'
                    scores = eng.get_scores(acti_fun, EDL_USED)


                    tmp_weights =scores[un]
                    targets = targets.numpy()
                    tmp_weights[predict!=targets]=1-tmp_weights[predict!=targets]
                    #tmp_weights[predict==targets]=tmp_weights[predict==targets]**2

                    weights.extend(tmp_weights)
                    #if EDL_USED:
                    #    un_vac_list.extend(scores['vacuity'])
                    #    un_diss_list.extend(scores['dissonance'])

            #np.save(cfg.DATA_PATH+f's{sb_n}/{folder}/W_d{day_n}_t{time_n}.npy', np.array(weights, dtype=np.float32))

    labels = np.array(labels)

    class_i, class_counts = np.unique(labels, return_counts=True)

    for i in class_i:
        index=np.where(labels[labels==i])
        weights[index] = weights[index]*np.sum(weights[index])/class_counts[i]

    np.save(cfg.DATA_PATH+f's{sb_n}/{folder}/W.npy', np.array(weights, dtype=np.float32))
    return

def get_weights_edl(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    print('Getting weights for sb: ', sb_n)
    # Load trained model
    checkpoint = torch.load(cfg.test_model)#, map_location=torch.device('cpu'))
    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)

    #model.load_state_dict(
    #    torch.load(checkpoint['model_state_dict'], map_location=DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    train_day_list = list(set(range(1,6))-set(cfg.DATA_CONFIG.day_list))

    for day_n in train_day_list:
        print('on day: ', day_n)

        for time_n in cfg.DATA_CONFIG.time_list:
            print('on time: ', time_n)

            for folder in ["train", "val"]:
                test_X = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/X_d{day_n}_t{time_n}.npy')
                Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/Y_d{day_n}_t{time_n}.npy')
                weights = []
                X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
                Y_torch = torch.from_numpy(Y_numpy)
                if TCN_USED:
                    X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
                test_data = torch.utils.data.TensorDataset(X_torch, Y_torch)
                test_loader = torch.utils.data.DataLoader(
                    test_data, batch_size=2048, shuffle=False, drop_last=False, num_workers=4)
                with torch.no_grad():
                    for _, (inputs,targets) in enumerate(test_loader):
                        inputs=inputs.to(DEVICE)
                        targets=targets.detach().cpu()
                        #print(targets.size())
                        outputs = model(inputs).detach().cpu()
                        # get results

                        eng = utils.EngineTest(outputs, targets)
                        predict = np.squeeze(eng.get_pred_labels())
                        acti_fun = cfg.HP.evi_fun if EDL_USED else 'softmax'
                        scores = eng.get_scores(acti_fun, EDL_USED)
                        targets = targets.numpy()
                        #print(np.median(scores['entropy']))
                        #print(np.median(scores['vacuity']))
                        #print(np.median(scores['dissonance']))
                        #exit()

                        #tmp_weights = scores['vacuity']*scores['dissonance']
                        #tmp_weights = scores['vacuity']*scores['dissonance']/10
                        tmp_weights = 1-scores['vacuity']
                        #tmp_weights[predict!=targets]=(1-scores['vacuity'][predict!=targets])*(1-scores['dissonance'][predict!=targets])
                        #tmp_weights[predict!=targets]=(1-scores['vacuity'][predict!=targets])
                        #tmp_weights[predict==targets]=tmp_weights[predict==targets]**2
                        #print(scores['vacuity'][50:55])
                        #print(scores['dissonance'][50:55])
                        #print(tmp_weights[50:55])
                        weights.extend(tmp_weights)
                        #if EDL_USED:
                        #    un_vac_list.extend(scores['vacuity'])
                        #    un_diss_list.extend(scores['dissonance'])
                np.save(cfg.DATA_PATH+f's{sb_n}/{folder}/W_d{day_n}_t{time_n}.npy', np.array(weights, dtype=np.float32))

    return

def seg_emg(samples, wl=400, ratio_non_overlap=0.1):
    segmented = []
    for n in range(0, samples.shape[0]-wl, round(wl*ratio_non_overlap)):
        segdata = samples[n:n+wl,:] # 400*14
        segmented.append(np.expand_dims(segdata, axis=0))
    return segmented

def get_weights_snr(cfg):
    DATA_SAVE = cfg.DATA_PATH
    DATA_READ = cfg.DATA_READ
    sb_n = cfg.DATA_CONFIG.sb_n
    train_folder = DATA_SAVE+f's{sb_n}/train'
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    val_folder = DATA_SAVE+f's{sb_n}/val'
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    with open(f'./data_distill/sb_{sb_n}', 'r') as f:
        clean_label_list = yaml.load(f, Loader=yaml.SafeLoader)

    min_len = 6000 # ignore trials which are less than 3s recording
    rest_len = 1000 # starts taking rest samples after 0.5s of each grasp
    time_list = [1,2]

    for day_n in [1,2]:
        W_train = []
        W_val = []
        for time_n in time_list:
            time = 'AM' if time_n==1 else 'PM'
            data_path = DATA_READ + f's{sb_n}/S{sb_n}_D{day_n}_T{time_n}.mat'
            emg = io.loadmat(data_path)['emg']
            label = io.loadmat(data_path)['restimulus']
            cycle = io.loadmat(data_path)['rerepetition'] # cycle(trial)
            valid_channel = np.where(emg.any(axis=0))[0] # only 14 valid channels in this case
            for g_n in range(1,8):
                sorted_trial_list = np.array(sorted(clean_label_list[f'd{day_n}'][time][f'g{g_n}'], key=lambda x: x[0]))
                val_cycle=[sorted_trial_list[sorted_trial_list%2==1][-1], sorted_trial_list[sorted_trial_list%2==0][-1]]
                for c_n, snr in clean_label_list[f'd{day_n}'][time][f'g{g_n}']:
                    selected_index = np.nonzero(np.logical_and(cycle==c_n, label==cfg.CLASS_LIST[g_n]))[0]
                    samples = emg[selected_index][:,valid_channel]
                    if len(samples) < min_len:
                        print(f'warning trial detected on sb{sb_n}, day{day_n}, {time}, g{g_n}, t{c_n}')
                        continue
                    segment = seg_emg(samples)
                    rest_samples = emg[selected_index[-1]+rest_len:selected_index[-1]+rest_len+round(len(samples)/6)][:,valid_channel]
                    segment_rest = seg_emg(rest_samples)
                    n_segments = [len(segment), len(segment_rest)]
                    if c_n in val_cycle:
                        W_val.extend(snr+np.zeros(n_segments[0]))
                        W_val.extend(snr+np.zeros(n_segments[1]))
                    else:
                        W_train.extend(snr+np.zeros(n_segments[0]))
                        W_train.extend(snr+np.zeros(n_segments[1]))

            np.save(train_folder + f'/W_d{day_n}_t{time_n}.npy', np.array(W_train, dtype=np.float32))
            np.save(val_folder + f'/W_d{day_n}_t{time_n}.npy', np.array(W_val, dtype=np.float32))



if __name__ == "__main__":

    # Load config file from hpo search
    with open("hpo_search_clean.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    # Load determined optimal hyperparameters
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    cfg.model_name = study_dir
    print(cfg.model_name)
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir

    #for sb_n in [1,3,4,5,6,7,8,10]:
    for sb_n in [10]:
        cfg.DATA_CONFIG.sb_n = sb_n

        #get_weights_snr(cfg)
        #exit()
        with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
            hp = yaml.load(f, Loader=yaml.SafeLoader)

        cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
        cfg.best_loss = hp[0]['best_loss']
        #cfg.HP = {}
        for key, item in hp[1].items():
            cfg.HP[key] = item

        if TCN_USED:
            cfg.HP['kernel_size'] = cfg.HP_SEARCH.TCN.kernel_list[cfg.HP.layer_n-3]

        if not os.path.exists(cfg.RESULT_PATH):
            os.makedirs(cfg.RESULT_PATH)

        cfg.index = 'window'


        cfg.test_model = cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test directly from the best model saved during HPO


        #get_weights(cfg)
        get_weights_edl(cfg)
        #get_weights_snr(cfg)
