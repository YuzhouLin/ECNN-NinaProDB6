import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
#import optuna
#from optuna.samplers import TPESampler
#from optuna.pruners import MedianPruner
#import copy
#from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from torch.utils.data import TensorDataset
import pandas as pd
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl (annealing); \
        3: edl with kl (trade-off)')
parser.add_argument('--tcn', action='store_true', default=False,
                    help='to use tcn if it activates, cnn otherwise')

args = parser.parse_args()

#ID = int(time.time())
#ID = 'tcn10'
ID = 'clean'
EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.try_gpu()
EPOCHS = 200
CLASS_N = 8
CHANNEL_N = 14
N_LAYERS = 3
KERNER = 51
N_WORKERS = 4

#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 4, 5, 7, 8, 10, 11]
VAL_TRIAL_LIST = [3, 6, 9, 12]
DATA_PATH = '/../../hy-nas/Data6/'


def test(params):
    # load_data
    sb_n = params['sb_n']
    #checkpoint = torch.load(params['saved_model'])
    checkpoint = torch.load(params['saved_model'], map_location=torch.device('cpu'))
    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=params['dropout_rate'])
    #model.load_state_dict(checkpoint['model_state_dict'], map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.to(DEVICE)
    model.eval()


    day_n = 5
    time_n = 1
    #trial_n = 4

    #for day_n in [2,3,4,5]:
    #    for time_n in [1,2]:

    test_X = np.load(DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
    Y_numpy = np.load(DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')

    print(np.shape(test_X))
    X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    if TCN_USED:
        X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
    with torch.no_grad():
        outputs = model(X_torch).detach().cpu()
        # get results
        eng = utils.EngineTest(outputs, Y_numpy)
        predict = np.squeeze(eng.get_pred_labels())
        print(f'on day{day_n}, time{time_n}')

        g_i, g_n = np.unique(Y_numpy, return_counts=True)
        for i,j in zip(g_i,g_n):
            print(f'test_acc on gesture{i}: {np.sum(predict[Y_numpy==i]==i)/j}')

    return #best_loss

def test_report(params):
    # load_data
    sb_n = params['sb_n']
    checkpoint = torch.load(params['saved_model'], map_location=torch.device('cpu'))

    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=params['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    sb_n_list = []
    model_list = []
    day_n_list = []
    time_n_list = []
    predict_list = []
    actual_list = []
    SNR_list = []
    un_nentropy_list = []
    un_nnmp_list = []
    un_vac_list = []
    un_diss_list = []
    un_overall_list =[]

    for day_n in [2,3,4,5]:
        print('day ', day_n)
        for time_n in [1,2]:
            print('time ', time_n)
            test_X = np.load(DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
            Y_numpy = np.load(DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')
            W_numpy = np.load(DATA_PATH+f's{sb_n}/test/W_d{day_n}_t{time_n}.npy')
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            with torch.no_grad():
                outputs = model(X_torch).detach().cpu()
                # get results
                eng = utils.EngineTest(outputs, Y_numpy)
                predict = np.squeeze(eng.get_pred_labels())
                #print(f'on day{day_n}, time{time_n}')
                #print('test_acc: ', np.sum(Y_numpy == predict)/len(predict))
                #print(np.shape(Y_numpy))
                #print(np.shape(predict))
                #exit()
                scores = eng.get_scores('softmax',EDL_USED)
                un_nentropy_list.extend(scores['entropy'])
                un_nnmp_list.extend(scores['un_prob'])
                un_overall_list.extend(scores['overall'])
                if EDL_USED:
                    un_vac_list.extend(scores['vacuity'])
                    un_diss_list.extend(scores['dissonance'])
                else:
                    un_vac_list.extend([pd.NA]*len(predict))
                    un_diss_list.extend([pd.NA]*len(predict))
                predict_list.extend(predict)
                actual_list.extend(Y_numpy)
                SNR_list.extend(W_numpy)
                day_n_list.extend(np.zeros((len(predict),), dtype=int)+day_n)
                time_n_list.extend(np.zeros((len(predict),), dtype=int)+time_n)


    model_type = 'etcn0' if TCN_USED else 'ecnn0'
    n = len(predict_list)
    test_dict = {
        'sb': np.zeros((n,), dtype=int)+sb_n,
        'model': [model_type]*n,
        'day': day_n_list,
        'time': time_n_list,
        'predict': predict_list,
        'actual': actual_list,
        'SNR': SNR_list,
        'un_nentropy': un_nentropy_list,
        'un_nnmp': un_nnmp_list,
        'un_vac': un_vac_list,
        'un_diss': un_diss_list,
        'un_overall': un_overall_list
    }

    df_new = pd.DataFrame(test_dict)
    result_file = './SampleReversedWeightedTrainingResults.csv'

    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(result_file, index=False)
    return

if __name__ == "__main__":

    #cv_hyperparam_study()
    #print(DEVICE)
    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED,
        'tcn_used': TCN_USED
    }

    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1


    if TCN_USED:
        params['channels']= [14]*N_LAYERS #[16,32,64,128,256]
        params['kernel_size'] = KERNER
        params['lr'] = 1.187671099086256e-04


    prefix_path = f'/../../hy-nas/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-nas/models/ecnn{EDL_USED}/'


    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    # retraining and save the models

    #for sb_n in range(3, 4): # modify it to (1, 11) later
    sb_n = 4
    params['sb_n'] = sb_n
        #core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        #study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        #loaded_study = optuna.load_study(
        #    study_name="STUDY", storage=study_path)
        #temp_best_trial = loaded_study.best_trial
        # Update for the optimal hyperparameters
        #for key, value in temp_best_trial.params.items():
        #    params[key] = value
    #filename = f'sb{sb_n}-{ID}.pt' # change it later
    filename = f'best_hpo_sb{sb_n}.pt'
    model_name = os.path.join(prefix_path, filename)
    params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value
    params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
    params['batch_size'] = 512# 256
    params['dropout_rate'] = 0.4#7611414535237153
    if not TCN_USED:
        params['lr'] = 0.001#0.0006 #0.00030867277604946856
        params['dropout_rate'] = 0.65022470716520555

    test(params)
    #test_report(params)
    #os.system('shutdown')
