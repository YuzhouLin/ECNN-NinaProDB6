import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
import copy
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
ID = 'tcn1'
EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.try_gpu()
CLASS_N = 8
CHANNEL_N = 14
N_LAYERS = 8
KERNER = 2

#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 4, 5, 7, 8, 10, 11]
VALID_TRIAL_LIST = [3, 6, 9, 12]
DATA_PATH = '/../../hy-nas/Data6/Processed/'



def test(params):
    # load_data
    sb_n = params['sb_n']
    checkpoint = torch.load(params['saved_model'])
      
    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    day_n = 3
    time_n = 2
    trial_n = 4
    X_torch, Y_numpy = pre.load_data_test(DATA_PATH, sb_n, day_n, time_n, trial_n, tcn_used=TCN_USED)
    with torch.no_grad():
        outputs = model(X_torch.to(DEVICE)).detach().cpu()
        # get results
        eng = utils.EngineTest(outputs, Y_numpy)
        predict = np.squeeze(eng.get_pred_labels())
        print('test_acc: ', np.sum(Y_numpy == predict)/len(predict))
    
    return #best_loss

if __name__ == "__main__":

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
        params['lr'] = 3.187671099086256e-04
    else:
        params['lr'] = 1e-4

    prefix_path = f'/../../hy-nas/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-nas/models/ecnn{EDL_USED}/'
    
    
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    # retraining and save the models

    for sb_n in range(2, 3): # modify it to (1, 11) later
        params['sb_n'] = sb_n
        #core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        #study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        #loaded_study = optuna.load_study(
        #    study_name="STUDY", storage=study_path)
        #temp_best_trial = loaded_study.best_trial
        # Update for the optimal hyperparameters
        #for key, value in temp_best_trial.params.items():
        #    params[key] = value
        filename = f'sb{sb_n}-{ID}.pt' # change it later
        model_name = os.path.join(prefix_path, filename)
        params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value
        params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
        params['batch_size'] = 256
        params['dropout_rate'] = 0.4#7611414535237153
        test(params)

    #os.system('shutdown')
