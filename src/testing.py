import argparse
import torch
import utils
import helps_pre as pre
import copy
import numpy as np
import pandas as pd
import os

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
DATA_PATH  = '/../../hy-tmp/Data6/Processed/'
TRIAL_LIST = list(range(1, 13))
DEVICE = torch.device('cpu')
#DEVICE = pre.try_gpu()
CHANNEL_N = 14
CLASS_N = 8


def test(params):
    # Load testing Data
    #inputs_torch, targets_numpy 


    # Load trained model
    model_path = f'/../../hy-tmp/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-tmp/models/ecnn{EDL_USED}/'
    saved_model = model_path + f'sb{params["sb_n"]}.pt' # later
    
    #dropout_rate = 0.5 # later
    dropout_rate= 0.16189036200261997
    if TCN_USED:
        tcn_channels = [16,32,64] # later
        k_s = 2 # later
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)

    model.load_state_dict(
        torch.load(saved_model, map_location=DEVICE))
    #model.load_state_dict(torch.load(saved_model))
    model.to(DEVICE)
    #print(model)
    model.eval()
    
    # Get testing model outputs
    folder = f'results/sb{params["sb_n"]}/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = folder + 'accuracy.csv'
    column_names = [*params, 'acc']
    column_names.remove('data_path')

    for trial_n in TRIAL_LIST:
        params['trial_n'] = trial_n
        if os.path.exists(filename):
            df = pd.read_csv(filename)
        else:
            df = pd.DataFrame(columns=column_names)
        inputs_torch, targets_numpy = pre.load_data_test(params)
        outputs = model(inputs_torch.to(DEVICE))
        del inputs_torch
        
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        
        results = preds == targets_numpy

        acc = np.sum(results)*1.0/len(results)
        params['acc'] = acc
        df = df.append([params])
        print(acc)
        df.to_csv(filename, index=False)
    
    '''
    dict_for_update_R = copy.deepcopy(dict_for_update_acc)
    # Get the optimal activation function
    if EDL_USED == 0:
        dict_for_update_R['acti_fun'] = 'softmax'
    else:
        # Get from hyperparameter study
        core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        loaded_study = optuna.load_study(
            study_name="STUDY", storage=study_path)
        temp_best_trial = loaded_study.best_trial
        dict_for_update_R['acti_fun'] = temp_best_trial.params['evi_fun']

    print(dict_for_update_R)
    eng.update_result_R(dict_for_update_R)
    '''
    return


if __name__ == "__main__":

    params = {'edl_used': EDL_USED, 'tcn_used': TCN_USED, 'data_path': DATA_PATH}
    # test temp
    params['sb_n'] = 1
    #params['day_n'] = 1
    #params['time_n'] = 1
    for day_n in [1, 2, 3, 4, 5]: # later
        params['day_n'] = day_n
        for time_n in [1, 2]: # later
            params['time_n'] = time_n
            test(params)
    

    '''
    for sb_n in range(1, 2): # later
        params['sb_n'] = sb_n
        model_name = f'models/etcn{EDL_USED}/' if TCN_USED else f'models/ecnn{EDL_USED}/' # later
        params['saved_model'] = model_name
        for day_n in [2, 3, 4, 5]: # later
            params['day_n'] = day_n
            for time_n in [1, 2]: # later
                params['time_n'] = time_n
                test(params)
                print(f'Testing done. sb{sb_n}-temp')
    '''
