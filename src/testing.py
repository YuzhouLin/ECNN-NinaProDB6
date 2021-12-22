import argparse
import torch
import utils
import helps_pre as pre
import copy
import optuna


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
DATA_PATH = '/data/'
TRIAL_LIST = list(range(1, 13))
DEVICE = torch.device('cpu')
CHANNEL_N = 14
CLASS_N = 8


def test(params):
    # Load testing Data
    inputs_torch, targets_numpy = pre.load_data_test(params)
    
    print(inputs_torch.size())
    # Load trained model
    model_path = f'models/etcn{EDL_USED}/' if TCN_USED else f'models/ecnn{EDL_USED}/'
    saved_model = model_path + f'sb{params["sb_n"]}_temp.pt' # later
    
    dropout_rate = 0.5 # later
    if TCN_USED:
        tcn_channels = [16,32,64] # later
        k_s = 3 # later
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)

    model.load_state_dict(
        torch.load(saved_model, map_location=DEVICE))
    model.eval()
    
    # Get testing model outputs
    outputs = model(inputs_torch.to(DEVICE)).detach()

    # Load the Testing Engine
    eng = utils.EngineTest(outputs, targets_numpy)
    del params['data_path']
    del params['trial_list']
    # update accuracy
    eng.update_result_acc(params)

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

    params = {'edl_used': EDL_USED, 'tcn_used': TCN_USED, 'trial_list': TRIAL_LIST, 'data_path': DATA_PATH}
    # test temp
    params['sb_n'] = 1
    params['day_n'] = 5
    params['time_n'] = 1
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
