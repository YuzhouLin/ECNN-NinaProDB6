import argparse
import torch
import utils
import helps_pre as pre
import numpy as np
import pandas as pd
import os
import yaml
from easydict import EasyDict as edict

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
DEVICE = torch.device('cpu')


def test(cfg):

    # Load trained model
    saved_model = cfg.model_path+f'/best_hpo_sb{cfg.DATA_CONFIG.sb_n}.pt'
    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP_SEARCH.TCN_CHANNELS, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)
        
    model.load_state_dict(
        torch.load(saved_model['model_state_dict'], map_location=DEVICE))
    #model.load_state_dict(torch.load(saved_model))
    model.to(DEVICE)
    model.eval()
    
    # Load testing Data
    #inputs_torch, targets_numpy
    trial_list = list(range(1,cfg.DATA_CONFIG.trial_n+1))
    

    for day_n in cfg.DATA_CONFIG.day_list:
        for time_n in cfg.DATA_CONFIG.time_list:
            for trial_n in trial_list:
                temp_dict = {}
                X_torch, Y_numpy = pre.load_data_test(cfg.DATA_PATH, cfg.DATA_CONFIG.sb_n, day_n, time_n, trial_n, tcn_used=TCN_USED)
                outputs = model(X_torch.to(DEVICE)).detach().cpu()
                # get results
                eng = utils.EngineTest(outputs, Y_numpy)
                predict = eng.get_pred_labels()
                acti_fun = cfg.HP.evi_fun if EDL_USED else 'softmax'
                scores = eng.get_scores(acti_fun, EDL_USED)
                temp_dict['actual'] = Y_numpy
                temp_dict['predict'] = predict
                temp_dict.update(scores)
                df = pd.DataFrame(temp_dict)
                filename = cfg.RESULT_PATH + f'd{day_n}_t{time_n}_T{trial_n}'
                df.to_csv(filename, index=True, index_label=cfg.index)
    return


if __name__ == "__main__":

    # Load config file from hpo search
    with open("hpo_search.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
        
    # Load determined optimal hyperparameters
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
        hp = yaml.load(f, Loader=yaml.SafeLoader)
    
    cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
    cfg.best_loss = hp[0]['best_loss']
    cfg.HP = {}
    for key, item in hp[1].items():
        cfg.HP[key] = item

    cfg.DATA_CONFIG.day_list = [1, 2, 3, 4, 5]
    #cfg.colunmns = ['sb', 'model', 'day', 'time', 'trial', 'window', 'actual', 'predict', 'u_entropy', 'u_nnmp', 'u_vac', 'u_diss', 'u_overall']
    #cfg.colunmns = ['actual', 'predict', 'u_entropy', 'u_nnmp', 'u_vac', 'u_diss', 'u_overall']
    cfg.index = 'window'
    
    cfg.RESULT_PATH = os.getcwd() + f'/results/sb{cfg.DATA_CONFIG.sb_n}'
    test(cfg)
