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
# DEVICE = torch.device('cpu')
#DEVICE = pre.try_gpu()

def test(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    print('Testing for sb: ', sb_n)
    # Load trained model
    checkpoint = torch.load(cfg.test_model, map_location=torch.device('cpu'))
    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)

    #model.load_state_dict(
    #    torch.load(checkpoint['model_state_dict'], map_location=DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.to(DEVICE)
    model.eval()

    # Load testing Data
    #inputs_torch, targets_numpy
    #trial_list = list(range(1,cfg.DATA_CONFIG.trial_n+1))

    sb_n_list = []
    model_list = []
    day_n_list = []
    time_n_list = []
    predict_list = []
    actual_list = []
    SNR_list = []
    un_nentropy_list = []
    un_nnmp_list = []
    un_overall_list =[]
    un_vac_list = []
    un_diss_list = []

    for day_n in cfg.DATA_CONFIG.day_list:
        print('on day: ', day_n)
        for time_n in cfg.DATA_CONFIG.time_list:
            print('on time: ', time_n)
            #temp_dict = {}
            #X_torch, Y_numpy = pre.load_data_test(cfg.DATA_PATH, cfg.DATA_CONFIG.sb_n, day_n, time_n, trial_n, tcn_used=TCN_USED)
            if sb_n==2 and day_n ==2 and time_n==2:
                continue
            test_X = np.load(cfg.DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
            Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')
            W_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/test/W_d{day_n}_t{time_n}.npy')
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            with torch.no_grad():
                outputs = model(X_torch).detach().cpu()
            # get results
            eng = utils.EngineTest(outputs, Y_numpy)
            predict = np.squeeze(eng.get_pred_labels())
            acti_fun = cfg.HP.evi_fun if EDL_USED else 'softmax'
            scores = eng.get_scores(acti_fun, EDL_USED)
            #temp_dict['actual'] = Y_numpy
            #temp_dict['predict'] = np.squeeze(predict)
            #temp_dict.update(scores)
            #df = pd.DataFrame(temp_dict, index=np.arange(1,len(Y_numpy)+1,1))
            #filename = cfg.result_path+ f'd{day_n}_t{time_n}_T{trial_n}.csv'
            #df.to_csv(filename, index=True, index_label=cfg.index)
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


    n = len(predict_list)
    test_dict = {
        'sb': np.zeros((n,), dtype=int)+sb_n,
        'model': [cfg.model_name]*n,
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

    #result_file = './SampleReversedWeightedTrainingResultscsv'

    if os.path.exists(cfg.result_file):
        df = pd.read_csv(cfg.result_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(cfg.result_file, index=False)
    return



if __name__ == "__main__":

    # Load config file from hpo search
    with open("hpo_search_clean.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    # Load determined optimal hyperparameters
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    cfg.model_name = study_dir
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir

    for sb_n in [1,2,3,4,5,6,7,8,10]:#[1,2,3,4,5,6,7,9]:
        cfg.DATA_CONFIG.sb_n = sb_n
        with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
            hp = yaml.load(f, Loader=yaml.SafeLoader)

        cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
        cfg.best_loss = hp[0]['best_loss']
        #cfg.HP = {}
        for key, item in hp[1].items():
            cfg.HP[key] = item

        if TCN_USED:
            cfg.HP['kernel_size'] = cfg.HP_SEARCH.TCN.kernel_list[cfg.HP.layer_n-3]
        # Check results saved path
        cfg.result_path = os.getcwd() + cfg.RESULT_PATH + study_dir
        if not os.path.exists(cfg.result_path):
            os.makedirs(cfg.result_path)

        cfg.DATA_CONFIG.day_list = [2, 3, 4, 5]
        #cfg.colunmns = ['sb', 'model', 'day', 'time', 'trial', 'window', 'actual', 'predict', 'u_entropy', 'u_nnmp', 'u_vac', 'u_diss', 'u_overall']
        #cfg.colunmns = ['actual', 'predict', 'u_entropy', 'u_nnmp', 'u_vac', 'u_diss', 'u_overall']
        cfg.index = 'window'

        cfg.result_path = cfg.result_path + f'/sb{cfg.DATA_CONFIG.sb_n}'

        cfg.test_model = cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test directly from the best model saved during HPO

        #cfg.test_model = cfg.model_path+f'/{cfg.RETRAINING.model_name}{cfg.RETRAINING.epochs}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test from the retrained model with the model initialisation using the best model saved during HPO
        cfg.result_file = './SampleReversedWeightedTrainingResults.csv'
        test(cfg)
