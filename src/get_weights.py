import argparse
import torch
import utils
import helps_pre as pre
import numpy as np
#import pandas as pd
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
DEVICE = pre.try_gpu()


def get_weights(cfg):
    un = "dissonance" #"vacuity"
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
        folder="val" if day_n not in cfg.DATA_CONFIG.day_list else "test"
        print(folder)

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
                        tmp_weights = scores[un]
                        tmp_weights[predict!=targets]=1-tmp_weights[predict!=targets]

                        weights.extend(tmp_weights)

                        #if EDL_USED:
                        #    un_vac_list.extend(scores['vacuity'])
                        #    un_diss_list.extend(scores['dissonance'])
                np.save(cfg.DATA_PATH+f's{sb_n}/{folder}/W_d{day_n}_t{time_n}.npy', np.array(weights, dtype=np.float32))

    return


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

        #cfg.result_path = cfg.result_path + f'/sb{cfg.DATA_CONFIG.sb_n}'

        cfg.test_model = cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test directly from the best model saved during HPO

        #cfg.result_file = cfg.RESULT_PATH+'SampleWeightedTrainingResultsVal.csv'#'./SampleReversedWeightedTrainingResults.csv'
        cfg.result_file = cfg.RESULT_PATH+cfg.RESULTS.outputfile
        get_weights(cfg)
