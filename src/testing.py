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
DEVICE = pre.try_gpu()


def get_weights(cfg):
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
                        tmp_weights = scores['entropy']
                        tmp_weights[predict!=targets]=1-tmp_weights[predict!=targets]

                        weights.extend(tmp_weights)

                        #if EDL_USED:
                        #    un_vac_list.extend(scores['vacuity'])
                        #    un_diss_list.extend(scores['dissonance'])
                np.save(cfg.DATA_PATH+f's{sb_n}/{folder}/W_d{day_n}_t{time_n}.npy', np.array(weights, dtype=np.float32))

    return


def test_all(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    print('Testing for sb: ', sb_n)
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

    # Load testing Data
    #inputs_torch, targets_numpy
    #trial_list = list(range(1,cfg.DATA_CONFIG.trial_n+1))

    day_n_list = []
    time_n_list = []
    predict_list = []
    actual_list = []
    un_nentropy_list = []
    un_nnmp_list = []
    un_overall_list =[]
    un_vac_list = []
    un_diss_list = []
    state_list = []


    for day_n in range(1,6):
        print('on day: ', day_n)

        for time_n in cfg.DATA_CONFIG.time_list:
            if day_n in cfg.DATA_CONFIG.day_list:
                folder="test"
                if day_n==2:
                    test_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d{day_n}_t{time_n}.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d{day_n}_t{time_n}.npy')), axis=0)
                    Y_numpy = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d{day_n}_t{time_n}.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d{day_n}_t{time_n}.npy')), axis=0)
                else:
                    test_X = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/X_d{day_n}_t{time_n}.npy')
                    Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/Y_d{day_n}_t{time_n}.npy')

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
                        un_nentropy_list.extend(scores['entropy'])
                        un_nnmp_list.extend(scores['un_prob'])
                        un_overall_list.extend(scores['overall'])
                        if EDL_USED:
                            un_vac_list.extend(scores['vacuity'])
                            un_diss_list.extend(scores['dissonance'])
                        else:
                            un_vac_list.extend([pd.NA]*len(predict))
                            un_diss_list.extend([pd.NA]*len(predict))
                        state_list.extend([folder]*len(predict))
                        predict_list.extend(predict)
                        actual_list.extend(targets.numpy())
                        #SNR_list.extend(weights.detach().cpu().numpy())
                        day_n_list.extend(np.zeros((len(predict),), dtype=int)+day_n)
                        time_n_list.extend(np.zeros((len(predict),), dtype=int)+time_n)
            else:
                for folder in ["train", "val"]:
                    test_X = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/X_d{day_n}_t{time_n}.npy')
                    Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/Y_d{day_n}_t{time_n}.npy')

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
                            un_nentropy_list.extend(scores['entropy'])
                            un_nnmp_list.extend(scores['un_prob'])
                            un_overall_list.extend(scores['overall'])
                            if EDL_USED:
                                un_vac_list.extend(scores['vacuity'])
                                un_diss_list.extend(scores['dissonance'])
                            else:
                                un_vac_list.extend([pd.NA]*len(predict))
                                un_diss_list.extend([pd.NA]*len(predict))
                            state_list.extend([folder]*len(predict))
                            predict_list.extend(predict)
                            actual_list.extend(targets.numpy())
                            #SNR_list.extend(weights.detach().cpu().numpy())
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
        'state': state_list,
        'un_nentropy': un_nentropy_list,
        'un_nnmp': un_nnmp_list,
        'un_vac': un_vac_list,
        'un_diss': un_diss_list,
        'un_overall': un_overall_list
    }

    df_new = pd.DataFrame(test_dict)


    '''
    if os.path.exists(cfg.result_file):
        df = pd.read_csv(cfg.result_file, dtype={"sb": np.int8, "model":np.string_, "day": np.int8, "time": np.int8, "predict": np.int8, "actual": np.int8, "state": np.string_, "un_nentropy": np.float16, "un_nnmp": np.float16, "un_vac": np.float16, "un_diss": np.float16, "un_overall": np.float16})
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(cfg.result_file, float_format=np.float16, index=False)
    '''
    df_new.to_csv(cfg.result_file, float_format=np.float16, index=False)
    return

def quick_test(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    print('Testing for sb: ', sb_n)
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


    for day_n in range(3,6):
        print('on day: ', day_n)
        folder="val" if day_n not in cfg.DATA_CONFIG.day_list else "test"
        print(folder)

        for time_n in cfg.DATA_CONFIG.time_list:
            print('on time: ', time_n)
            if day_n==2 and folder=="test":
                test_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d{day_n}_t{time_n}.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d{day_n}_t{time_n}.npy')), axis=0)
                Y_numpy = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d{day_n}_t{time_n}.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d{day_n}_t{time_n}.npy')), axis=0)
            else:
                test_X = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/X_d{day_n}_t{time_n}.npy')
                Y_numpy = np.load(cfg.DATA_PATH+f's{sb_n}/{folder}/Y_d{day_n}_t{time_n}.npy')

            #if sb_n==2 and day_n ==2 and time_n==2:
            #    continue
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            Y_torch = torch.from_numpy(Y_numpy)
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            test_data = torch.utils.data.TensorDataset(X_torch, Y_torch)
            test_loader = torch.utils.data.DataLoader(
                 test_data, batch_size=2048, shuffle=False, drop_last=False, num_workers=4)
            predict_list = []
            actual_list = []
            with torch.no_grad():
                for _, (inputs,targets) in enumerate(test_loader):
                    inputs=inputs.to(DEVICE)
                    targets=targets.detach().cpu()
                    #print(targets.size())
                    outputs = model(inputs).detach().cpu()
                    # get results
                    eng = utils.EngineTest(outputs, targets)
                    predict = np.squeeze(eng.get_pred_labels())
                    predict_list.extend(predict)
                    actual_list.extend(targets)

            predict_list = np.array(predict_list)
            actual_list = np.array(actual_list)
            print('test_acc: ', np.sum(predict_list == actual_list)/len(predict_list))



    return

if __name__ == "__main__":

    #retrained = True
    retrained = False
    # Load config file from hpo search
    with open("hpo_search_clean.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    # Load determined optimal hyperparameters
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    cfg.model_name = study_dir
    print(cfg.model_name)
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir

    #for sb_n in [1,3,4,5,6,7,8,10]:
    for sb_n in [4,5,6,7,8,10]:
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

        if retrained:
            cfg.test_model = cfg.model_path+f'/{cfg.TRAINING.retrained_model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt'
            #cfg.test_model = cfg.model_path+f'/{cfg.RETRAINING.model_name}{cfg.RETRAINING.epochs}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test from the retrained model with the model initialisation using the best model saved during HPO
        else:
            cfg.test_model = cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt' # test directly from the best model saved during HPO
        #
        #

        #cfg.result_file = cfg.RESULT_PATH+'SampleWeightedTrainingResultsVal.csv'#'./SampleReversedWeightedTrainingResults.csv'
        cfg.result_file = cfg.RESULT_PATH+cfg.RESULTS.outputfile
        test_all(cfg)

        #quick_test(cfg)
