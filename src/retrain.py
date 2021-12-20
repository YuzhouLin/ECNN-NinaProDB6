import argparse
import torch
import utils
import helps_pre as pre
# import optuna
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')
parser.add_argument('--tcn', action='store_true', default=False,
                    help='to use tcn if it activates, cnn otherwise')
    
args = parser.parse_args()


EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.get_device()
EPOCHS = 20
CLASS_N = 8
CHANNEL_N = 14
TRIAL_LIST = list(range(1, 13)) # change it later
DATA_PATH = '/data/'


def retrain(params):
    #  load_data

    sb_n = params['sb_n']

    train_params = {'data_path': DATA_PATH, 
                   'sb_n': sb_n,
                    'day_list': [1],
                    'time_list': [1, 2],
                    'trial_list': TRIAL_LIST,
                    'batch_size': params['batch_size'],
                    'tcn_used': TCN_USED
                   }

    train_loader = pre.load_data(train_params)

    dropout_rate=params['dropout_rate']

    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'])
    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = pre.update_loss_params(params)
    loss_params['device'] = DEVICE
    print(loss_params)
    
    #best_loss = params['best_loss']
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        if 'annealing_step' in loss_params:
            loss_params['epoch_num'] = epoch
        train_loss = eng.re_train(train_loader, loss_params)
        print(
            f"epoch:{epoch}, train_loss:{train_loss}")
            #f"best_loss_from_cv:{best_loss}")
        #if train_loss < best_loss:
        #    break
    print('{} seconds'.format(time.time() - t0))
    torch.save(model.state_dict(), params['saved_model'])
    return


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
        params['channels']=[16,32,64]
        params['kernel_size'] = 3
        params['lr'] = 1e-2
    else:
        params['lr'] = 1e-3

    prefix_path = f'models/etcn{EDL_USED}/' if TCN_USED else f'models/ecnn{EDL_USED}/'
    
    
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    # retraining and save the models

    for sb_n in range(1, 2): # modify it to (1, 11) later
        params['sb_n'] = sb_n
        #core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        #study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        #loaded_study = optuna.load_study(
        #    study_name="STUDY", storage=study_path)
        #temp_best_trial = loaded_study.best_trial
        # Update for the optimal hyperparameters
        #for key, value in temp_best_trial.params.items():
        #    params[key] = value
        filename = f'sb{sb_n}_temp.pt' # change it later
        model_name = os.path.join(prefix_path, filename)
        params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value
        params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
        params['batch_size'] = 512
        params['dropout_rate'] = 0.5
        retrain(params)
