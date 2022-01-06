import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
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
EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.try_gpu()
EPOCHS = 100
CLASS_N = 8
CHANNEL_N = 14
#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 3, 5, 6, 7, 9, 10, 11]
VALID_TRIAL_LIST = [4, 8, 12]
DATA_PATH = '/../../hy-tmp/Data6/Processed/'
TCN_CHANNELS = [[16, 32], [16, 32, 64], [16, 32, 64, 128]]


def run_training(params):
    # load_data
    sb_n = params['sb_n']
    train_params = {'data_path': DATA_PATH, 
                   'sb_n': sb_n,
                    'day_list': [1],
                    'time_list': [1, 2],
                    'trial_list': TRAIN_TRIAL_LIST,
                    'batch_size': params['batch_size'],
                    'tcn_used': TCN_USED
                   }
    valid_params = copy.deepcopy(train_params)
    valid_params['trial_list'] = VALID_TRIAL_LIST
    train_loader = pre.load_data(train_params)
    valid_loader = pre.load_data(valid_params)
    
    trainloaders = {
        "train": train_loader,
        "val": valid_loader,
    }

    dropout_rate=params['dropout_rate']
    # Load Model
    if TCN_USED:
        #tcn_channels = params['channels']
        tcn_channels = TCN_CHANNELS[1]
        #tcn_channels = TCN_CHANNELS[params['layer_o']]
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)
    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'])

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)
    
    loss_params = pre.update_loss_params(params)
    loss_params['device'] = DEVICE
    

    best_loss = np.inf
    early_stopping_iter = 10
    for epoch in range(1, EPOCHS + 1): 
        # if 'annealing_step' in loss_params:
        loss_params['epoch_num'] = epoch
        train_losses, acc = eng.train(trainloaders, loss_params)
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        train_acc = acc['train']
        valid_acc = acc['val']
        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            f"valid_loss: {valid_loss}. "
            f"train_acc: {train_acc}, "
            f"valid_acc: {valid_acc}. "
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': params['optimizer'],
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss # best_loss
                }, params['saved_model'])
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def objective(trial, params):
    # Update the params for tuning with cross validation
    params['optimizer'] = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])
    params['lr'] = trial.suggest_loguniform("lr", 1e-6, 1e-2)
    params['batch_size'] = trial.suggest_int("batch_size", 32, 512, step=32)
    params['dropout_rate'] = trial.suggest_float("dropout_rate", 0.1, 0.9)

    if TCN_USED:
        params['kernel_size'] = trial.suggest_int("kernel_size", 2, 6)
        #params['layer_o'] = trial.suggest_int("layer_o", 0, 2)


    if EDL_USED != 0:
        params['evi_fun'] = trial.suggest_categorical(
            "evi_fun", ["relu", "softplus", "exp"])
        if EDL_USED == 2:
            params['annealing_step'] = trial.suggest_int(
                "annealing_step", 10, 60, step=5)
        elif EDL_USED == 3:
            params['l'] = trial.suggest_float(
                "l", 0.01, 1.0, log=True)  # l:lambda

    temp_loss = run_training(params)
    
    #intermediate_value = temp_loss
    #trial.report(intermediate_value, i_f)

    #if trial.should_prune():
    #    raise optuna.TrialPruned()
    #all_losses.append(temp_loss)

    #return np.mean(all_losses)
    return temp_loss


def cv_hyperparam_study():
    prefix_path = f'/models/etcn{EDL_USED}/' if TCN_USED else f'/models/ecnn{EDL_USED}/'
    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)
    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED,
        'saved_model': os.path.join(prefix_path, 'study.pt')
    }
    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1
 
    sb_n=1

    params['sb_n'] = sb_n
    #study_path = f'/../../hy-tmp/study/ecnn{EDL_USED}/sb{sb_n}'
    study_path = f'/study/etcn{EDL_USED}/' if TCN_USED else f'/study/ecnn{EDL_USED}/'
    if not os.path.exists(study_path):
        os.makedirs(study_path)
    sampler = TPESampler()
    study = optuna.create_study(
        direction="minimize",  # maximaze or minimaze our objective
        sampler=sampler,  # parametrs sampling strategy
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5,  # let's say num epochs
            interval_steps=1,
        ), 
        study_name='STUDY_tcn',
        storage="sqlite:///" + study_path + f"/tcn_temp.db", # modify it later 
        # storing study results
        load_if_exists=True  # An error will be raised if same name
    )

    study.optimize(lambda trial: objective(trial, params), n_trials=25)
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return


if __name__ == "__main__":

    cv_hyperparam_study()
    
    os.system('shutdown')