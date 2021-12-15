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

parser = argparse.ArgumentParser()
parser.add_argument(
    'edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')

args = parser.parse_args()

EDL_USED = args.edl
DEVICE = pre.get_device()
EPOCHS = 500
CLASS_N = 8
TRIAL_LIST = list(range(1, 13))
DATA_PATH = '/data/'


def run_training(params, save_model):
    # load_data
    sb_n = params['sb_n']
    train_params = {'data_path': DATA_PATH, 
                   'sb_n': sb_n,
                    'day_list': [1],
                    'time_list': [1],
                    'trial_list': TRIAL_LIST,
                    'batch_size': params['batch_size']
                   }
    valid_params = copy.deepcopy(train_params)
    valid_params['time_list'] = [2]
    train_loader = pre.load_data_cnn(train_params)
    valid_loader = pre.load_data_cnn(valid_params)
    
    trainloaders = {
        "train": train_loader,
        "val": valid_loader,
    }

    # Load Model
    model = utils.Model(number_of_class=CLASS_N)
    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'])

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)
    
    loss_params = pre.update_loss_params(params)
    loss_params['device'] = DEVICE
    
    if save_model:
        prefix_path = f'model/ecnn{EDL_USED}/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        filename = f"sb{sb_n}_temp.pt" # modify it later
        model_name = os.path.join(prefix_path, filename)

    best_loss = np.inf
    early_stopping_iter = 10
    for epoch in range(1, EPOCHS + 1): 
        if 'annealing_step' in loss_params:
            loss_params['epoch_num'] = epoch
        train_losses = eng.train(trainloaders, loss_params)
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            f"valid_loss: {valid_loss}. "
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            if save_model:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer': params['optimizer'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss
                    }, model_name)
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def objective(trial, params):
    # Update the params for tuning with cross validation
    params['optimizer'] = trial.suggest_categorical(
        "optimizer", ["Adam", "RMSprop", "SGD"])
    params['lr'] = trial.suggest_loguniform("lr", 1e-3, 1e-2)
    params['batch_size'] = trial.suggest_int("batch_size", 128, 256, step=128)

    if EDL_USED != 0:
        params['evi_fun'] = trial.suggest_categorical(
            "evi_fun", ["relu", "softplus", "exp"])
        if EDL_USED == 2:
            params['annealing_step'] = trial.suggest_int(
                "annealing_step", 10, 60, step=5)
        elif EDL_USED == 3:
            params['l'] = trial.suggest_float(
                "l", 0.01, 1.0, log=True)  # l:lambda

    temp_loss = run_training(params, save_model=False)
    
    #intermediate_value = temp_loss
    #trial.report(intermediate_value, i_f)

    #if trial.should_prune():
    #    raise optuna.TrialPruned()
    #all_losses.append(temp_loss)

    #return np.mean(all_losses)
    return temp_loss

def cv_hyperparam_study():
    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED
    }
    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1
 
    for sb_n in range(1, 2): # modify it later
        params['sb_n'] = sb_n
        study_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
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
            study_name='STUDY',
            storage="sqlite:///" + study_path + f"/temp.db", # modify it later 
            # storing study results
            load_if_exists=False  # An error will be raised if same name
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

