import argparse
import os
import torch
import numpy as np
import utils
from src import helps_pre as pre
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

parser = argparse.ArgumentParser()
parser.add_argument(
    'edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')

args = parser.parse_args()

EDL_USED = args.edl
DEVICE = pre.get_device()
EPOCHS = 150
CLASS_N = 12
TRIAL_LIST = list(range(1, 7))
DATA_PATH = '/data/NinaproDB5/raw/'


def run_training(fold, params, save_model):
    # load_data
    '''
    temp_trial_list = [
        x for x in TRIAL_LIST if x not in params['test_trial_list']]
    '''
    o_f = params['outer_f']  # outer fold num
    temp_trial_list = [x for x in TRIAL_LIST if x != o_f]
    valid_trial_list = [temp_trial_list.pop(fold)]
    train_trial_list = temp_trial_list

    sb_n = params['sb_n']

    train_loader = pre.load_data_cnn(
        DATA_PATH, sb_n, train_trial_list, params['batch_size'])
    valid_loader = pre.load_data_cnn(
        DATA_PATH, sb_n, valid_trial_list, params['batch_size'])

    trainloaders = {
        "train": train_loader,
        "val": valid_loader,
    }

    # Load Model
    model = utils.Model()
    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'])

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)
    
    loss_params = pre.update_loss_params(params)
    loss_params['device'] = DEVICE
    
    if save_model:
        prefix_path = f'model_innerloop/ecnn{EDL_USED}/'
        if not os.path.exists(prefix_path):
            os.makedirs(prefix_path)

        filename = f"sb{sb_n}_o{o_f}_i{fold}.pt"
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
            f"fold:{fold}, "
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

    all_losses = []
    for i_f in range(len(TRIAL_LIST) - 1):  # len(TRIAL_LIST) - 1 = 5
        temp_loss = run_training(i_f, params, save_model=False)
        intermediate_value = temp_loss
        trial.report(intermediate_value, i_f)

        if trial.should_prune():
            raise optuna.TrialPruned()
        all_losses.append(temp_loss)

    return np.mean(all_losses)


def cv_hyperparam_study():
    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED
    }
    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1
 
    for test_trial in range(1, 7):
        params['outer_f'] = test_trial
        for sb_n in range(1, 11):
            params['sb_n'] = sb_n
            study_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
            if not os.path.exists(study_path):
                os.makedirs(study_path)
            sampler = TPESampler()
            study = optuna.create_study(
                direction="minimize",  # maximaze or minimaze our objective
                sampler=sampler,  # parametrs sampling strategy
                pruner=MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=3,  # let's say num epochs
                    interval_steps=1,
                ),
                study_name='STUDY',
                storage="sqlite:///" + study_path + f"/t{test_trial}.db",
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

