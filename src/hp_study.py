import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
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
DEVICE = pre.try_gpu()
GLOBAL_BEST_LOSS = np.inf


def run_training(fold, cfg):
    train_trial_list = list(set(range(1,cfg.DATA_CONFIG.trial_n+1))-set(cfg.CV.valid_trial_list[fold]))
    
    # load_data
    train_loader = pre.load_data(cfg.DATA_PATH, cfg.DATA_CONFIG.sb_n, cfg.DATA_CONFIG.day_list, cfg.DATA_CONFIG.time_list, train_trial_list, tcn_used=TCN_USED, batch_size=cfg.HP.batch_size, shuffle=cfg.DATA_LOADER.shuffle, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)
    
    valid_loader = pre.load_data(cfg.DATA_PATH, cfg.DATA_CONFIG.sb_n, cfg.DATA_CONFIG.day_list, cfg.DATA_CONFIG.time_list, cfg.CV.valid_trial_list[fold], tcn_used=TCN_USED, batch_size=cfg.HP.batch_size, shuffle=cfg.DATA_LOADER.shuffle, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)
    
    trainloaders = {
        "train": train_loader,
        "val": valid_loader,
    }

    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        #model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.tcn_channels, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,cfg.HP.optimizer)(model.parameters(), lr=cfg.HP.lr)

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)
    
    loss_params = {'edl_used': EDL_USED, 'device': DEVICE}
    if EDL_USED != 0:
        loss_params['class_n'] = n_class
        for item in cfg.HP_SEARCH[f'EDL{EDL_USED}']:
            loss_params[item]=cfg.HP[item]

    best_loss = np.inf
    early_stopping_iter = cfg.TRAINING.early_stopping_iter

    for epoch in range(1, cfg.TRAINING.epochs + 1): 
        # if 'annealing_step' in loss_params:
        if EDL_USED == 2:
            loss_params['epoch_num'] = epoch
        train_losses, acc = eng.train(trainloaders, loss_params)
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        train_acc = acc['train']
        valid_acc = acc['val']
        '''
        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            f"valid_loss: {valid_loss}. "
            f"train_acc: {train_acc}, "
            f"valid_acc: {valid_acc}. "
        )
        '''
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            '''
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': params['optimizer'],
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss # best_loss
                }, params['saved_model'])
            '''
        else:
            early_stopping_counter += 1
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break
    
    global GLOBAL_BEST_LOSS
    if best_loss < GLOBAL_BEST_LOSS:
        GLOBAL_BEST_LOSS = best_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss, # best_loss
            'train_acc:': train_acc,
            'valid_acc': valid_acc
            }, cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt') # modify it later
    return best_loss


def objective(trial, cfg):
    # cfg.HP={} # hyperparams
    # Update the params for tuning with cross validation
    #with open("hpo_search.yaml", 'r') as f:
    #    hyo_search = yaml.load(f, Loader=yaml.SafeLoader)    
    #trial.set_user_attr("batch_size", 256)
    #if TCN_USED:
    #    trial.set_user_attr("kernel_size", 5)
    for key, item in cfg.HP_SEARCH[f'EDL{EDL_USED}'].items():
        cfg.HP[key] =  eval(item) # Example of an item: trial.suggest_int("kernel_size", 2, 6)
    if TCN_USED:
        cfg.HP['layer_n'] = eval(cfg.HP_SEARCH['TCN'].layer_n)
        cfg.HP['kernel_size'] = cfg.HP_SEARCH['TCN'].kernel_list[cfg.HP['layer_n']-3]
        #for key, item in cfg.HP_SEARCH['TCN'].items():
        #    if key == 'tcn_channels':
        #        cfg.HP[key] = item
        #    else:
        #        cfg.HP[key] =  eval(item)
        
        #tcn_channels = [int(cfg.HP['init_channel'])]
        #for i in range(cfg.HP['tcn_layer_n']-1):
        #    tcn_channels.append(tcn_channels[-1]*2)
        #cfg.HP['TCN_CHANNELS'] = [32, 64, 128, 256, 512]
    #print(cfg.HP)
    #cfg.HP['batch_size'] = 256
    #if TCN_USED:
    #    cfg.HP['kernel_size'] = 5
    
    if cfg.CV.valid_trial_select is None:
        all_losses = []
        for fold_n in range(len(cfg.CV.valid_trial_list)):
            temp_loss = run_training(fold_n, cfg)
            intermediate_value = temp_loss
            trial.report(intermediate_value, fold_n)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
            all_losses.append(temp_loss)
        return np.mean(all_losses)
    else:
        temp_loss = run_training(cfg.CV.valid_trial_select, cfg)
        return temp_loss


def cv_hyperparam_study():

    # Load config file
    with open("hpo_search.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
    
    sb_n=cfg.DATA_CONFIG.sb_n
    # Check study path
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir
    if not os.path.exists(study_path):
        os.makedirs(study_path)

    # Check model saved path
    cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)
    
    # Create Optuna Study
    sampler = eval(cfg.HPO_STUDY.sampler)
    study = optuna.create_study(
        direction=cfg.HPO_STUDY.direction,  # maximaze or minimaze our objective
        sampler=sampler,  # parametrs sampling strategy
        pruner=eval(cfg.HPO_STUDY.pruner),
        study_name=study_dir+f'_sb{sb_n}',
        storage=f"sqlite:///study/{study_dir}/sb{sb_n}.db", 
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.HPO_STUDY.trial_n)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    study_results = [{'best_loss': trial.value}, trial.params]
    #if TCN_USED:
    #    study_results[1]['tcn_channels'] = cfg.HP_SEARCH.TCN.tcn_channels
    
    # Write (Save) HP to a yaml file 
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'w') as f:
        yaml.dump(study_results, f)
                     
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    return


if __name__ == "__main__":

    cv_hyperparam_study()
    
    os.system('shutdown')
