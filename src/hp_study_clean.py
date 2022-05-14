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
from torch.utils.data import TensorDataset, WeightedRandomSampler
from scipy.stats import hmean
import re

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
#GLOBAL_BEST_ACC = 0

def run_training(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    n_class = len(cfg.CLASS_NAMES)

    if cfg.TRAINING.day_n==1:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy')), axis=0)
        #train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy')), axis=0)
        #val_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy')), axis=0)
    elif cfg.TRAINING.day_n==2:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t2.npy')), axis=0)
        #train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t2.npy')), axis=0)
        #val_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t2.npy')), axis=0)

    print(np.shape(train_X))
    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))
    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
        X_val_torch = torch.squeeze(X_val_torch, 1)

    W_train_torch = torch.ones(len(Y_train_torch),dtype=torch.float32)
    W_val_torch = torch.ones(len(Y_val_torch),dtype=torch.float32)
    #W_train_torch = torch.from_numpy(np.array(100**train_W,dtype=np.float32))
    #W_val_torch = torch.from_numpy(np.array(100**val_W,dtype=np.float32))

    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch, W_val_torch)

    _, train_class_counts = np.unique(train_Y, return_counts=True)
    _, val_class_counts = np.unique(val_Y, return_counts=True)

    n_train = train_class_counts.sum()
    n_val = val_class_counts.sum()
    class_weights_train = [float(n_train)/train_class_counts[i] for i in range(n_class)]
    class_weights_val = [float(n_val)/val_class_counts[i] for i in range(n_class)]

    weights_train = train_Y
    weights_val = val_Y
    for i in range(n_class):
        weights_train[train_Y==i] = class_weights_train[i]
        weights_val[val_Y==i] = class_weights_val[i]
    sampler_train = WeightedRandomSampler(weights_train, int(n_train),replacement=True)
    sampler_val = WeightedRandomSampler(weights_val, int(n_val),replacement=True)
    # load_data
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.HP.batch_size, sampler=sampler_train, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=cfg.HP.batch_size, sampler=sampler_val, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)

    trainloaders = {
        "train": train_loader,
        "val": val_loader,
    }


    # Load Model
    if TCN_USED:
        #model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.tcn_channels, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,cfg.HP.optimizer)(model.parameters(), lr=cfg.HP.lr, weight_decay=cfg.HP.weight_decay, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.HP.lr_factor, patience=int(cfg.TRAINING.early_stopping_iter/2), verbose=True, eps=cfg.HP.scheduler_eps)
    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = {'edl_used': EDL_USED, 'device': DEVICE}
    if EDL_USED != 0:
        loss_params['class_n'] = n_class
        for item in cfg.HP_SEARCH[f'EDL{EDL_USED}']:
            loss_params[item]=cfg.HP[item]

    best_loss = np.inf
    early_stopping_iter = cfg.TRAINING.early_stopping_iter

    torch.backends.cudnn.benchmark = True
    for epoch in range(1, cfg.TRAINING.epochs + 1):
        # if 'annealing_step' in loss_params:
        if EDL_USED == 2:
            loss_params['epoch_num'] = epoch
        train_losses, tmp_pred, tmp_true = eng.train(trainloaders, loss_params)
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        scheduler.step(valid_loss)

        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            f"valid_loss: {valid_loss}. "
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stopping_counter = 0
            g_i_train, g_n_train = np.unique(tmp_true['train'], return_counts=True)
            _, g_n_val = np.unique(tmp_true['val'], return_counts=True)
            train_acc = []
            val_acc = []
            tmp_pred_train = np.array(tmp_pred['train'])
            tmp_true_train = np.array(tmp_true['train'])
            tmp_pred_val = np.array(tmp_pred['val'])
            tmp_true_val = np.array(tmp_true['val'])
            for g_i in g_i_train:
                train_acc.append(np.sum(tmp_pred_train[tmp_true_train==g_i]==g_i)/g_n_train[g_i])
                val_acc.append(np.sum(tmp_pred_val[tmp_true_val==g_i]==g_i)/g_n_val[g_i])
            best_acc = hmean(val_acc)

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
        if early_stopping_counter > early_stopping_iter:
            break

    global GLOBAL_BEST_LOSS
    #global GLOBAL_BEST_ACC
    if best_loss < GLOBAL_BEST_LOSS:
    #if best_acc > GLOBAL_BEST_ACC:
        print(
            f"epoch:{epoch}, "
            f"best_loss: {best_loss} "
            #f"best_acc: {best_acc}"
        )

        print('-'*20)
        print('training acc for each class: ', np.round(train_acc, decimals=2))
        print('val acc for each class: ', np.round(val_acc, decimals=2))
        print('-'*20)

        GLOBAL_BEST_LOSS = best_loss
        #GLOBAL_BEST_ACC = best_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'train_acc:': train_acc,
            'valid_acc': val_acc
            }, cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{sb_n}.pt') # modify it later
    return best_loss
    #return best_acc

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
    cfg.HP['batch_size'] = cfg.HP['batch_base']*2**cfg.HP['batch_factor']
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
    best_loss = run_training(cfg)
    return best_loss


def cv_hyperparam_study(sb_n):

    # Load config file
    with open("hpo_search_clean.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    #sb_n=cfg.DATA_CONFIG.sb_n
    cfg.DATA_CONFIG.sb_n = sb_n
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
        study_name=study_dir+f'_sb{sb_n}_{re.findall(r"_(.+)/", cfg.STUDY_PATH)[0]}', #baseline{cfg.TRAINING.day_n}
        storage=f"sqlite:///study/{study_dir}/sb{sb_n}.db",
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.HPO_STUDY.trial_n)


    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    study_results = [{'best_loss': trial.value}, trial.params]
    #study_results = [{'best_hormonic_mean_acc': trial.value}, trial.params]
    #if TCN_USED:
    #    study_results[1]['tcn_channels'] = cfg.HP_SEARCH.TCN.tcn_channels

    # Write (Save) HP to a yaml file
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'w') as f:
        yaml.dump(study_results, f)

    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    return


if __name__ == "__main__":

    #for sb_n in [1,3,4,5,6,7,8]:
    for sb_n in [1,3,4,5,6,7,8,10]:
        global GLOBAL_BEST_LOSS
        GLOBAL_BEST_LOSS = np.inf
        cv_hyperparam_study(sb_n)

    os.system('shutdown')

