import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
import yaml
from easydict import EasyDict as edict
from torch.utils.data import TensorDataset, WeightedRandomSampler
from scipy.stats import hmean

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


def run_training_try(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    n_class = len(cfg.CLASS_NAMES)
    #print(cfg.HP.lr)
    #cfg.HP.lr*=5
    #print(cfg.HP.lr)
    cfg.HP.lr*=0.1
    if cfg.TRAINING.day_n==1:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy')), axis=0)
        train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy')), axis=0)
        val_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy')), axis=0)
    elif cfg.TRAINING.day_n==2:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t2.npy')), axis=0)
        train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t2.npy')), axis=0)
        val_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t2.npy')), axis=0)

    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))



    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
        X_val_torch = torch.squeeze(X_val_torch, 1)


    #W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))
    #W_val_torch = torch.from_numpy(np.array(val_W,dtype=np.float32))

    #W_train_torch = torch.from_numpy(np.array(1*(train_W),dtype=np.float32))
    #W_val_torch = torch.from_numpy(np.array(1*(val_W),dtype=np.float32))


    W_train_torch = torch.ones(len(Y_train_torch),dtype=torch.float32)
    W_val_torch = torch.ones(len(Y_val_torch),dtype=torch.float32)


    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch, W_val_torch)


    _, train_class_counts = np.unique(train_Y, return_counts=True)
    _, val_class_counts = np.unique(val_Y, return_counts=True)

    print(train_class_counts)
    print(val_class_counts)
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

    # to do: write a function to get trainloaders

    # Load Model
    if TCN_USED:
        #model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.tcn_channels, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)

    if not cfg.TRAINING.retrained_from_scratch:
        print('train from best_hpo')
        checkpoint = torch.load(cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt')#, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])


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

        # 不仅要结果，还要出uncertainty
        # 重构trainloaders


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
            print('-'*20)
            print('training acc for each class: ', np.round(train_acc, decimals=2))
            print('val acc for each class: ', np.round(val_acc, decimals=2))
            print('-'*20)

            torch.save({
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc:': train_acc,
                'valid_acc': val_acc
                }, cfg.model_path+f'/{cfg.TRAINING.retrained_model_name}_sb{sb_n}.pt')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break

    return


def run_training(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    n_class = len(cfg.CLASS_NAMES)
    #print(cfg.HP.lr)
    #cfg.HP.lr*=5
    #print(cfg.HP.lr)
    #cfg.HP.lr*=0.1
    if cfg.TRAINING.day_n==1:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy')), axis=0)
        train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy')), axis=0)
        val_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy')), axis=0)
    elif cfg.TRAINING.day_n==2:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t2.npy')), axis=0)
        val_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t2.npy')), axis=0)
        val_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t2.npy')), axis=0)


    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))



    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
        X_val_torch = torch.squeeze(X_val_torch, 1)


    #W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))
    #W_val_torch = torch.from_numpy(np.array(val_W,dtype=np.float32))

    train_W = np.load(cfg.DATA_PATH+f's{sb_n}/train/W.npy')
    W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))

    #W_train_torch = torch.from_numpy(np.array(1*(train_W),dtype=np.float32))
    #W_val_torch = torch.from_numpy(np.array(1*(val_W),dtype=np.float32))


    #W_train_torch = torch.ones(len(Y_train_torch),dtype=torch.float32)
    W_val_torch = torch.ones(len(Y_val_torch),dtype=torch.float32)


    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch, W_val_torch)


    _, train_class_counts = np.unique(train_Y, return_counts=True)
    _, val_class_counts = np.unique(val_Y, return_counts=True)

    print(train_class_counts)
    print(val_class_counts)
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

    if not cfg.TRAINING.retrained_from_scratch:
        print('train from best_hpo')
        checkpoint = torch.load(cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt')#, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])


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
            print('-'*20)
            print('training acc for each class: ', np.round(train_acc, decimals=2))
            print('val acc for each class: ', np.round(val_acc, decimals=2))
            print('-'*20)

            torch.save({
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_acc:': train_acc,
                'valid_acc': val_acc
                }, cfg.model_path+f'/{cfg.TRAINING.retrained_model_name}_sb{sb_n}.pt')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            break

    return

def run_retraining(cfg):
    sb_n = cfg.DATA_CONFIG.sb_n
    n_class = len(cfg.CLASS_NAMES)

    if cfg.TRAINING.day_n==1:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'), np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy')), axis=0)
        train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy')), axis=0)
    elif cfg.TRAINING.day_n==2:
        train_X = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/X_d2_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/X_d2_t2.npy')), axis=0)
        train_Y = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/Y_d2_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/Y_d2_t2.npy')), axis=0)
        train_W = np.concatenate((np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/train/W_d2_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d1_t2.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t1.npy'),np.load(cfg.DATA_PATH+f's{sb_n}/val/W_d2_t2.npy')), axis=0)

    #W_train_torch = torch.from_numpy(np.array(np.exp(train_W),dtype=np.float32))

    print(np.max(train_W))
    print(np.min(train_W))
    #W_train_torch = torch.from_numpy(np.array(np.exp(train_W),dtype=np.float32))
    W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))
    #W_train_torch = torch.ones(len(Y_train_torch),dtype=torch.float32)



    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))

    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)

    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])







    _, train_class_counts = np.unique(train_Y, return_counts=True)

    n_train = train_class_counts.sum()
    class_weights_train = [float(n_train)/train_class_counts[i] for i in range(n_class)]

    weights_train = train_Y
    for i in range(n_class):
        weights_train[train_Y==i] = class_weights_train[i]
    sampler_train = WeightedRandomSampler(weights_train, int(n_train),replacement=True)
    # load_data
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=cfg.HP.batch_size, sampler=sampler_train, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)

    # Load Model
    if TCN_USED:
        #model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.tcn_channels, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.layer_n*[cfg.DATA_CONFIG.channel_n], kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)

    checkpoint = torch.load(cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt')#, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])


    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,cfg.HP.optimizer)(model.parameters(), lr=cfg.HP.lr, weight_decay=cfg.HP.weight_decay, betas=(0.5, 0.999))

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = {'edl_used': EDL_USED, 'device': DEVICE}
    if EDL_USED != 0:
        loss_params['class_n'] = n_class
        for item in cfg.HP_SEARCH[f'EDL{EDL_USED}']:
            loss_params[item]=cfg.HP[item]

    torch.backends.cudnn.benchmark = True
    for epoch in range(1, cfg.TRAINING.retrained_epochs + 1):
        # if 'annealing_step' in loss_params:
        if EDL_USED == 2:
            loss_params['epoch_num'] = epoch
        train_loss, tmp_pred, tmp_true = eng.retrain(train_loader, loss_params)

        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
        )

        g_i_train, g_n_train = np.unique(tmp_true, return_counts=True)

        train_acc = []

        tmp_pred_train = np.array(tmp_pred)
        tmp_true_train = np.array(tmp_true)

        for g_i in g_i_train:
            train_acc.append(np.sum(tmp_pred_train[tmp_true_train==g_i]==g_i)/g_n_train[g_i])
        print('-'*20)
        print('training acc for each class: ', np.round(train_acc, decimals=2))

    torch.save({
            'model_state_dict': model.state_dict(),
            'train_loss': train_loss,
            'train_acc:': train_acc,
            }, cfg.model_path+f'/{cfg.TRAINING.retrained_model_name}_sb{sb_n}.pt')

    return


def prepared_cfg(sb_n):

    # Load config file
    with open("hpo_search_clean.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))

    #sb_n=cfg.DATA_CONFIG.sb_n
    cfg.DATA_CONFIG.sb_n = sb_n
    # Check study path
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
        hp_study = yaml.load(f, Loader=yaml.SafeLoader)

    cfg.best_loss = hp_study[0]

    for key, item in hp_study[1].items():
        cfg.HP[key] =  item
    cfg.HP['batch_size'] = cfg.HP['batch_base']*2**cfg.HP['batch_factor']
    if TCN_USED:
        cfg.HP['layer_n'] = eval(cfg.HP_SEARCH['TCN'].layer_n)
        cfg.HP['kernel_size'] = cfg.HP_SEARCH['TCN'].kernel_list[cfg.HP['layer_n']-3]

    # Check model saved path
    cfg.model_path = os.getcwd() + cfg.MODEL_PATH + study_dir
    if not os.path.exists(cfg.model_path):
        os.makedirs(cfg.model_path)

    '''
    loaded_study = optuna.load_study(study_name=study_dir+f'_sb{sb_n}', storage=f"sqlite:///study/{study_dir}/sb{sb_n}.db")

    print("Number of finished trials: ", len(loaded_study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", loaded_study.best_trial.value)
    study_results = [{'best_loss': trial.value}, loaded_study.trial.params]
    #study_results = [{'best_hormonic_mean_acc': trial.value}, trial.params]
    #if TCN_USED:
    #    study_results[1]['tcn_channels'] = cfg.HP_SEARCH.TCN.tcn_channels
    '''
    # Write (Save) HP to a yaml file

    return cfg


if __name__ == "__main__":


    for sb_n in [10]:
        cfg = prepared_cfg(sb_n)
        run_training(cfg)
        #cfg.HP.lr*=0.1
        #run_retraining(cfg)
    #os.system('shutdown')
