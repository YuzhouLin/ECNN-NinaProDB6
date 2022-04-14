import argparse
import os
import torch
import torch.optim as optim
import numpy as np
import utils
import helps_pre as pre
#import optuna
#from optuna.samplers import TPESampler
#from optuna.pruners import MedianPruner
#import copy
#from torch.utils.tensorboard import SummaryWriter
import time
import yaml
from torch.utils.data import TensorDataset, WeightedRandomSampler
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl (annealing); \
        3: edl with kl (trade-off)')
parser.add_argument('--tcn', action='store_true', default=False,
                    help='to use tcn if it activates, cnn otherwise')

args = parser.parse_args()

#ID = int(time.time())
#ID = 'tcn10'
ID = 'clean'
EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.try_gpu()
EPOCHS = 200
CLASS_N = 8
CHANNEL_N = 14
N_LAYERS = 3
KERNER = 51
N_WORKERS = 8

#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 4, 5, 7, 8, 10, 11]
VAL_TRIAL_LIST = [3, 6, 9, 12]
DATA_PATH = '/../../hy-nas/Data6/'



def run_training(params):
    # load_data
    sb_n = params['sb_n']

    '''
    train_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], TRAIN_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)

    val_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], VAL_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    '''

    train_X = np.load(DATA_PATH+f's{sb_n}/train/X.npy')
    train_Y = np.load(DATA_PATH+f's{sb_n}/train/Y.npy')
    val_X = np.load(DATA_PATH+f's{sb_n}/val/X.npy')
    val_Y = np.load(DATA_PATH+f's{sb_n}/val/Y.npy')
    train_W = np.load(DATA_PATH+f's{sb_n}/train/W.npy')
    val_W = np.load(DATA_PATH+f's{sb_n}/val/W.npy')

    '''
    train_X = np.load(DATA_PATH+f's{sb_n}/train/X_d2.npy')
    train_Y = np.load(DATA_PATH+f's{sb_n}/train/Y_d2.npy')
    train_W = np.load(DATA_PATH+f's{sb_n}/train/W_d2.npy')
    val_X = np.load(DATA_PATH+f's{sb_n}/val/X_d2.npy')
    val_Y = np.load(DATA_PATH+f's{sb_n}/val/Y_d2.npy')
    val_W = np.load(DATA_PATH+f's{sb_n}/val/W_d2.npy')
    '''
    '''
    train_X = np.load(DATA_PATH+f's{sb_n}/train/X_steady.npy')
    train_Y = np.load(DATA_PATH+f's{sb_n}/train/Y_steady.npy')
    val_X = np.load(DATA_PATH+f's{sb_n}/val/X_steady.npy')
    val_Y = np.load(DATA_PATH+f's{sb_n}/val/Y_steady.npy')

    train_W = np.load(DATA_PATH+f's{sb_n}/train/W_steady.npy')
    val_W = np.load(DATA_PATH+f's{sb_n}/val/W_steady.npy')

    #train_W = 18-train_W
    #val_W = 18 -val_W
    '''
    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))
    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
        X_val_torch = torch.squeeze(X_val_torch, 1)

    W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))
    W_val_torch = torch.from_numpy(np.array(val_W,dtype=np.float32))

    #train_data = TensorDataset(X_train_torch, Y_train_torch)
    #val_data = TensorDataset(X_val_torch, Y_val_torch)
    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch, W_val_torch)

    _, train_class_counts = np.unique(train_Y, return_counts=True)
    _, val_class_counts = np.unique(val_Y, return_counts=True)

    n_train = train_class_counts.sum()
    n_val = val_class_counts.sum()
    class_weights_train = [float(n_train)/train_class_counts[i] for i in range(CLASS_N)]
    class_weights_val = [float(n_val)/val_class_counts[i] for i in range(CLASS_N)]

    weights_train = train_Y
    weights_val = val_Y
    for i in range(CLASS_N):
        weights_train[train_Y==i] = class_weights_train[i]
        weights_val[val_Y==i] = class_weights_val[i]
    sampler_train = WeightedRandomSampler(weights_train, int(n_train),replacement=True)
    sampler_val = WeightedRandomSampler(weights_val, int(n_val),replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], sampler=sampler_train, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=params['batch_size'], sampler=sampler_val, drop_last=True, num_workers=N_WORKERS, pin_memory=True)

    '''
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    '''


    '''
    for i, (x,y,z) in enumerate(train_loader):
        print(f"batch index {i}")
        tmp = []
        for g in range(8):
            tmp.append((y==g).sum())
        print(tmp)
    print('-------------')
    for i, (x,y,z) in enumerate(val_loader):
        print(f"batch index {i}")
        tmp = []
        for g in range(8):
            tmp.append((y==g).sum())
        print(tmp)
    exit()
    '''
    trainloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    dropout_rate=params['dropout_rate']
    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)

    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'], weight_decay= 1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=5, verbose=True, eps=1e-7)
    #lr_check = optimizer.param_groups[0]['lr']
    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = pre.update_loss_params(params)
    print(loss_params)
    #exit()

    loss_params['device'] = DEVICE


    #writer = SummaryWriter(f'/../../tf_logs/{ID}') # tensorboard try
    # images, labels = next(iter(train_loader)) # tensorboard try
    #writer.add_graph(model, images.to(DEVICE)) # tensorboard try
    best_loss = np.inf
    early_stopping_iter = 10
    torch.backends.cudnn.benchmark = True
    for epoch in range(1, EPOCHS + 1):
        #t0 = time.time()
        # if 'annealing_step' in loss_params:
        loss_params['epoch_num'] = epoch
        #train_losses, acc = eng.train(trainloaders, loss_params)
        train_losses, tmp_pred, tmp_true = eng.train(trainloaders, loss_params)
        train_loss = train_losses['train']
        valid_loss = train_losses['val']
        scheduler.step(valid_loss)
        #acc['train']
        #val_acc = acc['val']
        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            f"valid_loss: {valid_loss}. "
            #f"train_acc: {train_acc}, "
            #f"valid_acc: {val_acc}. "
        )
        #writer.add_scalars('Loss', {'train':train_loss, 'val':valid_loss}, global_step=epoch)
        #writer.add_scalar('ValidLoss', valid_loss, global_step=epoch)
        #writer.add_scalars('Acc', {'train': train_acc, 'val': valid_acc}, global_step=epoch)
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #    writer.add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), epoch)
        #writer.add_scalar('TrainAcc', train_acc, global_step=epoch)
        #writer.add_scalar('ValidAcc', valid_acc, global_step=epoch)
        #if lr_check != optimizer.param_groups[0]['lr']:
            #print('restart early stop')
            #lr_check = optimizer.param_groups[0]['lr']
            #early_stopping_counter = 0
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
            print('-'*20)
            print('training acc for each class: ', np.round(train_acc, decimals=2))
            print('val acc for each class: ', np.round(val_acc, decimals=2))
            print('-'*20)
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
            #writer.close()
            break
        #writer.close()
        #print('{} seconds'.format(time.time() - t0))
    return #best_loss

def re_train(params):
    sb_n = params['sb_n']

    #trial_n = 4

    #for day_n in [2,3,4,5]:
    #    for time_n in [1,2]:

    val_X = np.load(DATA_PATH+f's{sb_n}/val/X.npy')
    val_Y = np.load(DATA_PATH+f's{sb_n}/val/Y.npy')
    train_X = np.load(DATA_PATH+f's{sb_n}/train/X.npy')
    train_Y = np.load(DATA_PATH+f's{sb_n}/train/Y.npy')
    train_W = np.load(DATA_PATH+f's{sb_n}/train/W.npy')
    val_W = np.load(DATA_PATH+f's{sb_n}/val/W.npy')

    #X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    #Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_train_torch = torch.from_numpy(np.concatenate((train_X, val_X), dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.concatenate((train_Y, val_Y), dtype=np.int64))

    if TCN_USED:
        #X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
        X_train_torch = torch.squeeze(X_train_torch, 1)

    #W_train_torch = torch.from_numpy(np.array(train_W,dtype=np.float32))
    W_train_torch = torch.from_numpy(np.concatenate((train_W,val_W),dtype=np.float32))

    #train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)
    train_data = TensorDataset(X_train_torch, Y_train_torch, W_train_torch)

    #_, train_class_counts = np.unique(train_Y, return_counts=True)
    Y_total = np.concatenate((train_Y,val_Y))
    _, train_class_counts = np.unique(Y_total, return_counts=True)

    #n_train = train_class_counts.sum()
    n_train = train_class_counts.sum()
    #class_weights_train = [float(n_train)/train_class_counts[i] for i in range(CLASS_N)]
    class_weights_train = [float(n_train)/train_class_counts[i] for i in range(CLASS_N)]

    weights_train = Y_total
    for i in range(CLASS_N):
        weights_train[Y_total==i] = class_weights_train[i]
    sampler_train = WeightedRandomSampler(weights_train, int(n_train),replacement=True)
    #sampler_val = WeightedRandomSampler(weights_val, int(n_val),replacement=True)
    #train_loader = torch.utils.data.DataLoader(
    #    train_data, batch_size=params['batch_size'], sampler=sampler_train, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], sampler=sampler_train, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    # Load Model
    checkpoint = torch.load(params['saved_model'])
    dropout_rate=params['dropout_rate']
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=dropout_rate)
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=dropout_rate)



    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,amsgrad=True)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'], weight_decay= 1e-4)

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = pre.update_loss_params(params)
    print(loss_params)

    loss_params['device'] = DEVICE

    for epoch in range(1, 11):
        #t0 = time.time()
        # if 'annealing_step' in loss_params:
        loss_params['epoch_num'] = epoch
        #train_losses, acc = eng.train(trainloaders, loss_params)
        train_loss, tmp_pred, tmp_true = eng.re_train(train_loader, loss_params)
        g_i, g_n = np.unique(tmp_true, return_counts=True)
        tmp_pred = np.array(tmp_pred)
        tmp_true = np.array(tmp_true)
        train_acc = []
        for i in g_i:
            train_acc.append(np.sum(tmp_pred[tmp_true==i]==i)/g_n[i])

        print('training acc for each class: ', np.round(train_acc, decimals=2))
        print('-'*20)
        print(
            f"epoch:{epoch}, "
            f"train_loss: {train_loss}, "
            #f"train_acc: {train_acc}, "
            #f"valid_acc: {val_acc}. "
            )
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer': params['optimizer'],
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            #'valid_loss': valid_loss # best_loss
            }, params['retrained_model'])



def test(params):
    # load_data
    sb_n = params['sb_n']
    checkpoint = torch.load(params['saved_model'],map_location=torch.device('cpu'))
    #checkpoint = torch.load(params['retrained_model'],map_location=torch.device('cpu'))

    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=params['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.to(DEVICE)
    model.eval()

    day_n = 2
    time_n = 2
    #trial_n = 4

    #for day_n in [2,3,4,5]:
    #    for time_n in [1,2]:

    test_X = np.load(DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
    Y_numpy = np.load(DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')

    print(np.shape(test_X))
    X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    if TCN_USED:
        X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
    with torch.no_grad():
        outputs = model(X_torch).detach().cpu()
        # get results
        eng = utils.EngineTest(outputs, Y_numpy)
        predict = np.squeeze(eng.get_pred_labels())
        print(f'on day{day_n}, time{time_n}')

        g_i, g_n = np.unique(Y_numpy, return_counts=True)
        for i,j in zip(g_i,g_n):
            print(f'test_acc on gesture{i}: {np.sum(predict[Y_numpy==i]==i)/j}')

    #day_n = 3
    #time_n = 2
    #trial_n = 4
    '''
    for day_n in [2,3,4,5]:
        for time_n in [1,2]:
            test_X = np.load(DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
            Y_numpy = np.load(DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            with torch.no_grad():
                outputs = model(X_torch).detach().cpu()
                # get results
                eng = utils.EngineTest(outputs, Y_numpy)
                predict = np.squeeze(eng.get_pred_labels())
                print(f'on day{day_n}, time{time_n}')
                print('test_acc: ', np.sum(Y_numpy == predict)/len(predict))
    '''
    return #best_loss

def test_report(params):
    # load_data
    sb_n = params['sb_n']
    checkpoint = torch.load(params['saved_model'], map_location=torch.device('cpu'))

    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=params['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    sb_n_list = []
    model_list = []
    day_n_list = []
    time_n_list = []
    predict_list = []
    actual_list = []
    SNR_list = []


    for day_n in [2,3,4,5]:
        print('day ', day_n)
        for time_n in [1,2]:
            print('time ', time_n)
            test_X = np.load(DATA_PATH+f's{sb_n}/test/X_d{day_n}_t{time_n}.npy')
            Y_numpy = np.load(DATA_PATH+f's{sb_n}/test/Y_d{day_n}_t{time_n}.npy')
            W_numpy = np.load(DATA_PATH+f's{sb_n}/test/W_d{day_n}_t{time_n}.npy')
            X_torch = torch.from_numpy(np.array(test_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
            if TCN_USED:
                X_torch = torch.squeeze(X_torch, 1) # ([5101, 14, 400])
            with torch.no_grad():
                outputs = model(X_torch).detach().cpu()
                # get results
                eng = utils.EngineTest(outputs, Y_numpy)
                predict = np.squeeze(eng.get_pred_labels())
                #print(f'on day{day_n}, time{time_n}')
                #print('test_acc: ', np.sum(Y_numpy == predict)/len(predict))
                #print(np.shape(Y_numpy))
                #print(np.shape(predict))
                #exit()
                predict_list.extend(predict)
                actual_list.extend(Y_numpy)
                SNR_list.extend(W_numpy)
                day_n_list.extend(np.zeros((len(predict),), dtype=int)+day_n)
                time_n_list.extend(np.zeros((len(predict),), dtype=int)+time_n)

    model_type = 'etcn0' if TCN_USED else 'ecnn0'
    n = len(predict_list)
    test_dict = {
        'sb': np.zeros((n,), dtype=int)+sb_n,
        'model': [model_type]*n,
        'day': day_n_list,
        'time': time_n_list,
        'predict': predict_list,
        'actual': actual_list,
        'SNR': SNR_list
    }

    df_new = pd.DataFrame(test_dict)

    result_file = './SampleNormalisedWeightedTrainingResultsSteady.csv'

    if os.path.exists(result_file):
        df = pd.read_csv(result_file)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(result_file, index=False)
    return

if __name__ == "__main__":

    #cv_hyperparam_study()
    #print(DEVICE)
    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED,
        'tcn_used': TCN_USED
    }

    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1
        params['evi_fun'] = 'relu'
        params['annealing_step'] = 10
        params['l'] = 0.1

    if TCN_USED:
        params['channels']= [14]*N_LAYERS #[16,32,64,128,256]
        params['kernel_size'] = KERNER
        params['lr'] = 1.087671099086256e-04


    prefix_path = f'/../../hy-nas/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-nas/models/ecnn{EDL_USED}/'


    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    # retraining and save the models

    #for sb_n in range(3, 4): # modify it to (1, 11) later
    sb_n = 4
    params['sb_n'] = sb_n
        #core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        #study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        #loaded_study = optuna.load_study(
        #    study_name="STUDY", storage=study_path)
        #temp_best_trial = loaded_study.best_trial
        # Update for the optimal hyperparameters
        #for key, value in temp_best_trial.params.items():
        #    params[key] = value
    filename = f'sb{sb_n}-{ID}.pt' # change it later
    model_name = os.path.join(prefix_path, filename)
    params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value

    params['retrained_model'] = os.path.join(prefix_path, f'sb{sb_n}-{ID}-retrained.pt')
    params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
    params['batch_size'] = 128# 256
    params['dropout_rate'] = 0.4#7611414535237153
    if not TCN_USED:
        params['lr'] = 0.0004#0.0006 #0.001#0.0006 #0.00030867277604946856
        params['dropout_rate'] = 0.65022470716520555
    run_training(params)
    #re_train(params)
    test(params)
    #test_report(params)
    #os.system('shutdown')
