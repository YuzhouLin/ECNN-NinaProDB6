import argparse
import os
import torch
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
from torch.utils.data import TensorDataset

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
N_WORKERS = 4

#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 4, 5, 7, 8, 10, 11]
VAL_TRIAL_LIST = [3, 6, 9, 12]
DATA_PATH = '/../../hy-nas/Data6/Processed/'



def run_training(params):
    # load_data
    sb_n = params['sb_n']

    '''
    train_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], TRAIN_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)

    val_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], VAL_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    '''

    train_X = np.load(os.getcwd()+'/data/trainX_sb{sb_n}.npy')
    train_Y = np.load(os.getcwd()+'/data/trainY_sb{sb_n}.npy')
    val_X = np.load(os.getcwd()+'/data/valX_sb{sb_n}.npy')
    val_Y = np.load(os.getcwd()+'/data/valY_sb{sb_n}.npy')

    X_train_torch = torch.from_numpy(np.array(train_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_train_torch = torch.from_numpy(np.array(train_Y, dtype=np.int64))
    X_val_torch = torch.from_numpy(np.array(val_X, dtype=np.float32)).permute(0, 1, 3, 2) # ([5101, 1, 14, 400])
    Y_val_torch = torch.from_numpy(np.array(val_Y, dtype=np.int64))
    if TCN_USED:
        X_train_torch = torch.squeeze(X_train_torch, 1) # ([5101, 14, 400])
    train_data = TensorDataset(X_train_torch, Y_train_torch)
    val_data = TensorDataset(X_val_torch, Y_val_torch)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=N_WORKERS, pin_memory=True)

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
        params['optimizer'])(model.parameters(), lr=params['lr'])

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = pre.update_loss_params(params)
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
        #writer.add_scalars('Loss', {'train':train_loss, 'val':valid_loss}, global_step=epoch)
        #writer.add_scalar('ValidLoss', valid_loss, global_step=epoch)
        #writer.add_scalars('Acc', {'train': train_acc, 'val': valid_acc}, global_step=epoch)
        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #    writer.add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), epoch)
        #writer.add_scalar('TrainAcc', train_acc, global_step=epoch)
        #writer.add_scalar('ValidAcc', valid_acc, global_step=epoch)
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
            #writer.close()
            break
        #writer.close()
        #print('{} seconds'.format(time.time() - t0))
    return #best_loss

def test(params):
    # load_data
    sb_n = params['sb_n']
    checkpoint = torch.load(params['saved_model'])

    # Load Model
    if TCN_USED:
        tcn_channels = params['channels']
        k_s = params['kernel_size']
        model = utils.TCN(input_size=CHANNEL_N, output_size=CLASS_N, num_channels=tcn_channels, kernel_size=k_s, dropout=params['dropout_rate'])
    else:
        model = utils.Model(number_of_class=CLASS_N, dropout=params['dropout_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    day_n = 3
    time_n = 2
    trial_n = 4

    X_torch, Y_numpy = pre.load_data_test(DATA_PATH, sb_n, day_n, time_n, trial_n, tcn_used=TCN_USED)
    with torch.no_grad():
        outputs = model(X_torch.to(DEVICE)).detach().cpu()
        # get results
        eng = utils.EngineTest(outputs, Y_numpy)
        predict = np.squeeze(eng.get_pred_labels())
        print('test_acc: ', np.sum(Y_numpy == predict)/len(predict))

    return #best_loss


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


    if TCN_USED:
        params['channels']= [14]*N_LAYERS #[16,32,64,128,256]
        params['kernel_size'] = KERNER
        params['lr'] = 3.187671099086256e-04


    prefix_path = f'/../../hy-nas/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-nas/models/ecnn{EDL_USED}/'


    if not os.path.exists(prefix_path):
        os.makedirs(prefix_path)

    # retraining and save the models

    for sb_n in range(3, 4): # modify it to (1, 11) later
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
        params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
        params['batch_size'] = 256
        params['dropout_rate'] = 0.4#7611414535237153
        if not TCN_USED:
            params['lr'] = 0.00040867277604946856
            params['dropout_rate'] = 0.4022470716520555
        run_training(params)
        #test(params)
    #os.system('shutdown')
