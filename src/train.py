import argparse
import os
import torch
import numpy as np
import utils
import helps_pre as pre
#import optuna
#from optuna.samplers import TPESampler
#from optuna.pruners import MedianPruner
import copy
from torch.utils.tensorboard import SummaryWriter
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
ID = 'tcn2a'
EDL_USED = args.edl
TCN_USED = args.tcn
DEVICE = pre.try_gpu()
EPOCHS = 100
CLASS_N = 8
CHANNEL_N = 14
#TRAIN_TRIAL_LIST = list(range(1, 11))
#VALID_TRIAL_LIST = list(range(11, 13))
TRAIN_TRIAL_LIST =  [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]
VALID_TRIAL_LIST = [6, 12]
DATA_PATH = '/../../hy-nas/Data6/Processed/'



def run_training(params):
    # load_data
    sb_n = params['sb_n']

    train_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], TRAIN_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    valid_loader = pre.load_data(DATA_PATH, sb_n, [1], [1, 2], VALID_TRIAL_LIST, tcn_used=TCN_USED, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    
    trainloaders = {
        "train": train_loader,
        "val": valid_loader,
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
    

    writer = SummaryWriter(f'/../../tf_logs/{ID}') # tensorboard try
    images, labels = next(iter(train_loader)) # tensorboard try
    writer.add_graph(model, images.to(DEVICE)) # tensorboard try
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
        writer.add_scalars('Loss', {'train':train_loss, 'val':valid_loss}, global_step=epoch)
        #writer.add_scalar('ValidLoss', valid_loss, global_step=epoch)
        writer.add_scalars('Acc', {'train': train_acc, 'val': valid_acc}, global_step=epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            writer.add_histogram(name + '/grad', param.requires_grad_().clone().cpu().data.numpy(), epoch)
        #writer.add_scalar('TrainAcc', train_acc, global_step=epoch)
        #writer.add_scalar('ValidAcc', valid_acc, global_step=epoch)
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
                }, model_name)
            '''
        else:
            early_stopping_counter += 1
        if early_stopping_counter > early_stopping_iter:
            writer.close()
            break
        writer.close()
        #print('{} seconds'.format(time.time() - t0))
    return #best_loss

'''
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
'''

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
        params['channels']=[16,32,64,128,256]
        params['kernel_size'] = 5
        params['lr'] = 3.187671099086256e-04
    else:
        params['lr'] = 1e-5

    prefix_path = f'/../../hy-nas/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-nas/models/ecnn{EDL_USED}/'
    
    
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
        filename = f'sb{sb_n}-{ID}.pt' # change it later
        model_name = os.path.join(prefix_path, filename)
        params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value
        params['optimizer'] = "Adam"
        #params['lr'] = 1e-3
        params['batch_size'] = 256
        params['dropout_rate'] = 0.1 #0.7611414535237153
        run_training(params)

    #os.system('shutdown')