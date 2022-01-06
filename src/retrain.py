import argparse
import torch
import utils
import helps_pre as pre
# import optuna
import os
import time
#import yaml
from torch.utils.tensorboard import SummaryWriter

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
EPOCHS = 100
CLASS_N = 8
CHANNEL_N = 14
TRIAL_LIST = list(range(1, 13))
DATA_PATH = '/../../hy-tmp/Data6/Processed/'


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
    # print(loss_params)
    
    #best_loss = params['best_loss']
    writer = SummaryWriter('/../../tf_logs/sb1') # tensorboard try
    images, labels = next(iter(train_loader)) # tensorboard try
    writer.add_graph(model, images.to(DEVICE)) # tensorboard try
    t0 = time.time()
    #best_loss = 0.6514658182859421 # CNN
    best_loss = 1.2521
    for epoch in range(1, EPOCHS + 1):
        #if 'annealing_step' in loss_params:
        #    loss_params['epoch_num'] = epoch
        loss_params['epoch_num'] = epoch
        train_loss = eng.re_train(train_loader, loss_params)
        print(
            f"epoch:{epoch}, train_loss:{train_loss}, best_loss_from_hpo:{best_loss}")
            #f"best_loss_from_cv:{best_loss}")
        writer.add_scalar('TrainLoss', train_loss, global_step=epoch) # tensorboard try
        if train_loss < best_loss:
            break
    print('{} seconds'.format(time.time() - t0))
    torch.save(model.state_dict(), params['saved_model'])
    writer.close() # tensorboard try
    return


if __name__ == "__main__":
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
        params['channels']= [16,32,64]
        params['kernel_size'] = 2
        params['lr'] = 9.93597013034459e-05
        params['optimizer'] = "Adam"
        params['batch_size'] = 32
        params['dropout_rate'] = 0.16189036200261997
    else:
        params['lr'] = 0.0009982096692042234
        params['optimizer'] = "Adam"
        params['batch_size'] = 128
        params['dropout_rate'] = 0.4642061743187629

    prefix_path = f'/../../hy-tmp/models/etcn{EDL_USED}/' if TCN_USED else f'/../../hy-tmp/models/ecnn{EDL_USED}/'
    
    
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
        filename = f'sb{sb_n}.pt' # change it later
        model_name = os.path.join(prefix_path, filename)
        params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value

        retrain(params)
