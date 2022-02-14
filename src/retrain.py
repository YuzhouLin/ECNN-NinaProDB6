import argparse
import torch
import utils
import helps_pre as pre
# import optuna
import os
import time
import yaml
from easydict import EasyDict as edict

#from torch.utils.tensorboard import SummaryWriter

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

# Concern if training loss is less than best loss already without training
# check if this is the most case, if so, train 1 epoch or few epochs instead


def retrain(cfg):
    # load_data
    train_trail_list = list(range(1,cfg.DATA_CONFIG.trial_n+1))

    train_loader = pre.load_data(cfg.DATA_PATH, cfg.DATA_CONFIG.sb_n, cfg.DATA_CONFIG.day_list, cfg.DATA_CONFIG.time_list, train_trail_list, tcn_used=TCN_USED, batch_size=cfg.HP.batch_size, shuffle=cfg.DATA_LOADER.shuffle, drop_last=cfg.DATA_LOADER.drop_last, num_workers=cfg.DATA_LOADER.num_workers, pin_memory=cfg.DATA_LOADER.pin_memory)

    n_class = len(cfg.CLASS_NAMES)
    # Load Model
    if TCN_USED:
        model = utils.TCN(input_size=cfg.DATA_CONFIG.channel_n, output_size=n_class, num_channels=cfg.HP.tcn_channels, kernel_size=cfg.HP.kernel_size, dropout=cfg.HP.dropout_rate)
    else:
        model = utils.Model(number_of_class=n_class, dropout=cfg.HP.dropout_rate)
    
    saved_model_path = cfg.model_path+f'/{cfg.TRAINING.model_name}_sb{cfg.DATA_CONFIG.sb_n}.pt'
    checkpoint = torch.load(saved_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,cfg.HP.optimizer)(model.parameters(), lr=cfg.HP.lr)

    eng = utils.EngineTrain(model, optimizer, device=DEVICE)
    
    loss_params = {'edl_used': EDL_USED, 'device': DEVICE}
    if EDL_USED != 0:
        loss_params['class_n'] = n_class
        for item in cfg.HP_SEARCH[f'EDL{EDL_USED}']:
            loss_params[item]=cfg.HP[item]


    t0 = time.time()
    for epoch in range(1, cfg.RETRAINING.epochs + 1): 
        # if 'annealing_step' in loss_params:
        if EDL_USED == 2:
            loss_params['epoch_num'] = epoch

        train_loss = eng.re_train(train_loader, loss_params)
        print(
            f"epoch:{epoch}, train_loss:{train_loss}")

        #if train_loss < cfg.best_loss:
        #    break
    t1 = time.time() - t0
    print('Training completed in {} seconds'.format(t1))

    torch.save({
    'model_state_dict': model.state_dict(),
    'train_loss': train_loss,
    'train_time': t1,
    'epoch': epoch
    }, cfg.model_path+f'/{cfg.RETRAINING.model_name}{cfg.RETRAINING.epochs}_sb{cfg.DATA_CONFIG.sb_n}.pt') # modify it later
    return



if __name__ == "__main__":
    # Load config file from hpo search
    with open("hpo_search.yaml", 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
        
    # Load determined optimal hyperparameters
    study_dir = f'etcn{EDL_USED}' if TCN_USED else f'ecnn{EDL_USED}'
    study_path = os.getcwd() + cfg.STUDY_PATH + study_dir
    with open(f'{study_path}/sb_{cfg.DATA_CONFIG.sb_n}', 'r') as f:
        hp = yaml.load(f, Loader=yaml.SafeLoader)
    
    #cfg['best_loss'] = hp[0]['best_loss']
    cfg.HP = {}
    for key, item in hp[1].items():
        cfg.HP[key] = item

    #cfg.TRAINING.epochs = 10
    # retraining and save the models
    retrain(cfg)
