import argparse
import torch
import utils
import helps_pre as pre
# import optuna
import os


parser = argparse.ArgumentParser()
parser.add_argument(
    'edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl')
args = parser.parse_args()


EDL_USED = args.edl
DEVICE = pre.get_device()
EPOCHS = 10
CLASS_N = 8
TRIAL_LIST = list(range(1, 13))
DATA_PATH = '/data/'


def retrain(params):
    #  load_data

    sb_n = params['sb_n']

    train_params = {'data_path': DATA_PATH, 
                   'sb_n': sb_n,
                    'day_list': [1],
                    'time_list': [1, 2],
                    'trial_list': TRIAL_LIST,
                    'batch_size': params['batch_size']
                   }

    train_loader = pre.load_data_cnn(train_params)

    model = utils.Model(number_of_class=CLASS_N)
    model.to(DEVICE)
    optimizer = getattr(
        torch.optim,
        params['optimizer'])(model.parameters(), lr=params['lr'])
    eng = utils.EngineTrain(model, optimizer, device=DEVICE)

    loss_params = pre.update_loss_params(params)
    loss_params['device'] = DEVICE
    print(loss_params)
    
    #best_loss = params['best_loss']
    for epoch in range(1, EPOCHS + 1):
        if 'annealing_step' in loss_params:
            loss_params['epoch_num'] = epoch
        train_loss = eng.re_train(train_loader, loss_params)
        print(
            f"epoch:{epoch}, train_loss:{train_loss}")
            #f"best_loss_from_cv:{best_loss}")
        #if train_loss < best_loss:
        #    break

    torch.save(model.state_dict(), params['saved_model'])
    return


if __name__ == "__main__":

    params = {
        'class_n': CLASS_N,
        'edl_used': EDL_USED
    }

    if EDL_USED != 0:
        params['edl_fun'] = 'mse'
        params['kl'] = EDL_USED - 1

    prefix_path = f'models/ecnn{EDL_USED}/'
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
        filename = f'sb{sb_n}_temp.pt' # change it later
        model_name = os.path.join(prefix_path, filename)
        params['saved_model'] = model_name
        #params['best_loss'] = temp_best_trial.value
        params['optimizer'] = "Adam"
        params['lr'] = 1e-3
        params['batch_size'] = 512
        retrain(params)
