import argparse
import torch
import utils
from src import helps_pre as pre
import copy
import optuna


parser = argparse.ArgumentParser()
parser.add_argument(
    'edl', type=int, default=0,
    help='0: no edl; 1: edl without kl; 2: edl with kl (annealing); \
        3: edl with kl (trade-off)')
args = parser.parse_args()

EDL_USED = args.edl
DEVICE = pre.get_device()
DATA_PATH = '/data/NinaproDB5/raw/'


def test(params):
    #  load_data
    device = torch.device('cpu')
    test_trial = params['outer_f']
    sb_n = params['sb_n']

    # Load testing Data
    inputs, targets = pre.load_data_test_cnn(
        DATA_PATH, sb_n, test_trial)

    # Load trained model
    model = utils.Model()
    model.load_state_dict(
        torch.load(params['saved_model'], map_location=device))
    model.eval()

    # Get Results
    outputs = model(inputs.to(device)).detach()

    # Load the Testing Engine
    eng = utils.EngineTest(outputs, targets)

    common_keys_for_update_results = ['sb_n', 'edl_used', 'outer_f']

    dict_for_update_acc = \
        {key: params[key] for key in common_keys_for_update_results}
    dict_for_update_R = copy.deepcopy(dict_for_update_acc)

    eng.update_result_acc(dict_for_update_acc)

    # Get the optimal activation function
    if EDL_USED == 0:
        dict_for_update_R['acti_fun'] = 'softmax'
    else:
        # Get from hyperparameter study
        core_path = f'study/ecnn{EDL_USED}/sb{sb_n}'
        study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
        loaded_study = optuna.load_study(
            study_name="STUDY", storage=study_path)
        temp_best_trial = loaded_study.best_trial
        dict_for_update_R['acti_fun'] = temp_best_trial.params['evi_fun']

    print(dict_for_update_R)
    eng.update_result_R(dict_for_update_R)

    return


if __name__ == "__main__":

    params = {'edl_used': EDL_USED}

    for test_trial in range(1, 7):
        params['outer_f'] = test_trial
        for sb_n in range(1, 11):
            params['sb_n'] = sb_n
            model_name = f"models/ecnn{EDL_USED}/sb{sb_n}_t{test_trial}.pt"
            params['saved_model'] = model_name
            test(params)
            print(f'Testing done. sb{sb_n}-t{test_trial}')

