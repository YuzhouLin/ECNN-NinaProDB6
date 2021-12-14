import torch
import utils
from src import helps_pre as pre
import optuna
import pandas as pd


DEVICE = pre.get_device()
EPOCHS = 100
TRIAL_LIST = list(range(1, 7))
DATA_PATH = '/data/NinaproDB5/raw/'


if __name__ == "__main__":

    '''
    for test_trial in range(1, 7):
        for sb_n in range(1, 11):
            core_path = f'study/ecnn/sb{sb_n}'
            study_path = "sqlite:///" + core_path + f"/t{test_trial}.db"
            loaded_study = optuna.load_study(
                study_name="STUDY", storage=study_path)
            temp_best_trial = loaded_study.best_trial
            # Update for the optimal hyperparameters
            for key, value in temp_best_trial.params.items():
                print(key, value)
                break
    '''
    rec_r1 = pd.read_csv('results/cv/accuracy.csv')
    rec_r2 = pd.read_csv('results/cv/accuracy_temp.csv') 
    rec_r = pd.concat([rec_r1, rec_r2], ignore_index=True)
    print(rec_r)
    rec_r.to_csv('results/cv/accuracy.csv', index=False)


    mis_r1 = pd.read_csv('results/cv/reliability.csv')
    mis_r2 = pd.read_csv('results/cv/reliability_temp.csv')
    mis_r = pd.concat([mis_r1, mis_r2], ignore_index=True)
    print(mis_r)
    mis_r.to_csv('results/cv/reliability.csv', index=False)

    '''
    rec_r = pd.read_csv('results/cv/accuracy.csv')
    
    rec_r = rec_r.rename({"edl": "edl_used"}, axis="columns")
    rec_r = rec_r.replace(False, 0)
    rec_r.to_csv('results/cv/accuracy.csv', index=False)

    print('rec done')

    mis_r = pd.read_csv('results/cv/reliability.csv')
    mis_r = mis_r.rename({"edl": "edl_used"}, axis="columns")
    mis_r = mis_r.replace(False, 0)
    mis_r.to_csv('results/cv/reliability.csv', index=False)

    print('R done')
    '''


