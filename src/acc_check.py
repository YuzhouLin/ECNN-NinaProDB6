import numpy as np
import pandas as pd
import os


## Functions for calculating accuracy
def get_acc(model, sb, read_folder="/results"):
    
    day_n_list = []
    time_n_list = []
    trial_n_list = []
    acc_list = []
    
    for day_n in range(1,6):
        for t_n in range(1,3):
            for T_n in range(1,13):
                tmp_R = pd.read_csv(os.getcwd()+f"{read_folder}/{model}/sb{sb}d{day_n}_t{t_n}_T{T_n}.csv")
                day_n_list.append(day_n)
                time_n_list.append('AM') if t_n == 1 else time_n_list.append('PM')
                #time_n_list.append(t_n)
                #if day_n == 1 and T_n not in valid_trials:
                #    continue
                trial_n_list.append(T_n)
                acc_list.append(np.sum(tmp_R['actual'] == tmp_R['predict'])/len(tmp_R['predict']))
    
    n = len(acc_list)
    acc_dict={
            'model': [model]*n,
            'sb': [sb]*n,
            'day': day_n_list,
            'time': time_n_list,
            'trial': trial_n_list,
            'acc': acc_list
    }
    
    #df_new = pd.DataFrame(acc_dict, index=np.arange(0,n,1))
    df = pd.DataFrame(acc_dict)
    return df


if __name__ == "__main__":
    
    folder = "/results"
    filename = './analysis/acc_report_etcn0.csv'
    models = ['ecnn0', 'ecnn1']
    sb_n = 2
    #models = ['ecnn0', 'ecnn1', 'ecnn2', 'ecnn3', 'etcn0', 'etcn1', 'etcn2', 'etcn3']

    for i in models:
        df_new = get_acc(model=i, sb=sb_n, read_folder=folder)
        if os.path.exists(filename):
            print('Update new')
            df = pd.read_csv(filename)
            #df = df.append(df_new, ignore_index=True)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            print('Create new')
            df = df_new
        df.to_csv(filename, index=False)
        
    df_acc = pd.read_csv(filename)
    df_acc_summary = df_acc.groupby(by=['day', 'time','model']).mean().unstack()
    print(df_acc_summary['acc'])