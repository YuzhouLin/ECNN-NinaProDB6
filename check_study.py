import optuna

study_name = 'etcn2_sb1'
study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{study_name}.db")

for i in range(len(study.trials)):
    print(study.trials[i])
    print('-----------------------')
