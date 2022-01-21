import optuna

study = optuna.load_study(study_name="etcn1_sb1", storage="sqlite:///etcn1_sb1.db")

for i in range(len(study.trials)):
    print(study.trials[i])
