import optuna
import os

# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).

if __name__ == "__main__":
    '''
    # Let us minimize the objective function above.
    sampler = optuna.samplers.CmaEsSampler(n_startup_trials=1)
    print("Running 2 trials...")
    study = optuna.study.create_study(sampler=sampler)
    study.optimize(objective, n_trials=2)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
    '''
    loaded_study = optuna.load_study(study_name="SB1_STUDY2", storage="sqlite:///example.db")
    #importances = optuna.importance.get_param_importances(loaded_study)
    
    fig = optuna.visualization.plot_param_importances(loaded_study)
    cwd = os.getcwd()
    fig.write_image(cwd+"/results/hyper_importance.png")
    
    trial_ = loaded_study.best_trial
    print(trial_.params)
