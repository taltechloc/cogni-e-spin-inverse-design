from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    bo_param_grid = {
        "n_init": [5, 10, 15],
        "n_iter": [30, 50, 70],
        "acquisition_function": ["EI", "PI", "UCB"],
        "kappa": [1.5, 2.5],

        "xi": [0.005, 0.01, 0.02],
        "early_stop_patience": [20]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Bayesian Optimization",
        param_grid=bo_param_grid
    )
