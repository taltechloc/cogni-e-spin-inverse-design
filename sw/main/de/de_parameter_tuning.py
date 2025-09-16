from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    de_param_grid = {
        "population_size": [50, 80, 100],
        "generations": [100, 140, 180],
        "crossover_rate": [0.8, 0.9],
        "mutation_factor": [0.3, 0.5, 0.7],
        "early_stop_patience": [20, 50]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Differential Evolution",
        param_grid=de_param_grid
    )