from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    de_param_grid = {
        "population_size": [30, 50, 70],
        "generations": [80, 110, 140, 170],
        "crossover_rate": [0.6, 0.75, 0.9],
        "mutation_factor": [0.3, 0.45, 0.6, 0.75]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Differential Evolution",
        param_grid=de_param_grid
    )
