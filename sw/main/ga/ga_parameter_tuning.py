from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    param_grid = {
        "population_size": [20, 30, 40, 50],
        "generations": [50, 100],
        "crossover_rate": [0.6, 0.8, 1.0],
        "mutation_rate": [0.01, 0.05, 0.1],
        "mutation_scale": [0.05, 0.1, 0.2]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Genetic Algorithm",
        param_grid=param_grid
    )
