from id.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    # Define the hyperparameter grid for Simulated Annealing (SA)
    param_grid = {
        "initial_temperature": [50.0, 100.0, 150.0],
        "cooling_rate": [0.85, 0.90, 0.95],
        "iterations_per_temp": [10, 20, 30],
        "step_size": [0.1, 0.2, 0.3]
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Simulated Annealing",
        param_grid=param_grid
    )
