from eSpinID.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    # ~150 total combinations
    param_grid = {
        "initial_temperature": [50, 100, 150, 200, 300],
        "cooling_rate": [0.80, 0.85, 0.90, 0.95, 0.98],
        "iterations_per_temp": [20, 50, 100],
        "step_size": [0.1, 0.2],
    }

    run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Simulated Annealing",
        param_grid=param_grid
    )
