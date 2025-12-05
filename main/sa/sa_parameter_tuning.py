import joblib

from main.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    surrogate_model = joblib.load("../saved_models/XGBoost_model.pkl")  # path to your saved model

    param_grid = {
        "initial_temperature": [50, 100, 150, 200, 300],
        "cooling_rate": [0.80, 0.85, 0.90, 0.95, 0.98],
        "iterations_per_temp": [20, 50, 100],
        "step_size": [0.1, 0.2],
    }

    best_params, mean_mae, std_mae = run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Simulated Annealing",
        param_grid=param_grid,
        surrogate_model=surrogate_model
    )

    print("Best hyperparameters:", best_params)
    print(f"Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}")
