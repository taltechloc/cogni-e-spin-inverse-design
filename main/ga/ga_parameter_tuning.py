import joblib

from main.evaluator.hyperparameter_tuner import run_hyperparameter_tuning

if __name__ == "__main__":
    surrogate_model = joblib.load("../saved_models/XGBoost_model.pkl")  # path to your saved model


    param_grid = {
        "population_size": [20, 30, 40, 50],
        "generations": [50, 100],
        "crossover_rate": [0.6, 0.8, 1.0],
        "mutation_rate": [0.01, 0.05, 0.1],
        "mutation_scale": [0.05, 0.1, 0.2]
    }

    best_params, mean_mae, std_mae = run_hyperparameter_tuning(
        config_path="config.json",
        optimizer_name="Genetic Algorithm",
        param_grid=param_grid,
        surrogate_model=surrogate_model
    )
    print("Best hyperparameters:", best_params)
    print(f"Validation MAE: {mean_mae:.4f} ± {std_mae:.4f}")
