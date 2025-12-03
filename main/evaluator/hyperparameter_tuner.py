import json
import itertools
import numpy as np
from copy import deepcopy
from sklearn.metrics import r2_score
from eSpinID.dataset import Dataset
from eSpinID.pipeline_factory import PipelineFactory
import os
from datetime import datetime


def mean_std_string(values):
    """Return formatted mean ± std string."""
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


def run_hyperparameter_tuning(config_path, optimizer_name, param_grid, surrogate_model):
    """
    Hyperparameter tuning for optimizer with mean ± std reporting.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    dataset = Dataset(dataset_cfg)
    X_full, y_full = dataset.get_features_target(scaled=False)

    run_folder = "hyperparameter_tuning"
    os.makedirs(run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(run_folder, f"{optimizer_name.replace(' ', '_')}_tuning_{timestamp}.txt")

    results = []

    with open(log_file_path, "w") as log_f:
        def log(msg):
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        # Use full data as evaluation targets
        y_targets = y_full

        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        log(f"=== Optimizer Hyperparameter Tuning: {optimizer_name} ===")
        log(f"Total combinations: {len(param_combinations)}")
        log(f"Number of target evaluations: {len(y_targets)}\n")

        for i, params in enumerate(param_combinations, 1):
            pipeline_cfg = deepcopy(config["pipeline"])
            pipeline_cfg["optimizer"].update(params)
            pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_full, surrogate_model)

            # Evaluate optimizer for all targets
            y_preds = []
            for target in y_targets:
                result = pipeline.run(float(target))
                predicted = surrogate_model.predict(result.best_candidates.reshape(1, -1))[0]
                y_preds.append(predicted)

            y_preds = np.array(y_preds)

            # Calculate errors
            errors = np.abs(y_preds - y_targets)
            squared_errors = (y_preds - y_targets) ** 2

            # Calculate metrics
            mae_mean = np.mean(errors)
            mae_std = np.std(errors, ddof=1)

            rmse_values = np.sqrt(squared_errors)
            rmse_mean = np.mean(rmse_values)
            rmse_std = np.std(rmse_values, ddof=1)

            r2 = r2_score(y_targets, y_preds)

            # Format with mean ± std
            mae_str = f"{mae_mean:.3f} ± {mae_std:.3f}"
            rmse_str = f"{rmse_mean:.3f} ± {rmse_std:.3f}"

            log(f"{i}/{len(param_combinations)} | MAE: {mae_str} | RMSE: {rmse_str} | R²: {r2:.3f} | params: {params}")

            # Store for selection (using mean MAE)
            results.append((mae_mean, errors, rmse_values, r2, params, y_preds))

        # Select best hyperparameters based on mean MAE
        results.sort(key=lambda x: x[0])
        best_mae_mean, best_errors, best_rmse, best_r2, best_params, best_preds = results[0]

        # Final metrics for best parameters
        final_mae_str = mean_std_string(best_errors)
        final_rmse_str = mean_std_string(best_rmse)

        log("\n=== Best Optimizer Hyperparameters ===")
        log(str(best_params))
        log(f"\nValidation metrics (mean ± std):")
        log(f"MAE:  {final_mae_str}")
        log(f"RMSE: {final_rmse_str}")
        log(f"R²:   {best_r2:.3f}")

        # Also show raw values for reference
        log(f"\nRaw values:")
        log(f"MAE mean: {np.mean(best_errors):.3f}")
        log(f"MAE std:  {np.std(best_errors, ddof=1):.3f}")
        log(f"RMSE mean: {np.mean(best_rmse):.3f}")
        log(f"RMSE std:  {np.std(best_rmse, ddof=1):.3f}")

    print(f"\nAll logs saved in: {log_file_path}")

    # Return both formatted strings and raw values
    return {
        'best_params': best_params,
        'mae': final_mae_str,
        'rmse': final_rmse_str,
        'r2': best_r2,
        'mae_raw': (np.mean(best_errors), np.std(best_errors, ddof=1)),
        'rmse_raw': (np.mean(best_rmse), np.std(best_rmse, ddof=1)),
        'r2_raw': best_r2
    }