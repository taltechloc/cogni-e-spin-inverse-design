from config import DataConfig, Global, xgb_config, pso_config
from id.data.data_loader import DataLoader
from id.models.model_type import ModelType
from id.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from datetime import datetime
import os
from id.utils.plot_utils import (
    plot_target_vs_prediction_per_fold,
    plot_target_vs_prediction_overall,
    plot_cost_trajectories
)

# -------------------------
# Helper functions
# -------------------------

def create_run_folder(base="main"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base, timestamp)
    plots_folder = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    return run_folder, plots_folder, os.path.join(run_folder, "log.txt")


def train_surrogate_model(X, y):
    model = ModelType.from_str('XGBoostSurrogate').create(xgb_config)
    model.train(X, y)
    return model


def pad_and_average_costs(cost_histories):
    max_len = max(len(ch) for ch in cost_histories)
    padded = [np.pad(ch, (0, max_len - len(ch)), constant_values=np.nan) for ch in cost_histories]
    return np.nanmean(padded, axis=0)


def log_and_print(message, log_file):
    print(message, end="")
    log_file.write(message)
    log_file.flush()

# -------------------------
# Main workflow
# -------------------------

def main():
    # Load data
    df = DataLoader(DataConfig).get_dataframe()
    df.drop("diameter_stdev", axis=1, inplace=True)
    X_full, y_full = df.drop(columns=[DataConfig.target_column]), df[DataConfig.target_column]

    # Create run folder
    run_folder, plots_folder, log_file = create_run_folder()

    all_results = {"targets": [], "predictions": [], "candidates": [],
                   "cost_histories": [], "mae_list": [], "rmse_list": [], "r2_list": []}
    fold_cost_histories_all = []

    candidate_records = []  # for storing dataframe of candidates

    with open(log_file, "w") as log_f:
        kf = KFold(n_splits=5, shuffle=True, random_state=Global.seed)
        all_fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
            log_f.write(f"\n=== Fold {fold_idx} ===\n")
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            model = train_surrogate_model(X_train, y_train)

            fold_metrics = {"abs_errors": [], "y_test": [], "y_pred": [],
                            "cost_histories": [], "candidates": []}

            for idx, target_value in enumerate(y_test):
                pipeline = Pipeline(optimizer_config=pso_config, data_x=X_train, model=model)
                result = pipeline.run(target_value)

                candidate = result.best_candidates
                predicted = model.predict(np.array(candidate).reshape(1, -1))[0]
                abs_error = abs(predicted - target_value)

                cost_history = result.cost_history
                if isinstance(cost_history, (float, np.float32, np.float64)):
                    cost_history = [cost_history]

                # Store metrics
                fold_metrics["abs_errors"].append(abs_error)
                fold_metrics["y_test"].append(target_value)
                fold_metrics["y_pred"].append(predicted)
                fold_metrics["cost_histories"].append(cost_history)
                fold_metrics["candidates"].append(candidate)

                # Record for candidate DataFrame
                candidate_records.append({
                    "real_inputs": X_test.iloc[idx].tolist(),
                    "given_candidate": candidate,
                    "target": target_value,
                    "predicted": predicted
                })

                # Log per sample
                log_and_print(
                    f"------\nTest index: {X_test.index[idx]}\n"
                    f"Target: {target_value}\n"
                    f"Candidate: {candidate}\n"
                    f"Predicted: {predicted}\n"
                    f"Absolute error: {abs_error}\n"
                    f"Final cost: {cost_history[-1]}\n------\n",
                    log_f
                )

            mae_mean = np.mean(fold_metrics["abs_errors"])
            mae_std = np.std(fold_metrics["abs_errors"])
            rmse_mean = np.sqrt(np.mean(np.square(fold_metrics["abs_errors"])))
            rmse_std = np.sqrt(np.mean(np.square(fold_metrics["abs_errors"])))
            fold_r2 = r2_score(fold_metrics["y_test"], fold_metrics["y_pred"])
            fold_cost_histories_all.append(pad_and_average_costs(fold_metrics["cost_histories"]))

            log_and_print(
                f"\nFold {fold_idx} averages:\n"
                f"MAE: {mae_mean:.4f} ± {mae_std:.4f}\n"
                f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}\n"
                f"R²: {fold_r2:.4f}\n",
                log_f
            )

            all_results["targets"].extend(fold_metrics["y_test"])
            all_results["predictions"].extend(fold_metrics["y_pred"])
            all_results["candidates"].extend(fold_metrics["candidates"])
            all_fold_metrics.append({"mae_mean": mae_mean, "mae_std": mae_std,
                                     "rmse_mean": rmse_mean, "rmse_std": rmse_std, "r2": fold_r2})

    # Save candidate DataFrame
    df_candidates = pd.DataFrame(candidate_records)
    df_candidates_path = os.path.join(run_folder, "pso_candidates.csv")
    df_candidates.to_csv(df_candidates_path, index=False)
    print(f"Candidate DataFrame saved to {df_candidates_path}")

    # Plot results
    plot_target_vs_prediction_per_fold(
        all_results, "PSO", n_folds=5,
        save_path=os.path.join(plots_folder, "target_vs_prediction_per_fold.png")
    )

    plot_cost_trajectories(
        fold_cost_histories_all, "PSO",
        save_path=os.path.join(plots_folder, "cost_trajectories.png")
    )


if __name__ == "__main__":
    main()
