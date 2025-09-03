from config import DataConfig, Global, xgb_config, pso_config
from id.data.data_loader import DataLoader
from id.models.model_type import ModelType
from id.objective.surrogate_objective import SurrogateObjective
from id.optimizers.optimizer_type import OptimizerType
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
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


def create_boundaries(X):
    return [(X[col].min(), X[col].max()) for col in X.columns]


def train_surrogate_model(X, y):
    model = ModelType.from_str('XGBoostSurrogate').create(xgb_config)
    model.train(X, y)
    return model


def pad_and_average_costs(cost_histories):
    max_len = max(len(ch) for ch in cost_histories)
    padded = [np.pad(ch, (0, max_len - len(ch)), constant_values=np.nan) for ch in cost_histories]
    return np.nanmean(padded, axis=0)


def evaluate_fold(X_train, y_train, X_test, y_test, log_f):
    """Train surrogate, optimize, and log results for a single fold."""
    boundaries = create_boundaries(X_train)
    model = train_surrogate_model(X_train, y_train)
    log_f.write(f"Model training evaluation: {model.evaluate(X_train, y_train)}\n")
    log_f.write(f"Model testing evaluation: {model.evaluate(X_test, y_test)}\n")
    log_f.write(f"Test targets: {list(y_test)}\n")

    objective = SurrogateObjective(model)
    optimizer = OptimizerType.PSO.create(config=pso_config, objective=objective, boundaries=boundaries)

    fold_metrics = {"mae": [], "rmse": [], "y_test": [], "y_pred": [], "cost_histories": []}
    candidates_list = []

    for idx, target_value in enumerate(y_test):
        result = optimizer.optimize(target_value)
        candidate = result.best_candidates
        predicted = model.predict(np.array(candidate).reshape(1, -1))[0]
        abs_error = abs(predicted - target_value)

        cost_history = result.cost_history
        if isinstance(cost_history, (float, np.float32, np.float64)):
            cost_history = [cost_history]

        fold_metrics["mae"].append(abs_error)
        fold_metrics["rmse"].append(abs_error)
        fold_metrics["y_test"].append(target_value)
        fold_metrics["y_pred"].append(predicted)
        fold_metrics["cost_histories"].append(cost_history)
        candidates_list.append(candidate)

        log_f.write(
            f"------\nTest index: {X_test.index[idx]}\n"
            f"Target: {target_value}\n"
            f"X_test: {X_test.iloc[idx].values}\n"
            f"Candidate: {candidate}\n"
            f"Predicted: {predicted}\n"
            f"Absolute error: {abs_error}\n"
            f"Final cost: {cost_history[-1]}\n------\n"
        )

    fold_avg_cost = pad_and_average_costs(fold_metrics["cost_histories"])
    fold_r2 = r2_score(fold_metrics["y_test"], fold_metrics["y_pred"])
    return fold_metrics, fold_avg_cost, fold_r2, candidates_list


def log_fold_summary(log_f, fold_idx, fold_metrics, fold_r2):
    mae_mean, mae_std = np.mean(fold_metrics["mae"]), np.std(fold_metrics["mae"])
    rmse_mean, rmse_std = np.mean(fold_metrics["rmse"]), np.std(fold_metrics["rmse"])

    log_f.write(f"\nFold {fold_idx} averages:\n"
                f"MAE: {mae_mean:.4f} ± {mae_std:.4f}\n"
                f"RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}\n"
                f"R²: {fold_r2:.4f}\n")

    return {"mae_mean": mae_mean, "mae_std": mae_std,
            "rmse_mean": rmse_mean, "rmse_std": rmse_std,
            "r2": fold_r2}


def plot_results(all_results, fold_cost_histories, plots_folder):
    plot_target_vs_prediction_per_fold(
        all_results, "PSO", n_folds=5,
        save_path=os.path.join(plots_folder, "target_vs_prediction_per_fold.png")
    )
    plot_target_vs_prediction_overall(
        all_results, "PSO",
        save_path=os.path.join(plots_folder, "target_vs_prediction_overall.png")
    )
    plot_cost_trajectories(
        fold_cost_histories, "PSO",
        save_path=os.path.join(plots_folder, "cost_trajectories.png")
    )


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

    with open(log_file, "w") as log_f:
        kf = KFold(n_splits=5, shuffle=True, random_state=Global.seed)
        all_fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
            log_f.write(f"\n=== Fold {fold_idx} ===\n")
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            fold_metrics, fold_avg_cost, fold_r2, candidates_list = evaluate_fold(X_train, y_train, X_test, y_test, log_f)
            fold_cost_histories_all.append(fold_avg_cost)

            all_results["targets"].extend(fold_metrics["y_test"])
            all_results["predictions"].extend(fold_metrics["y_pred"])
            all_results["candidates"].extend(candidates_list)
            print(all_results["candidates"])

            fold_summary = log_fold_summary(log_f, fold_idx, fold_metrics, fold_r2)
            all_fold_metrics.append(fold_summary)

        # Aggregate overall metrics
        for metric in ["mae", "rmse", "r2"]:
            all_results[f"{metric}_list"] = [f"{metric}_mean" for f in all_fold_metrics]

    # Plot results
    plot_results(all_results, fold_cost_histories_all, plots_folder)


if __name__ == "__main__":
    main()
