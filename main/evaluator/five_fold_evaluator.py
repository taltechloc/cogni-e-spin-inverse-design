# eSpinID/evaluator/five_fold_evaluator.py

import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from eSpinID.dataset import Dataset
from eSpinID.models.model_type import ModelType
from eSpinID.pipeline_factory import PipelineFactory
from eSpinID.utils.plot_utils import plot_target_vs_prediction_per_fold, plot_cost_trajectories


# -------------------------
# Helper functions
# -------------------------
def create_run_folder(base="run"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base, timestamp)
    plots_folder = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    log_file = os.path.join(run_folder, "log.txt")
    return run_folder, plots_folder, log_file

def train_surrogate_model(X, y, model_def):
    model_type = model_def["type"]
    model_params = model_def.get("params", {})
    model = ModelType.from_str(model_type).create(model_params)
    model.train(X, y)
    return model

def pad_and_average_costs(cost_histories):
    max_len = max(len(ch) for ch in cost_histories)
    padded = [np.pad(ch, (0, max_len - len(ch)), constant_values=np.nan) for ch in cost_histories]
    return np.nanmean(padded, axis=0)

def log_and_print(message, log_file_handle):
    print(message, end="")
    log_file_handle.write(message)
    log_file_handle.flush()


# -------------------------
# Main workflow
# -------------------------
def run_evaluation(config_path: str, optimizer_name: str):
    total_start = time.time()

    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    pipeline_cfg = config["pipeline"]
    model_cfg = config["model"]

    # Load dataset
    dataset = Dataset(dataset_cfg)
    X_full, y_full = dataset.get_features_target(scaled=False)

    # Create run folder per optimizer
    run_folder, plots_folder, log_file_path = create_run_folder(base=optimizer_name)

    candidate_records = []
    all_results = {"targets": [], "predictions": [], "candidates": [], "cost_histories": []}
    fold_cost_histories_all = []
    fold_metrics_list = []
    fold_times_all = []  # store per-candidate evaluation times per fold

    # -------------------------
    # Keep log file open for entire evaluation
    # -------------------------
    with open(log_file_path, "w") as log_f:
        kf = KFold(n_splits=5, shuffle=True, random_state=config.get("seed", 42))
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
            log_and_print(f"\n=== Fold {fold_idx} ===\n", log_f)
            fold_start_time = time.time()

            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            # Train surrogate model
            model = train_surrogate_model(X_train, y_train, model_cfg)

            fold_cost_histories = []
            fold_targets = []
            fold_predictions = []
            fold_candidate_times = []  # per-candidate timing

            for idx, target_value in enumerate(y_test):
                candidate_start_time = time.time()

                pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train, model)
                result = pipeline.run(float(target_value))

                candidate_elapsed_time = time.time() - candidate_start_time
                fold_candidate_times.append(candidate_elapsed_time)

                candidate = result.best_candidates
                predicted = model.predict(np.array(candidate).reshape(1, -1))[0]
                abs_error = abs(predicted - target_value)

                cost_history = result.cost_history
                if isinstance(cost_history, (float, np.float32, np.float64)):
                    cost_history = [cost_history]

                # Get top 5 candidates
                top_candidates = getattr(result, "top_candidates", [])[:5]

                fold_cost_histories.append(cost_history)
                all_results["targets"].append(target_value)
                all_results["predictions"].append(predicted)
                all_results["candidates"].append(candidate)
                all_results["cost_histories"].append(cost_history)

                fold_targets.append(target_value)
                fold_predictions.append(predicted)

                candidate_records.append({
                    "real_inputs": X_test.iloc[idx].tolist(),
                    "given_candidate": candidate,
                    "target": target_value,
                    "predicted": predicted,
                    "top_candidates": top_candidates,
                    "eval_time_sec": candidate_elapsed_time
                })

                log_and_print(
                    f"------\nTest index: {X_test.index[idx]}\n"
                    f"Target: {target_value}\n"
                    f"Candidate: {candidate}\n"
                    f"Predicted: {predicted}\n"
                    f"Absolute error: {abs_error}\n"
                    f"Final cost: {cost_history[-1]}\n"
                    f"Top 5 Candidates:\n    " + "\n    ".join(str(c) for c in top_candidates) + "\n"
                    f"Evaluation time: {candidate_elapsed_time:.4f} sec\n------\n",
                    log_f
                )

            fold_cost_histories_all.append(pad_and_average_costs(fold_cost_histories))
            fold_times_all.append(fold_candidate_times)

            fold_r2 = r2_score(fold_targets, fold_predictions)
            fold_rmse = np.sqrt(mean_squared_error(fold_targets, fold_predictions))
            fold_mae = mean_absolute_error(fold_targets, fold_predictions)

            fold_metrics_list.append({
                "r2": fold_r2,
                "rmse": fold_rmse,
                "mae": fold_mae
            })

            fold_elapsed = time.time() - fold_start_time
            mean_time = np.mean(fold_candidate_times)
            std_time = np.std(fold_candidate_times)

            log_and_print(
                f"\nFold {fold_idx} metrics:\n"
                f"  R²: {fold_r2:.4f}\n"
                f"  RMSE: {fold_rmse:.4f}\n"
                f"  MAE: {fold_mae:.4f}\n"
                f"Fold total time: {fold_elapsed:.2f} sec\n"
                f"Fold candidate evaluation time: {mean_time:.4f} ± {std_time:.4f} sec\n",
                log_f
            )

        # -------------------------
        # Save candidate DataFrame and timing summary
        # -------------------------
        df_candidates = pd.DataFrame(candidate_records)
        df_candidates_path = os.path.join(run_folder, "candidates.csv")
        df_candidates.to_csv(df_candidates_path, index=False)
        log_and_print(f"\nCandidate DataFrame saved to {df_candidates_path}\n", log_f)

        # Save candidate time summary
        df_time_summary = pd.DataFrame(fold_times_all).describe()
        df_time_summary_path = os.path.join(run_folder, "candidate_time_summary.csv")
        df_time_summary.to_csv(df_time_summary_path)
        log_and_print(f"Candidate time summary saved to {df_time_summary_path}\n", log_f)

        # Compute cross-validation summary
        r2_values = [m["r2"] for m in fold_metrics_list]
        rmse_values = [m["rmse"] for m in fold_metrics_list]
        mae_values = [m["mae"] for m in fold_metrics_list]

        summary = (
            f"\n=== Cross-Validation Summary ===\n"
            f"R²:   {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}\n"
            f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}\n"
            f"MAE:  {np.mean(mae_values):.4f} ± {np.std(mae_values):.4f}\n"
        )
        log_and_print(summary, log_f)

        # Overall candidate evaluation time across 5 folds
        all_times = np.concatenate(fold_times_all)
        overall_mean = np.mean(all_times)
        overall_std = np.std(all_times)
        log_and_print(
            f"\nOverall candidate evaluation time across 5 folds: {overall_mean:.4f} ± {overall_std:.4f} sec\n", log_f
        )

        total_elapsed = time.time() - total_start
        log_and_print(f"Total optimizer run time: {total_elapsed:.2f} sec\n", log_f)

    # -------------------------
    # Generate plots
    # -------------------------
    plot_target_vs_prediction_per_fold(
        all_results, optimizer_name, np.mean(r2_values), n_folds=5,
        save_path=os.path.join(plots_folder, "target_vs_prediction_per_fold.png")
    )

    plot_cost_trajectories(
        fold_cost_histories_all, optimizer_name,
        save_path=os.path.join(plots_folder, "cost_trajectories.png")
    )
