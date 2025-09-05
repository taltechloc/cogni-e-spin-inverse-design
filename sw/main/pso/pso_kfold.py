import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from id.dataset import Dataset
from id.models.model_type import ModelType
from id.pipeline_factory import PipelineFactory
from id.utils.plot_utils import (
    plot_target_vs_prediction_per_fold,
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
    log_file = os.path.join(run_folder, "log.txt")
    return run_folder, plots_folder, log_file


def train_surrogate_model(X, y, model_def):
    """Train a surrogate model from a model definition dict."""
    model_type = model_def["type"]
    model_params = model_def.get("params", {})
    model = ModelType.from_str(model_type).create(model_params)
    model.train(X, y)
    return model


def pad_and_average_costs(cost_histories):
    """Pad cost histories to equal length and average."""
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

def main():
    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    pipeline_cfg = config["pipeline"]
    model_cfg = config["model"]

    # Load dataset
    dataset = Dataset(dataset_cfg)
    if "diameter_stdev" in dataset.df.columns:
        dataset.df.drop("diameter_stdev", axis=1, inplace=True)
    X_full, y_full = dataset.get_features_target(scaled=False)

    # Create run folder
    run_folder, plots_folder, log_file_path = create_run_folder()

    candidate_records = []
    all_results = {"targets": [], "predictions": [], "candidates": [], "cost_histories": []}
    fold_cost_histories_all = []

    with open(log_file_path, "w") as log_f:
        kf = KFold(n_splits=5, shuffle=True, random_state=config.get("seed", 42))
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
            log_and_print(f"\n=== Fold {fold_idx} ===\n", log_f)

            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

            # Train surrogate model
            model = train_surrogate_model(X_train, y_train, model_cfg)

            fold_cost_histories = []

            for idx, target_value in enumerate(y_test):
                pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train, model)
                result = pipeline.run(float(target_value))

                candidate = result.best_candidates
                predicted = model.predict(np.array(candidate).reshape(1, -1))[0]
                abs_error = abs(predicted - target_value)

                cost_history = result.cost_history
                if isinstance(cost_history, (float, np.float32, np.float64)):
                    cost_history = [cost_history]

                # Store metrics
                fold_cost_histories.append(cost_history)
                all_results["targets"].append(target_value)
                all_results["predictions"].append(predicted)
                all_results["candidates"].append(candidate)

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

            # Save fold cost history for plotting
            fold_cost_histories_all.append(pad_and_average_costs(fold_cost_histories))

            # Fold-level metrics
            fold_r2 = r2_score(y_test, [model.predict(np.array(c).reshape(1, -1))[0] for c in all_results["candidates"][-len(y_test):]])
            log_and_print(f"\nFold {fold_idx} R²: {fold_r2:.4f}\n", log_f)

    # Save candidate DataFrame
    df_candidates = pd.DataFrame(candidate_records)
    df_candidates_path = os.path.join(run_folder, "candidates.csv")
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
