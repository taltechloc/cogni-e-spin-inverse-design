# eSpinID/evaluator/heldout_evaluator.py
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import t
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib
matplotlib.use("Agg")  # safe backend for headless servers
import matplotlib.pyplot as plt

from eSpinID.dataset import Dataset
from eSpinID.pipeline_factory import PipelineFactory
from eSpinID.utils.plot_utils import plot_target_vs_prediction, plot_cost_trajectories


# -------------------------
# Helper functions
# -------------------------
def create_run_folder(base: str = "run"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base, timestamp)
    plots_folder = os.path.join(run_folder, "plots")
    os.makedirs(plots_folder, exist_ok=True)
    log_file = os.path.join(run_folder, "log.txt")
    return run_folder, plots_folder, log_file


def pad_and_average_costs(cost_histories: List[List[float]]) -> np.ndarray:
    if len(cost_histories) == 0:
        return np.array([])
    max_len = max(len(ch) for ch in cost_histories)
    padded = [
        np.pad(ch, (0, max_len - len(ch)), constant_values=np.nan)
        for ch in cost_histories
    ]
    return np.nanmean(padded, axis=0)


def compute_ci_from_samples(samples: np.ndarray, confidence: float = 0.95):
    """
    t-distribution two-sided CI for the mean computed from raw sample values.
    Returns (ci_low, ci_high). If n<=1, returns (mean, mean).
    """
    if samples is None:
        return float("nan"), float("nan")
    samples = np.asarray(samples, dtype=float)
    n = samples.size
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.nanmean(samples))
    if n <= 1:
        return mean, mean
    std = float(np.nanstd(samples, ddof=1))
    t_val = t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    margin = t_val * (std / np.sqrt(n))
    return mean - margin, mean + margin


def format_ci(mean: float, std: float, ci_low: float, ci_high: float):
    return f"{mean:.3f} ± {std:.3f} [{ci_low:.3f}, {ci_high:.3f}]"


def log_and_print(message: str, log_file_handle):
    if not message.endswith("\n"):
        message = message + "\n"
    print(message, end="")
    try:
        log_file_handle.write(message)
        log_file_handle.flush()
    except Exception:
        # don't crash on logging failures
        pass


def _build_replica_series(predictions_replicas: List[List[float]], n_replicas: int):
    """
    Convert per-sample list-of-replicas into replica-wise list-of-lists.
    Input: predictions_replicas[sample_idx] = [rep1_pred, rep2_pred, ...]
    Output: replica_series[replica_idx] = [pred_for_sample0, pred_for_sample1, ...]
    """
    n_samples = len(predictions_replicas)
    replica_series = [[np.nan] * n_samples for _ in range(n_replicas)]
    for s_idx, preds in enumerate(predictions_replicas):
        for r_idx in range(min(n_replicas, len(preds))):
            replica_series[r_idx][s_idx] = preds[r_idx]
    return replica_series


def plot_spaghetti_predictions(
    all_results: Dict[str, Any],
    optimizer_name: str,
    save_path: str,
    sample_indices: List[int] = None
):
    """
    Spaghetti plot:
    - One line per replica across samples
    - Integer x-axis
    - Color-coded replicas with legend entries 'Replica 1', 'Replica 2', ...
    """
    preds_replicas = all_results.get("predictions_replicas", [])
    preds_mean = all_results.get("predictions_mean", [])
    targets = all_results.get("targets", [])

    if len(preds_replicas) == 0 or len(targets) == 0:
        return

    n_samples = len(preds_replicas)
    n_replicas = max(len(pr) for pr in preds_replicas)
    replica_series = _build_replica_series(preds_replicas, n_replicas)

    # X-axis forced to integer positions
    if sample_indices is None:
        x = np.arange(n_samples, dtype=int)
    else:
        x = np.array(sample_indices, dtype=int)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()

    # Color map for replicas (tab10 supports up to 10 distinct colors)
    cmap = plt.cm.get_cmap('tab10', max(1, n_replicas))

    for r_idx, series in enumerate(replica_series, start=1):
        ax.plot(
            x,
            series,
            linestyle='-',
            marker='o',
            linewidth=1.2,
            markersize=4,
            alpha=0.75,
            color=cmap(r_idx - 1),
            label=f"Replica {r_idx}"
        )

    # mean (black dashed) and targets
    ax.plot(
        x,
        preds_mean,
        linestyle='--',
        marker='s',
        linewidth=2.3,
        markersize=6,
        color='black',
        label='Mean Prediction'
    )

    ax.plot(
        x,
        targets,
        linestyle='',
        marker='x',
        markersize=8,
        color='red',
        label='Target'
    )

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Prediction / Target Nanofiber Diameter (nm)")
    ax.set_title(f"{optimizer_name}")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=9)

    # Force integer x ticks (one per sample or subset)
    if x.size <= 50:
        ax.set_xticks(x)
    else:
        # avoid overcrowding: show a subset but ensure integer ticks
        nticks = 20
        idxs = np.linspace(0, x.size - 1, nticks, dtype=int)
        ax.set_xticks(x[idxs])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_fn, n_bootstrap: int = 1000, seed: int = 0):
    """
    Nonparametric paired bootstrap for a metric computed on (y_true, y_pred).
    Returns: mean, std, ci_low (percentile 2.5), ci_high (percentile 97.5), boots_array
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), np.array([])

    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)  # with replacement
        try:
            val = float(metric_fn(y_true[idx], y_pred[idx]))
            if np.isfinite(val):
                boots.append(val)
        except Exception:
            # skip invalid resamples (e.g., degenerate R^2)
            continue

    boots = np.array(boots, dtype=float)
    if boots.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), boots

    mean_b = float(np.mean(boots))
    std_b = float(np.std(boots, ddof=1)) if boots.size > 1 else 0.0
    ci_low_p, ci_high_p = np.percentile(boots, [2.5, 97.5])
    return mean_b, std_b, ci_low_p, ci_high_p, boots


# -------------------------
# Main workflow
# -------------------------
def run_evaluation(
    trained_model,
    config_path: str,
    optimizer_name: str,
    n_replicas: int = 5,
    seed: int = 42,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 12345,
):
    total_start = time.time()

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    pipeline_cfg = config["pipeline"]

    # Load dataset (expects held-out test set returned here)
    dataset = Dataset(dataset_cfg)
    X_test, y_test = dataset.get_features_target(scaled=False)

    # Create run folder
    run_folder, plots_folder, log_file_path = create_run_folder(base=optimizer_name)

    candidate_records = []
    all_results = {
        "targets": [],
        "predictions_mean": [],
        "predictions_replicas": [],
        "candidates": [],
        "cost_histories": []
    }

    cost_histories_all_samples = []
    times_all_samples = []

    # maintain legacy seeding for reproducibility; bootstrap uses its own RNG
    np.random.seed(seed)

    with open(log_file_path, "w") as log_f:
        log_and_print(f"=== Held-Out Evaluation ===", log_f)
        log_and_print(f"Samples: {len(X_test)}  Replicas per sample: {n_replicas}", log_f)

        # ---------------------------------------------------
        # Iterate test samples
        # ---------------------------------------------------
        for idx in range(len(X_test)):
            # support both pandas and numpy inputs
            x_row = X_test.iloc[idx] if hasattr(X_test, "iloc") else X_test[idx]
            target_value = float(y_test.iloc[idx]) if hasattr(y_test, "iloc") else float(y_test[idx])

            idx_display = getattr(X_test, "index", None)
            idx_val = idx_display[idx] if idx_display is not None else idx

            log_and_print(f"\n--- Sample {idx} (Test index {idx_val}) ---", log_f)
            log_and_print(f"Target: {target_value}", log_f)

            replica_predictions = []
            replica_candidates = []
            replica_costs = []
            replica_times = []

            # ----------------------------------------------
            # Run optimizer n_replicas times
            # ----------------------------------------------
            for r in range(n_replicas):
                t0 = time.time()

                # Create pipeline. Keep the call signature consistent with your factory.
                pipeline = PipelineFactory.create_pipeline(
                    pipeline_cfg,
                    data_x=X_test,
                    model=trained_model
                )

                result = pipeline.run(target_value)

                elapsed = time.time() - t0
                replica_times.append(elapsed)

                candidate = result.best_candidates
                # Validate candidate and predict
                try:
                    cand_arr = np.asarray(candidate, dtype=float).reshape(1, -1)
                    predicted = float(trained_model.predict(cand_arr)[0])
                except Exception as e:
                    log_and_print(f"Prediction failed for candidate (replica {r+1}): {e}", log_f)
                    predicted = float("nan")

                cost_history = result.cost_history
                if isinstance(cost_history, (float, np.float32, np.float64)):
                    cost_history = [float(cost_history)]
                else:
                    cost_history = list(map(float, cost_history))

                replica_predictions.append(predicted)
                replica_candidates.append(candidate)
                replica_costs.append(cost_history)

                log_and_print(
                    f"Replica {r+1} | Predicted: {predicted} | Final cost: {cost_history[-1] if len(cost_history)>0 else 'NA'} | Time: {elapsed:.4f}s",
                    log_f
                )

            # ----------------------------------------------
            # Aggregate statistics for this sample
            # ----------------------------------------------
            preds = np.array(replica_predictions, dtype=float)
            mean_pred = float(np.nanmean(preds))
            std_pred = float(np.nanstd(preds, ddof=1)) if np.sum(~np.isnan(preds)) > 1 else 0.0
            ci_low, ci_high = compute_ci_from_samples(preds[~np.isnan(preds)]) if np.sum(~np.isnan(preds)) > 0 else (mean_pred, mean_pred)

            avg_cost_history = pad_and_average_costs(replica_costs)
            cost_histories_all_samples.append(avg_cost_history)
            times_all_samples.append(replica_times)

            # choose best candidate by lowest final cost among replicas
            final_costs = [c[-1] if len(c) > 0 else np.nan for c in replica_costs]
            try:
                best_idx = int(np.nanargmin(final_costs))
            except Exception:
                best_idx = 0
            best_candidate = replica_candidates[best_idx]
            top_candidates = getattr(result, "top_candidates", [])[:5]

            abs_error_mean = abs(mean_pred - target_value)

            candidate_records.append({
                "test_index": int(idx_val),
                "inputs": list(x_row) if hasattr(x_row, "__iter__") else [float(x_row)],
                "target": target_value,
                "prediction_mean": mean_pred,
                "prediction_std": std_pred,
                "prediction_ci_low": ci_low,
                "prediction_ci_high": ci_high,
                "replica_predictions": replica_predictions,
                "best_candidate": best_candidate,
                "top_candidates": top_candidates,
                "replica_times": replica_times,
                "final_costs_replicas": final_costs,
                "avg_cost_history": avg_cost_history.tolist() if hasattr(avg_cost_history, "tolist") else []
            })

            all_results["targets"].append(target_value)
            all_results["predictions_mean"].append(mean_pred)
            all_results["predictions_replicas"].append(replica_predictions)
            all_results["candidates"].append(best_candidate)
            all_results["cost_histories"].append(avg_cost_history.tolist())

            log_and_print(
                f"Sample {idx} aggregated: Mean prediction: {mean_pred:.6f} | Std: {std_pred:.6f} | 95% CI: [{ci_low:.6f}, {ci_high:.6f}] | Abs error (mean pred): {abs_error_mean:.6f}",
                log_f
            )

        # ---------------------------------------------------
        # Save results (CSV)
        # ---------------------------------------------------
        df_candidates = pd.DataFrame(candidate_records)

        # Expand inputs into columns if they are consistent vectors
        try:
            inputs_arr = np.vstack(df_candidates["inputs"].values)
            for i in range(inputs_arr.shape[1]):
                df_candidates[f"input_{i}"] = inputs_arr[:, i]
            df_candidates.drop(columns=["inputs"], inplace=True)
        except Exception:
            # leave 'inputs' column intact if expansion fails
            pass

        df_candidates_path = os.path.join(run_folder, "candidates.csv")
        df_candidates.to_csv(df_candidates_path, index=False)
        log_and_print(f"Candidate DataFrame saved to {df_candidates_path}", log_f)

        df_targets = pd.DataFrame({
            "test_index": [rec["test_index"] for rec in candidate_records],
            "target":     [rec["target"] for rec in candidate_records]
        })
        df_targets_path = os.path.join(run_folder, "targets.csv")
        df_targets.to_csv(df_targets_path, index=False)
        log_and_print(f"Targets saved to {df_targets_path}", log_f)

        # Save timing summary
        max_repl = max(len(t) for t in times_all_samples) if times_all_samples else 0
        padded_times = [t + [np.nan] * (max_repl - len(t)) for t in times_all_samples]
        df_time = pd.DataFrame(padded_times, columns=[f"replica_{i+1}_sec" for i in range(max_repl)])
        df_time_path = os.path.join(run_folder, "candidate_time_summary.csv")
        df_time.to_csv(df_time_path, index=False)
        log_and_print(f"Candidate time summary saved to {df_time_path}", log_f)

        # ---------------------------------------------------
        # Final metrics and bootstrap CIs
        # ---------------------------------------------------
        y_true = np.array(all_results["targets"], dtype=float)
        y_pred = np.array(all_results["predictions_mean"], dtype=float)

        # Point metrics
        R2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
        RMSE = float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) > 0 else float("nan")
        MAE = float(mean_absolute_error(y_true, y_pred)) if len(y_true) > 0 else float("nan")

        # metric functions for bootstrap
        def metric_r2(yt, yp):
            return float(r2_score(yt, yp)) if len(yt) > 1 else float("nan")

        def metric_rmse(yt, yp):
            return float(np.sqrt(mean_squared_error(yt, yp)))

        def metric_mae(yt, yp):
            return float(mean_absolute_error(yt, yp))

        # Run bootstrap for each metric (percentile CI)
        r2_mean_b, r2_std_b, r2_ci_low_pct, r2_ci_high_pct, r2_boots = bootstrap_metric(
            y_true, y_pred, metric_r2, n_bootstrap=n_bootstrap, seed=bootstrap_seed
        )

        rmse_mean_b, rmse_std_b, rmse_ci_low_pct, rmse_ci_high_pct, rmse_boots = bootstrap_metric(
            y_true, y_pred, metric_rmse, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 1
        )

        mae_mean_b, mae_std_b, mae_ci_low_pct, mae_ci_high_pct, mae_boots = bootstrap_metric(
            y_true, y_pred, metric_mae, n_bootstrap=n_bootstrap, seed=bootstrap_seed + 2
        )

        # Format strings using bootstrap mean/std and percentile CI bounds
        r2_str = format_ci(r2_mean_b, r2_std_b, r2_ci_low_pct, r2_ci_high_pct)
        rmse_str = format_ci(rmse_mean_b, rmse_std_b, rmse_ci_low_pct, rmse_ci_high_pct)
        mae_str = format_ci(mae_mean_b, mae_std_b, mae_ci_low_pct, mae_ci_high_pct)

        log_and_print("\n=== Held-out Evaluation Summary ===", log_f)
        log_and_print(f"Samples: {len(y_true)}  Replicas per sample: {n_replicas}", log_f)
        log_and_print(f"R²:   {r2_str}", log_f)
        log_and_print(f"RMSE: {rmse_str}", log_f)
        log_and_print(f"MAE:  {mae_str}", log_f)

        # Also print explicit percentile CIs for clarity
        log_and_print("\n(Bootstrap percentile 95% CIs)", log_f)
        log_and_print(f"R² pct CI: [{r2_ci_low_pct:.4f}, {r2_ci_high_pct:.4f}]", log_f)
        log_and_print(f"RMSE pct CI: [{rmse_ci_low_pct:.4f}, {rmse_ci_high_pct:.4f}]", log_f)
        log_and_print(f"MAE pct CI: [{mae_ci_low_pct:.4f}, {mae_ci_high_pct:.4f}]", log_f)

        # Overall timing
        all_times_flat = np.concatenate([np.asarray(t) for t in times_all_samples]) if times_all_samples else np.array([])
        if all_times_flat.size > 0:
            overall_mean_time = float(np.mean(all_times_flat))
            overall_std_time = float(np.std(all_times_flat, ddof=1)) if all_times_flat.size > 1 else 0.0
            log_and_print(f"\nOverall candidate evaluation time (all replicas & samples): {overall_mean_time:.4f} ± {overall_std_time:.4f} sec", log_f)

        total_elapsed = time.time() - total_start
        log_and_print(f"Total optimizer evaluation run time: {total_elapsed:.2f} sec", log_f)

    # ---------------------------------------------------
    # Plotting (adapter + spaghetti)
    # ---------------------------------------------------
    try:
        preds_replicas_all = all_results.get("predictions_replicas", [])
        if preds_replicas_all:
            preds_mean_from_replicas = []
            for sample_preds in preds_replicas_all:
                try:
                    mean_val = float(np.nanmean(np.array(sample_preds, dtype=float)))
                except Exception:
                    mean_val = float("nan")
                preds_mean_from_replicas.append(mean_val)
        else:
            preds_mean_from_replicas = all_results.get("predictions_mean", [])

        plot_results = {
            "targets": all_results.get("targets", []),
            "predictions": preds_mean_from_replicas,
            "predictions_replicas": all_results.get("predictions_replicas", []),
            "candidates": all_results.get("candidates", []),
            "cost_histories": all_results.get("cost_histories", []),
        }

        plot_target_vs_prediction(
            plot_results,
            optimizer_name,
            float(r2_mean_b) if np.isfinite(r2_mean_b) else float("nan"),
            save_path=os.path.join(plots_folder, "target_vs_prediction.png")
        )
    except Exception as e:
        print(f"Warning: plot_target_vs_prediction failed: {e}")

    # Spaghetti plot: one line per replica across samples
    try:
        sample_indices = [rec["test_index"] for rec in candidate_records]
        plot_spaghetti_predictions(
            all_results,
            optimizer_name,
            save_path=os.path.join(plots_folder, "spaghetti_predictions.png"),
            sample_indices=sample_indices
        )
    except Exception as e:
        print(f"Warning: plot_spaghetti_predictions failed: {e}")

    try:
        plot_cost_trajectories(
            cost_histories_all_samples,
            optimizer_name,
            save_path=os.path.join(plots_folder, "cost_trajectories.png")
        )
    except Exception as e:
        print(f"Warning: plot_cost_trajectories failed: {e}")

    return {
        "run_folder": run_folder,
        "results": all_results,
        "metrics": {
            "r2_point": R2,
            "rmse_point": RMSE,
            "mae_point": MAE,
            "r2_boot_mean": r2_mean_b,
            "r2_boot_std": r2_std_b,
            "r2_boot_ci_pct": (r2_ci_low_pct, r2_ci_high_pct),
            "rmse_boot_mean": rmse_mean_b,
            "rmse_boot_std": rmse_std_b,
            "rmse_boot_ci_pct": (rmse_ci_low_pct, rmse_ci_high_pct),
            "mae_boot_mean": mae_mean_b,
            "mae_boot_std": mae_std_b,
            "mae_boot_ci_pct": (mae_ci_low_pct, mae_ci_high_pct),
            "r2_boot_samples": r2_boots,
            "rmse_boot_samples": rmse_boots,
            "mae_boot_samples": mae_boots
        }
    }
