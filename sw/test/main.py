from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data.data_loader import DataLoader
from data.splitter import Splitter
from id.models.xgboost import XGBoostSurrogate
from objective.surrogate_objective import SurrogateObjective
from optimizers.pso import PSOOptimizer
from id.inverse_design_pipeline import InverseDesignPipeline

# =====================
# WRAPPERS
# =====================

def pso_inverse_design(model, target, boundaries, optimizer_config: OptimizerConfig):
    objective = SurrogateObjective(model)
    optimizer = PSOOptimizer(config=optimizer_config, objective=objective, boundaries=boundaries)
    best_input, pred, cost_hist = optimizer.optimize(target)
    return best_input, pred, cost_hist


# =====================
# EVALUATION HELPERS
# =====================

def evaluate_with_perturbation(results, model, eval_config):
    noise_std = np.sqrt(eval_config.noise_var)
    all_targets = results["targets"]
    all_preds = results["predictions"]
    all_candidates = results["candidates"]

    all_perturbed_preds = []
    for best_input in all_candidates:
        perturbations = np.random.normal(
            0, noise_std, size=(eval_config.n_perturb, best_input.shape[0])
        )
        perturbed_inputs = best_input + perturbations
        perturbed_preds = model.predict(perturbed_inputs)
        perturbed_pred_avg = np.mean(perturbed_preds)
        all_perturbed_preds.append(perturbed_pred_avg)

    # Metrics
    metrics = {
        "rmse": mean_squared_error(all_targets, all_preds),
        "mae": mean_absolute_error(all_targets, all_preds),
        "r2": r2_score(all_targets, all_preds),
        "rmse_pert": mean_squared_error(all_targets, all_perturbed_preds),
        "mae_pert": mean_absolute_error(all_targets, all_perturbed_preds),
        "r2_pert": r2_score(all_targets, all_perturbed_preds),
    }

    return {
        "perturbed_predictions": all_perturbed_preds,
        "metrics": metrics,
    }


def plot_results(results, eval_results, method_name, n_perturb):
    all_targets = results["targets"]
    all_preds = results["predictions"]
    all_perturbed_preds = eval_results["perturbed_predictions"]

    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.scatter(all_targets, all_preds, alpha=0.7, edgecolors="k")
    plt.plot([min(all_targets), max(all_targets)],
             [min(all_targets), max(all_targets)], "r--")
    plt.xlabel("Actual Fiber Diameter (nm)")
    plt.ylabel("Predicted Fiber Diameter (nm)")
    plt.title(f"{method_name} - Original Predictions")

    # Perturbed
    plt.subplot(1, 2, 2)
    plt.scatter(
        all_targets, all_perturbed_preds,
        alpha=0.7, edgecolors="k", color="orange"
    )
    plt.plot([min(all_targets), max(all_targets)],
             [min(all_targets), max(all_targets)], "r--")
    plt.xlabel("Actual Fiber Diameter (nm)")
    plt.ylabel("Perturbed Predicted Fiber Diameter (nm)")
    plt.title(f"{method_name} - Perturbed Predictions (avg over {n_perturb})")

    plt.tight_layout()
    plt.show()

# =====================
# MAIN
# =====================

def main():
    # --- Load dataset ---
    data_config = DataConfig()
    loader = DataLoader(data_config)
    df = loader.get_dataframe()

    splitter = Splitter(df, data_config)
    X, y = splitter.get_features_target()

    # Boundaries for each feature
    boundaries = [(X[col].min(), X[col].max()) for col in X.columns]

    # --- Configs ---
    model_config = ModelConfig()
    optimizer_config = OptimizerConfig()
    eval_config = EvalConfig()
    pipeline = InverseDesignPipeline(eval_config)

    # --- KFold Loop ---
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    all_metrics = []
    all_targets_all_folds = []
    all_preds_all_folds = []
    all_perturbed_preds_all_folds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n=== Fold {fold} ===")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBoostSurrogate(model_config)

        results = pipeline.run(
            method_func=lambda m, target, bounds: pso_inverse_design(m, target, bounds, optimizer_config),
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            boundaries=boundaries,
        )

        eval_results = evaluate_with_perturbation(results, model, eval_config)

        print(results["targets"])
        print(results["predictions"])
        # Accumulate results across folds
        all_targets_all_folds.extend(results["targets"])
        all_preds_all_folds.extend(results["predictions"])
        all_perturbed_preds_all_folds.extend(eval_results["perturbed_predictions"])
        all_metrics.append(eval_results["metrics"])

    # --- Plot all folds together ---
    plt.figure(figsize=(12, 5))

    # Original
    plt.subplot(1, 2, 1)
    plt.scatter(all_targets_all_folds, all_preds_all_folds, alpha=0.7, edgecolors="k")
    plt.plot([min(all_targets_all_folds), max(all_targets_all_folds)],
             [min(all_targets_all_folds), max(all_targets_all_folds)], "r--")
    plt.xlabel("Actual Fiber Diameter (nm)")
    plt.ylabel("Predicted Fiber Diameter (nm)")
    plt.title(f"{eval_config.method_name} - Original Predictions (All Folds)")

    # Perturbed
    plt.subplot(1, 2, 2)
    plt.scatter(all_targets_all_folds, all_perturbed_preds_all_folds,
                alpha=0.7, edgecolors="k", color="orange")
    plt.plot([min(all_targets_all_folds), max(all_targets_all_folds)],
             [min(all_targets_all_folds), max(all_targets_all_folds)], "r--")
    plt.xlabel("Actual Fiber Diameter (nm)")
    plt.ylabel("Perturbed Predicted Fiber Diameter (nm)")
    plt.title(f"{eval_config.method_name} - Perturbed Predictions (avg over {eval_config.n_perturb}, All Folds)")

    plt.tight_layout()
    plt.show()

    # --- Final Summary ---
    print("\n=== Final Cross-Validation Summary ===")
    for i, metrics in enumerate(all_metrics, 1):
        print(f"Fold {i}: {metrics}")
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
    print(f"\nAverage Metrics: {avg_metrics}")


if __name__ == "__main__":
    main()
