# import json
# import itertools
# import numpy as np
# from copy import deepcopy
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_absolute_error
# from eSpinID.dataset import Dataset
# from eSpinID.models.model_type import ModelType
# from eSpinID.pipeline_factory import PipelineFactory
# import os
# from datetime import datetime
#
#
# def run_nested_cv_tuning(config_path, optimizer_name, param_grid,
#                          n_splits_outer=5, n_splits_inner=3, random_state=42):
#     """
#     Nested cross-validation hyperparameter tuning for eSpinID pipeline.
#     Returns a list of tuples: (outer_fold_idx, outer_test_mae, best_inner_params)
#     """
#     # Load config
#     with open(config_path, "r") as f:
#         config = json.load(f)
#
#     dataset_cfg = config["dataset"]
#     dataset = Dataset(dataset_cfg)
#     X_full, y_full = dataset.get_features_target(scaled=False)
#
#     # Ensure y is 1D
#     if hasattr(y_full, "iloc"):
#         y_full = y_full.squeeze()
#
#     # Prepare logging
#     run_folder = "nested_cv_hyperparameter_tuning"
#     os.makedirs(run_folder, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_file_path = os.path.join(run_folder, f"{optimizer_name.replace(' ', '_')}_nested_cv_{timestamp}.txt")
#
#     results_outer = []
#
#     with open(log_file_path, "w") as log_f:
#         def log(msg):
#             print(msg)
#             log_f.write(msg + "\n")
#             log_f.flush()
#
#         # Outer KFold
#         outer_kf = KFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
#
#         for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X_full), 1):
#             # Use .iloc for pandas row indexing
#             X_train_outer, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
#             y_train_outer, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
#
#             log(f"\n=== Outer Fold {fold_idx}/{n_splits_outer} ===")
#
#             # Inner KFold for hyperparameter tuning
#             inner_kf = KFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
#             param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
#
#             results_inner = []
#
#             for i, params in enumerate(param_combinations, 1):
#                 maes_inner = []
#
#                 for inner_train_idx, inner_val_idx in inner_kf.split(X_train_outer):
#                     X_train_inner, X_val_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
#                     y_train_inner, y_val_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_val_idx]
#
#                     # Initialize model
#                     model_type = config["model"]["type"]
#                     model_params = config["model"].get("params", {})
#                     model = ModelType.from_str(model_type).create(model_params)
#                     model.train(X_train_inner, y_train_inner)
#
#                     # Create pipeline with current hyperparameters
#                     pipeline_cfg = deepcopy(config["pipeline"])
#                     pipeline_cfg["optimizer"].update(params)
#                     pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train_inner, model)
#
#                     # Compute predictions on inner validation set
#                     preds_inner = []
#                     for target in y_val_inner:
#                         result = pipeline.run(float(target))
#                         predicted = model.predict(result.best_candidates.reshape(1, -1))[0]
#                         preds_inner.append(predicted)
#
#                     mae_fold = mean_absolute_error(y_val_inner, preds_inner)
#                     maes_inner.append(mae_fold)
#
#                 mean_mae_inner = np.mean(maes_inner)
#                 std_mae_inner = np.std(maes_inner, ddof=1)
#                 log(f"  {i}/{len(param_combinations)}: MAE = {mean_mae_inner:.4f} ± {std_mae_inner:.4f}, params = {params}")
#                 results_inner.append((mean_mae_inner, std_mae_inner, params))
#
#             # Select best hyperparameters from inner CV
#             results_inner.sort(key=lambda x: x[0])
#             best_mean_mae, best_std_mae, best_params = results_inner[0]
#             log(f"  >> Best params for outer fold {fold_idx}: {best_params} (MAE {best_mean_mae:.4f} ± {best_std_mae:.4f})")
#
#             # Train on full outer training set with best params
#             model_type = config["model"]["type"]
#             model_params = config["model"].get("params", {})
#             model = ModelType.from_str(model_type).create(model_params)
#             model.train(X_train_outer, y_train_outer)
#
#             pipeline_cfg = deepcopy(config["pipeline"])
#             pipeline_cfg["optimizer"].update(best_params)
#             pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train_outer, model)
#
#             # Evaluate on outer test set
#             preds_test = []
#             for target in y_test:
#                 result = pipeline.run(float(target))
#                 predicted = model.predict(result.best_candidates.reshape(1, -1))[0]
#                 preds_test.append(predicted)
#
#             mae_test = mean_absolute_error(y_test, preds_test)
#             results_outer.append((fold_idx, mae_test, best_params))
#             log(f"Outer fold {fold_idx} MAE on test: {mae_test:.4f}")
#
#         # Summary
#         all_maes = [r[1] for r in results_outer]
#         mean_mae = np.mean(all_maes)
#         std_mae = np.std(all_maes, ddof=1)
#         log(f"\n=== Nested CV Summary ===")
#         log(f"Mean MAE across outer folds: {mean_mae:.4f} ± {std_mae:.4f}")
#         log(f"Best hyperparameters per fold: {[r[2] for r in results_outer]}")
#
#     print(f"\nAll logs saved in: {log_file_path}")
#     return results_outer

import json
import itertools
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from eSpinID.dataset import Dataset
from eSpinID.models.model_type import ModelType
from eSpinID.pipeline_factory import PipelineFactory
import os
from datetime import datetime

def run_simple_cv_tuning(config_path, optimizer_name, param_grid,
                         n_splits=5, random_state=42):
    """
    Simple cross-validation hyperparameter tuning for eSpinID pipeline.
    Returns the best hyperparameters found on the development set.
    """
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    dataset = Dataset(dataset_cfg)
    X_dev, y_dev = dataset.get_features_target(scaled=False)

    # Ensure y is 1D
    if hasattr(y_dev, "iloc"):
        y_dev = y_dev.squeeze()

    # Prepare logging
    run_folder = "cv_hyperparameter_tuning"
    os.makedirs(run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(run_folder, f"{optimizer_name.replace(' ', '_')}_cv_{timestamp}.txt")

    with open(log_file_path, "w") as log_f:
        def log(msg):
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        log(f"Starting simple CV tuning for {optimizer_name} on development set")

        # Create all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

        # KFold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        results = []

        for i, params in enumerate(param_combinations, 1):
            maes = []

            for train_idx, val_idx in kf.split(X_dev):
                X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
                y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

                # Initialize model
                model_type = config["model"]["type"]
                model_params = config["model"].get("params", {})
                model = ModelType.from_str(model_type).create(model_params)
                model.train(X_train, y_train)

                # Create pipeline with current hyperparameters
                pipeline_cfg = deepcopy(config["pipeline"])
                pipeline_cfg["optimizer"].update(params)
                pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train, model)

                # Predict on validation set
                preds = []
                for target in y_val:
                    result = pipeline.run(float(target))
                    predicted = model.predict(result.best_candidates.reshape(1, -1))[0]
                    preds.append(predicted)

                mae_fold = mean_absolute_error(y_val, preds)
                maes.append(mae_fold)

            mean_mae = np.mean(maes)
            std_mae = np.std(maes, ddof=1)
            log(f"{i}/{len(param_combinations)}: MAE = {mean_mae:.4f} ± {std_mae:.4f}, params = {params}")
            results.append((mean_mae, std_mae, params))

        # Select best hyperparameters
        results.sort(key=lambda x: x[0])
        best_mean_mae, best_std_mae, best_params = results[0]
        log(f"\nBest hyperparameters found: {best_params}")
        log(f"Mean MAE (CV on dev set): {best_mean_mae:.4f} ± {best_std_mae:.4f}")

    print(f"\nAll logs saved in: {log_file_path}")
    return best_params, best_mean_mae, best_std_mae
