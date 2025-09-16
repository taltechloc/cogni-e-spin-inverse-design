import json
import itertools
import numpy as np
from copy import deepcopy
from sklearn.model_selection import KFold
from id.dataset import Dataset
from id.models.model_type import ModelType
from id.pipeline_factory import PipelineFactory
import os
from datetime import datetime

def run_hyperparameter_tuning(config_path, optimizer_name, param_grid, n_splits=5):
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

        keys, values = zip(*param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        for i, params in enumerate(param_combinations, 1):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=config["global"]["seed"])
            fold_maes = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full), start=1):
                X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
                y_train, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

                model_type = config["model"]["type"]
                model_params = config["model"].get("params", {})
                model = ModelType.from_str(model_type).create(model_params)
                model.train(X_train, y_train)

                pipeline_cfg = deepcopy(config["pipeline"])
                pipeline_cfg["optimizer"].update(params)
                pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train, model)

                errors = []
                for target in y_val:
                    result = pipeline.run(float(target))
                    predicted = model.predict(result.best_candidates.reshape(1, -1))[0]
                    errors.append(np.abs(predicted - target))

                fold_maes.append(np.mean(errors))

            mean_mae = np.mean(fold_maes)
            std_mae = np.std(fold_maes)
            log(f"{i}/{len(param_combinations)}: MAE = {mean_mae:.4f} ± {std_mae:.4f}, params = {params}")
            results.append((mean_mae, std_mae, params))

        results.sort(key=lambda x: x[0])
        best_mean_mae, best_std_mae, best_params = results[0]

        log("\n=== Best Hyperparameters ===")
        log(str(best_params))
        log(f"Mean MAE across folds: {best_mean_mae:.4f} ± {best_std_mae:.4f}")

    print(f"\nAll logs saved in: {log_file_path}")
