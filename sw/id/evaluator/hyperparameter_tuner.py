import json
import itertools
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from id.dataset import Dataset
from id.models.model_type import ModelType
from id.pipeline_factory import PipelineFactory
import os
from datetime import datetime

def run_hyperparameter_tuning(config_path, optimizer_name, param_grid):
    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    dataset = Dataset(dataset_cfg)
    X, y = dataset.get_features_target(scaled=False)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["global"]["seed"]
    )

    # Train surrogate model
    model_type = config["model"]["type"]
    model_params = config["model"].get("params", {})
    model = ModelType.from_str(model_type).create(model_params)
    model.train(X_train, y_train)

    # Generate all hyperparameter combinations
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Create folder for hyperparameter tuning logs
    run_folder = "hyperparameter_tuning"
    os.makedirs(run_folder, exist_ok=True)

    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(run_folder, f"{optimizer_name.replace(' ', '_')}_tuning_{timestamp}.txt")

    results = []

    with open(log_file_path, "w") as log_f:

        def log(msg):
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        # Loop over hyperparameter combinations
        for i, params in enumerate(param_combinations, 1):
            # Prepare pipeline
            pipeline_cfg = deepcopy(config["pipeline"])
            pipeline_cfg["optimizer"].update(params)
            pipeline = PipelineFactory.create_pipeline(pipeline_cfg, X_train, model)

            # Evaluate on test set
            errors = []
            for target in y_test:
                result = pipeline.run(float(target))
                predicted = model.predict(result.best_candidates.reshape(1, -1))[0]
                errors.append(np.abs(predicted - target))

            errors = np.array(errors)
            mean_mae = np.mean(errors)
            std_mae = np.std(errors)

            log(f"{i}/{len(param_combinations)}: MAE = {mean_mae:.4f} ± {std_mae:.4f}, params = {params}")
            results.append((mean_mae, std_mae, params))

        # Select best parameters
        results.sort(key=lambda x: x[0])
        best_mean_mae, best_std_mae, best_params = results[0]

        log("\n=== Best Hyperparameters ===")
        log(str(best_params))
        log(f"Mean MAE on test set targets: {best_mean_mae:.4f} ± {best_std_mae:.4f}")

    print(f"\nAll logs saved in: {log_file_path}")
