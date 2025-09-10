import json
import os
import datetime
import numpy as np

from id.dataset import Dataset
from id.models.model_type import ModelType
from id.pipeline_factory import PipelineFactory
from id.out_of_distribution_detection.mahalanobis_distance import MahalanobisDistance


def _train_surrogate_model(X, y, model_def):
    model_type = model_def["type"]
    model_params = model_def.get("params", {})
    model = ModelType.from_str(model_type).create(model_params)
    model.train(X, y)
    return model


def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    dataset_cfg = config["dataset"]
    dataset = Dataset(dataset_cfg)

    # Features and target
    X, y = dataset.get_features_target(scaled=False)

    # Train surrogate model
    model = _train_surrogate_model(X, y, config["model"])

    # Fit Mahalanobis detector
    maha_detector = MahalanobisDistance()
    maha_detector.fit(X)

    # Create pipeline
    pipeline = PipelineFactory.create_pipeline(config["pipeline"], X, model)

    # Get target value
    target_value = float(input("Enter target fiber diameter: "))

    # Run pipeline multiple times and check novelty
    n_runs = 100
    novel_flags = []

    for i in range(n_runs):
        result = pipeline.run(target_value)
        # Ensure candidate is 2D
        candidate = np.atleast_2d(result.best_candidates)
        distance = maha_detector.find_distance(candidate)
        print("Distance", distance)

    # Optional: save plots for the last run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join("plots", timestamp)
    os.makedirs(run_folder, exist_ok=True)

    for name, fig in result.plots_data.items():
        file_path = os.path.join(run_folder, f"{name}.png")
        fig.savefig(file_path)
        print(f"Saved plot: {file_path}")


if __name__ == "__main__":
    main()
