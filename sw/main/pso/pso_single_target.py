import json
import os
import datetime

from eSpinID.dataset import Dataset
from eSpinID.models.model_type import ModelType
from eSpinID.pipeline_factory import PipelineFactory
from storage.out_of_distribution_detection.mahalanobis_distance import MahalanobisDistance


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


    X, y = dataset.get_features_target(scaled=False)
    model = _train_surrogate_model(X, y, config["model"])

    maha_distance = MahalanobisDistance()
    maha_distance.fit(X)

    pipeline = PipelineFactory.create_pipeline(config["pipeline"], X, model)

    target_value = float(input("Enter target fiber diameter: "))
    result = pipeline.run(target_value)

    print(pipeline)
    is_novel = maha_distance.find_distance(result.best_candidates.reshape(1, -1))
    print(f"Is the point novel? {is_novel[0]}")

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join("plots", timestamp)
    os.makedirs(run_folder, exist_ok=True)

    for name, fig in result.plots_data.items():
        file_path = os.path.join(run_folder, f"{name}.png")
        fig.savefig(file_path)
        print(f"Saved plot: {file_path}")

if __name__ == "__main__":
    main()
