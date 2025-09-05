import json
from id.dataset import Dataset
from id.models.model_type import ModelType
from id.pipeline_factory import PipelineFactory


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

    if "diameter_stdev" in dataset.df.columns:
        dataset.df.drop("diameter_stdev", axis=1, inplace=True)

    X, y = dataset.get_features_target(scaled=False)

    model = _train_surrogate_model(X, y, config["model"])

    pipeline = PipelineFactory.create_pipeline(config["pipeline"], X, model)

    target_value = float(input("Enter target fiber diameter: "))
    pipeline.run(target_value)

    print(pipeline)


if __name__ == "__main__":
    main()
