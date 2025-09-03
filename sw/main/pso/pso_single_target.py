from config import DataConfig, xgb_config, pso_config
import config
from id.data.data_loader import DataLoader
from id.data.splitter import Splitter
from id.models.model_type import ModelType
from id.pipeline import Pipeline


def _train_surrogate_model(X, y):
    model = ModelType.from_str('XGBoostSurrogate').create(xgb_config)
    model.train(X, y)
    return model


def main():
    # --- Load and preprocess data ---
    df = DataLoader(DataConfig).get_dataframe()
    df.drop("diameter_stdev", axis=1, inplace=True)
    splitter = Splitter(df, DataConfig)
    X, y = splitter.get_features_target()

    # --- Train surrogate model ---
    model = _train_surrogate_model(X, y)

    # --- Create pipeline and run optimization ---
    pipeline = Pipeline(optimizer_config=pso_config, data_x=X, model=model)
    target_value = float(input("Enter target fiber diameter: "))
    pipeline.run(target_value)

    print(pipeline)


if __name__ == "__main__":
    main()
