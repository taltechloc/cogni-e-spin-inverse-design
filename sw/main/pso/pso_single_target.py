from config import DataConfig, Global, xgb_config, pso_config

from id.data.data_loader import DataLoader
from id.data.splitter import Splitter
from id.models.model_type import ModelType
from id.objective.surrogate_objective import SurrogateObjective
from id.optimizers.optimizer_type import OptimizerType


def _create_boundaries(X):
    return [(X[col].min(), X[col].max()) for col in X.columns]


def _train_surrogate_model(X, y):
    model = ModelType.from_str('XGBoostSurrogate').create(xgb_config)
    model.train(X, y)
    return model


def main():
    # --- Load data ---
    df = DataLoader(DataConfig).get_dataframe()
    print(df.columns)
    df.drop("diameter_stdev", axis=1, inplace=True)
    splitter = Splitter(df, DataConfig)

    X, y = splitter.get_features_target()

    # Boundaries for each feature
    boundaries = _create_boundaries(X)

    # --- Train model ---
    model = _train_surrogate_model(X, y)

    # --- Define optimizer ---
    objective = SurrogateObjective(model)
    optimizer = OptimizerType.PSO.create(config=pso_config, objective=objective, boundaries=boundaries)
    target_value = float(input("Enter target fiber diameter: "))  # single target

    # --- Run PSO for single target ---
    optimizer_result = optimizer.optimize(target_value)

    print("\n=== Inverse Design Result ===")
    print(f"Target: {target_value}")
    print(f"Predicted at best input: {optimizer_result.best_prediction}")
    print(f"Best candidate input: {optimizer_result.best_candidates}")
    print(f"Final cost: {optimizer_result.cost_history[-1]}")


if __name__ == "__main__":
    main()
