from id.data.data_loader import DataLoader
from id.data.splitter import Splitter
from id.models.xgboost import XGBoostSurrogate
from id.objective.surrogate_objective import SurrogateObjective
from id.optimizers.pso import PSOOptimizer


# =====================
# MAIN
# =====================

def main():
    # --- Load data ---
    data_config = DataConfig()
    loader = DataLoader(data_config)
    df = loader.get_dataframe()

    splitter = Splitter(df, data_config)
    X, y = splitter.get_features_target()

    # Boundaries for each feature
    boundaries = [(X[col].min(), X[col].max()) for col in X.columns]

    # --- Train model ---
    model_config = ModelConfig()
    model = XGBoostSurrogate(model_config)
    model.train(X, y)  # train on full data or specific train split

    # --- Define optimizer ---
    optimizer_config = OptimizerConfig()
    target_value = float(input("Enter target fiber diameter: "))  # single target
    objective = SurrogateObjective(model)
    optimizer = PSOOptimizer(config=optimizer_config, objective=objective, boundaries=boundaries)

    # --- Run PSO for single target ---
    best_input, predicted, cost_history = optimizer.optimize(target_value)

    print("\n=== Inverse Design Result ===")
    print(f"Target: {target_value}")
    print(f"Predicted at best input: {predicted}")
    print(f"Best candidate input: {best_input}")
    print(f"Final cost: {cost_history[-1]}")

if __name__ == "__main__":
    main()
