import joblib

from main.evaluator.evaluator import run_evaluation


if __name__ == "__main__":
    surrogate_model = joblib.load("../saved_models/XGBoost_model.pkl")

    run_evaluation(surrogate_model, config_path="eval_config.json", optimizer_name="Bayesian Optimizer")
