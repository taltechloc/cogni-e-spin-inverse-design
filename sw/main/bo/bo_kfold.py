from id.evaluator.five_fold_evaluator import run_evaluation


if __name__ == "__main__":
    run_evaluation(config_path="config.json", optimizer_name="BO")
