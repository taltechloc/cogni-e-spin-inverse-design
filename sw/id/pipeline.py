class Pipeline:
    def __init__(self, optimizer, objective_function, boundaries):
        self.optimizer = optimizer
        self.objective_function = objective_function
        self.boundaries = boundaries
        self.result = None

    def run(self, target):
        self.result = self.optimizer.optimize(target)
        return self.result

    def __str__(self):
        if self.result is None:
            return "Pipeline result: Not yet run."
        return (
            f"Pipeline result:\n"
            f"  Best Candidates: {self.result.best_candidates}\n"
            f"  Best Prediction: {self.result.best_prediction}\n"
            f"  Iterations: {self.result.n_iterations}"
        )
