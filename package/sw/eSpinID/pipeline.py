# pipeline.py


class Pipeline:
    def __init__(self, optimizer, objective_function, boundaries):
        self.optimizer = optimizer
        self.objective_function = objective_function
        self.boundaries = boundaries
        self.result = None

    def run(self, target):
        """Run the optimizer for a given target value."""
        self.result = self.optimizer.optimize(target)
        return self.result

    def __str__(self):
        if self.result is None:
            return "Pipeline result: Not yet run."

        top_candidates_str = "\n    ".join(str(c) for c in getattr(self.result, "top_candidates", []))
        return (
            f"Pipeline result:\n"
            f"  Best Candidates: {self.result.best_candidates}\n"
            f"  Best Prediction: {self.result.best_prediction}\n"
            f"  Iterations: {self.result.n_iterations}\n"
            f"  Top 5 Candidates:\n    {top_candidates_str if top_candidates_str else 'N/A'}"
        )

    @property
    def top_candidates(self):
        """Return the top 5 candidates from the last run."""
        if self.result is None:
            return []
        return getattr(self.result, "top_candidates", [])
