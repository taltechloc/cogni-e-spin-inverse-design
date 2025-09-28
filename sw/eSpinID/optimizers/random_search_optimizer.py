# optimizers/random_search_optimizer.py
import numpy as np

from eSpinID.optimizers.base_optimizer import BaseOptimizer
from eSpinID.optimizers.optimization_result import OptimizationResult


RSConfig = {
    "n_iter": None,          # int
    "optimizer_type": "RandomSearch"
}


class RandomSearchOptimizer(BaseOptimizer):
    """
    Random Search Optimizer.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.n_iter = definition.get("n_iter", 20)

        # Derived values
        self.lower_bounds = np.array([b[0] for b in self.boundaries])
        self.upper_bounds = np.array([b[1] for b in self.boundaries])
        self.dim = len(boundaries)

    def optimize(self, target):
        best_x = None
        best_cost = float('inf')
        cost_history = []

        top_candidates = []

        for _ in range(self.n_iter):
            x = np.array([np.random.uniform(self.lower_bounds[i], self.upper_bounds[i])
                          for i in range(self.dim)])
            cost = self.objective.evaluate(x, target)
            cost_history.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_x = x.copy()

            # Maintain top 5 candidates
            top_candidates.append((cost, x.copy()))
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

        top_positions = [p for c, p in top_candidates]
        predicted = self.objective.model.predict(best_x.reshape(1, -1)).item()

        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Iteration",
                ylabel="Cost",
                title="Random Search Convergence",
                label="Cost per iteration"
            )
        }
        return OptimizationResult(
            best_candidates=best_x,
            best_prediction=predicted,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )
