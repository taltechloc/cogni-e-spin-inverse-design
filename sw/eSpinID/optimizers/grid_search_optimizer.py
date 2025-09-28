# optimizers/grid_search_optimizer.py
import numpy as np
from itertools import product

from eSpinID.optimizers.base_optimizer import BaseOptimizer
from eSpinID.optimizers.optimization_result import OptimizationResult


GridSearchConfig = {
    "optimizer_type": "GridSearch",
    "param_levels": [
        [8, 9, 10, 12],            # Solution Concentration
        [10, 12.5, 15, 20],        # Tip–Collector Distance
        [15, 22.5, 20, 25],        # Applied Voltage
        [0.2, 0.25, 0.3, 0.4]      # Feed Rate
    ]
}


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search Optimizer for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.param_levels = definition.get("param_levels", None)

        self.dim = len(boundaries)
        if self.param_levels is not None:
            assert len(self.param_levels) == self.dim, "Levels must match number of parameters"

        else:
            # fallback to linspace if levels not provided
            self.steps_per_param = definition.get("steps_per_param", 5)
            self.lower_bounds = np.array([b[0] for b in boundaries])
            self.upper_bounds = np.array([b[1] for b in boundaries])

    def optimize(self, target):
        # Create grid axes for each parameter
        if self.param_levels is not None:
            grid_axes = self.param_levels
        else:
            grid_axes = [np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.steps_per_param)
                         for i in range(self.dim)]

        grid_points = list(product(*grid_axes))

        cost_history = []
        best_cost = float('inf')
        best_input = None
        top_candidates = []

        for point in grid_points:
            x = np.array(point).reshape(1, -1)
            pred = self.objective.model.predict(x).item()
            cost = (pred - target) ** 2
            cost_history.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_input = np.array(point)
                best_pred = pred

            # Track top 5 candidates
            top_candidates.append((cost, np.array(point)))
            top_candidates = sorted(top_candidates, key=lambda t: t[0])[:5]

        top_positions = [p for c, p in top_candidates]
        plots_data = {
            "cost_history": self.plot_cost_history(
                cost_history,
                xlabel="Grid Point Index",
                ylabel="Squared Error",
                title="Grid Search Convergence",
                label="Cost per Grid Point"
            )
        }
        return OptimizationResult(
            best_candidates=best_input,
            best_prediction=best_pred,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )