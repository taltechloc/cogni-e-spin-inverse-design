# optimizers/grid_search.py
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

GridSearchConfig = {
    "steps_per_param": None,
    "optimizer_type": "GridSearch"
}


class GridSearchOptimizer(BaseOptimizer):
    """
    Grid Search Optimizer for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.steps_per_param = definition.get("steps_per_param", 5)
        self.dim = len(boundaries)

        self.lower_bounds = np.array([b[0] for b in boundaries])
        self.upper_bounds = np.array([b[1] for b in boundaries])

    def optimize(self, target):
        # Create grid axes for each parameter
        grid_axes = [np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.steps_per_param)
                     for i in range(self.dim)]
        grid_points = list(product(*grid_axes))

        cost_history = []
        best_cost = float('inf')
        best_input = None
        top_candidates = []

        for point in grid_points:
            x = np.array(point).reshape(1, -1)
            pred = self.objective.model.predict(x)[0]
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
        plots_data = self._generate_all_plots(cost_history)

        return OptimizationResult(
            best_candidates=best_input,
            best_prediction=best_pred,
            cost_history=cost_history,
            top_candidates=top_positions,
            n_iterations=len(cost_history),
            plots_data=plots_data
        )

    # ----------------------
    # Plot generation
    # ----------------------
    def _generate_all_plots(self, cost_history):
        plots = {}
        plots["cost_history"] = self._plot_cost_history(cost_history)
        return plots

    def _plot_cost_history(self, cost_history):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cost_history, label="Cost per grid point")
        ax.set_xlabel("Grid Point Index")
        ax.set_ylabel("Squared Error")
        ax.set_title("Grid Search Convergence")
        ax.legend()
        plt.close(fig)
        return fig
