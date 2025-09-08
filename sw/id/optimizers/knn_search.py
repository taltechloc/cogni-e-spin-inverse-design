# optimizers/knn_search.py
import numpy as np
from matplotlib import pyplot as plt

from id.optimizers.optimization_result import OptimizationResult
from id.optimizers.base_optimizer import BaseOptimizer

KNNSearchConfig = {
    "k": None,                 # int, number of neighbors
    "optimizer_type": "KNNSearch"
}

class KNNSearch(BaseOptimizer):
    """
    KNN-based search for inverse design.
    """

    def __init__(self, definition, objective, boundaries):
        super().__init__(objective, boundaries)
        self.k = definition.get("k", 5)
        self.dim = len(boundaries)

        # Expect training data to be in objective
        if not hasattr(objective, "X_train") or not hasattr(objective, "y_train"):
            raise ValueError("objective must contain X_train and y_train for KNNSearch")
        self.X_train = np.array(objective.X_train)
        self.y_train = np.array(objective.y_train)

    def optimize(self, target):
        # Find k nearest neighbors in output space
        idx_sorted = np.argsort(np.abs(self.y_train - target))
        neighbors = self.X_train[idx_sorted[:self.k]]
        preds = self.objective.model.predict(neighbors)
        errors = (preds - target) ** 2

        best_idx = np.argmin(errors)
        best_input = neighbors[best_idx]
        best_pred = preds[best_idx]

        cost_history = list(errors)
        top_candidates = [neighbors[i] for i in np.argsort(errors)[:5]]

        plots_data = self._generate_all_plots(cost_history)

        return OptimizationResult(
            best_candidates=best_input,
            best_prediction=best_pred,
            cost_history=cost_history,
            top_candidates=top_candidates,
            n_iterations=1,
            plots_data=plots_data
        )

    # ----------------------
    # Plot generation methods
    # ----------------------
    def _generate_all_plots(self, cost_history):
        plots = {}
        plots["cost_history"] = self._plot_cost_history(cost_history)
        return plots

    def _plot_cost_history(self, cost_history):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cost_history, marker='o', linestyle='-', label="Squared Error")
        ax.set_xlabel("Neighbor Index (sorted by closeness)")
        ax.set_ylabel("Squared Error")
        ax.set_title("KNN Search Errors")
        ax.legend()
        plt.close(fig)
        return fig
