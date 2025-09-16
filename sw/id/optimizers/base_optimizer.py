# optimizers/base_optimizer.py
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

from id.optimizers.optimization_result import OptimizationResult


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """

    def __init__(self, objective, boundaries):
        """
        Parameters:
            objective: An instance of BaseObjective
            boundaries (list of tuples): [(min1, max1), (min2, max2), ...]
        """
        self.objective = objective
        self.boundaries = boundaries

    @abstractmethod
    def optimize(self, target)  -> OptimizationResult:
        """
        Run optimization to minimize objective for given target.
        Returns best_solution, best_score.
        """
        pass

    @staticmethod
    def plot_cost_history(cost_history,
                          xlabel="Iteration",
                          ylabel="Cost",
                          title="Convergence",
                          label="Best Cost"):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cost_history, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        plt.close(fig)
        return fig
