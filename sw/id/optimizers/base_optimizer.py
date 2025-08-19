# optimizers/base_optimizer.py
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    """

    def __init__(self, objective, bounds, n_iter=50):
        """
        Parameters:
            objective: An instance of BaseObjective
            bounds (list of tuples): [(min1, max1), (min2, max2), ...]
            n_iter (int): Number of iterations
        """
        self.objective = objective
        self.bounds = bounds
        self.n_iter = n_iter

    @abstractmethod
    def optimize(self, target):
        """
        Run optimization to minimize objective for given target.
        Returns best_solution, best_score.
        """
        pass
