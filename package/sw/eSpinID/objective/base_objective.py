# objective/base_objective.py
from abc import ABC, abstractmethod


class BaseObjective(ABC):
    """
    Abstract base class for optimization objectives.
    """

    @abstractmethod
    def evaluate(self, X, target):
        """
        Evaluate the objective at point X against a target.
        Should return a scalar loss/error value.
        """
        pass
