# objective/surrogate_objective.py
import numpy as np
from .base_objective import BaseObjective


class SurrogateObjective(BaseObjective):
    """
    Wraps a surrogate model as an optimization objective.
    """

    def __init__(self, model):
        """
        Parameters:
            model: Trained surrogate model (e.g., XGBoostSurrogate)
        """
        self.model = model

    def evaluate(self, X, target):
        """
        Evaluate the objective function.
        (Predicted - target)²
        """
        X = np.array(X).reshape(1, -1)
        predicted = self.model.predict(X)[0]
        return (predicted - target) ** 2
