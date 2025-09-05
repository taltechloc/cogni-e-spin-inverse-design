# objective/surrogate_objective.py
import numpy as np
from id.objective.base_objective import BaseObjective


class SurrogateObjective(BaseObjective):
    """
    Wraps a surrogate model as an optimization objective.
    """

    def __init__(self, model, **kwargs):
        """
        Parameters
        ----------
        model : object
            Trained surrogate model (e.g., XGBoostSurrogate)
        kwargs : dict
            Extra parameters for customization (e.g., regularization, weighting).
        """
        self.model = model
        self.params = kwargs

    def evaluate(self, X, target):
        """
        Evaluate the objective function.
        Default: (Predicted - target)²

        Parameters
        ----------
        X : array-like
            Candidate input.
        target : float
            Desired target value.

        Returns
        -------
        float
            Loss value.
        """
        X = np.array(X).reshape(1, -1)
        predicted = self.model.predict(X)[0]

        # basic squared error
        loss = (predicted - target) ** 2

        return loss
