# models/base_model.py
from abc import ABC, abstractmethod

from eSpinID.utils.metrics import regression_metrics


class BaseModel(ABC):
    """
    Abstract base class for surrogate models.
    """

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def evaluate(self, X, y):
        """
        Evaluate predictions with RMSE, MAE, R².
        Returns a dict of metrics.
        """
        y_pred = self.predict(X)
        return regression_metrics(y, y_pred)
