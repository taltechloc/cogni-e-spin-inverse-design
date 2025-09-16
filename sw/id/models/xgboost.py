# models/xgboost.py
from xgboost import XGBRegressor
from dataclasses import dataclass

from id.models.base_model import BaseModel


@dataclass
class XGBoostConfig:
    """
    Configuration template for XGBoostSurrogate model.
    Actual values assigned in config.py
    """
    n_estimators: int
    colsample_bytree: float
    learning_rate: float
    max_depth: int
    subsample: float
    random_state: int = 42  # default value


class XGBoostSurrogate(BaseModel):
    """
    XGBoost surrogate model for regression tasks.
    """

    def __init__(self, definition):
        self.model = XGBRegressor(
            n_estimators=definition.get("n_estimators"),
            colsample_bytree=definition.get("colsample_bytree"),
            learning_rate=definition.get("learning_rate"),
            max_depth=definition.get("max_depth"),
            subsample=definition.get("subsample"),
            random_state=definition.get("random_state")
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
