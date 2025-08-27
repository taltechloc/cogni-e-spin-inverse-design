# models/xgboost_model.py
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

    def __init__(self, config):
        self.model = XGBRegressor(
            n_estimators=config.n_estimators,
            colsample_bytree=config.colsample_bytree,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            subsample=config.subsample,
            random_state=getattr(config, "random_state", 42)
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
