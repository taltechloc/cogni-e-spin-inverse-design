# models/xgboost_model.py
from id.models.base_model import BaseModel
from xgboost import XGBRegressor


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
