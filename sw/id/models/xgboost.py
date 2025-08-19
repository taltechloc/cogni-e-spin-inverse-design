# models/xgboost_model.py
from id.models.base_model import BaseModel
from xgboost import XGBRegressor


class XGBoostSurrogate(BaseModel):
    """
    XGBoost surrogate model for regression tasks.
    """

    def __init__(self, **kwargs):
        # sensible defaults, can be overridden
        self.model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **kwargs
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
