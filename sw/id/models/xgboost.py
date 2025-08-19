# models/xgboost_model.py
from id.models.base_model import BaseModel
from xgboost import XGBRegressor


class XGBoostSurrogate(BaseModel):
    """
    XGBoost surrogate model for regression tasks.
    """

    def __init__(self, **kwargs):
        self.model = XGBRegressor(
            colsample_bytree=1.0,
            learning_rate=0.2,
            max_depth=3,
            n_estimators=50,
            subsample=0.6,
            **kwargs
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
