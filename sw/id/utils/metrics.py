# utils/metrics.py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def regression_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, and R².
    Returns a dict with results.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "R2": r2}
