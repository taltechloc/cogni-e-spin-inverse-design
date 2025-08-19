# data/preprocessing.py
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Handles preprocessing transformations for features.
    """
    def __init__(self, scaler=None):
        self.scaler = scaler or StandardScaler()

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def transform(self, X):
        return self.scaler.transform(X)
