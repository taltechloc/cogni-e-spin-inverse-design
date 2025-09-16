import numpy as np
from sklearn.preprocessing import StandardScaler


class MahalanobisDistance:
    def __init__(self, reg: float = 1e-6):
        self.reg = reg
        self.scaler = StandardScaler()
        self.mu = None
        self.cov_inv = None

    def fit(self, X: np.ndarray):
        """
        Fit the Mahalanobis distance model on training data.
        Computes mean vector and regularized inverse covariance matrix.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.mu = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled, rowvar=False)
        cov_reg = cov + self.reg * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.pinv(cov_reg)

    def find_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance(s) of given point(s) from the training distribution.
        Returns a numpy array of distances.
        """
        X_scaled = self.scaler.transform(np.atleast_2d(X))
        diff = X_scaled - self.mu
        left = diff.dot(self.cov_inv)
        squared = np.sum(left * diff, axis=1)
        return np.sqrt(np.maximum(squared, 0.0))
