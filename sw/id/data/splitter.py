# data/splitter.py
from sklearn.model_selection import train_test_split, KFold


class Splitter:
    """
    Responsible for splitting DataFrame into features (X) and target (y),
    and applying train/test or k-fold splits.
    """

    def __init__(self, df, target_column):
        if target_column not in df.columns:
            raise ValueError(f"{target_column} not found in DataFrame.")
        self.df = df
        self.target_column = target_column

    def get_features_target(self):
        """Return X and y."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y

    def train_test_split(self, test_size=0.2, random_state=None):
        """Return train/test splits for X and y."""
        X, y = self.get_features_target()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def kfold_split(self, n_splits=5, shuffle=True, random_state=None):
        """
        Generator for K-Fold splits.
        Yields (X_train, X_test, y_train, y_test) for each fold.
        """
        X, y = self.get_features_target()
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            yield X_train, X_test, y_train, y_test
