# data/splitter.py
from sklearn.model_selection import train_test_split, KFold


class Splitter:
    """
    Responsible for splitting DataFrame into features (X) and target (y),
    and applying train/test or k-fold splits.
    """

    def __init__(self, df, config):
        if config.target_column not in df.columns:
            raise ValueError(f"{config.arget_column} not found in DataFrame.")
        self.df = df
        self.target_column = config.target_column

    def get_features_target(self):
        """Return X and y."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return X, y

    def train_test_split(self, test_size=0.2, random_state=None):
        """Return train/test splits for X and y."""
        X, y = self.get_features_target()
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

