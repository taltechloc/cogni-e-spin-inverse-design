# data/dataset.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold


class Dataset:
    """
    Unified class for loading, preprocessing, and splitting data.
    """

    def __init__(self, definition, scaler=None):
        """
        Parameters
        ----------
        definition : dict
            Must contain:
                - file_path : path to CSV/Excel file
                - file_type : 'csv' or 'excel'
                - target_column : name of target column
        scaler : scikit-learn transformer, optional
            e.g., StandardScaler, MinMaxScaler
        """
        self.definition = definition

        # --- Load data ---
        file_path = definition.get("file_path")
        file_type = definition.get("file_type", "csv")
        if file_path:
            if file_type == "csv":
                self.df = pd.read_csv(file_path)
                self.df.drop("diameter_stdev", axis=1, inplace=True)

            elif file_type == "excel":
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("file_type must be 'csv' or 'excel'")
        else:
            raise ValueError("file_path must be provided in definition.")

        # --- Check target column ---
        self.target_column = definition.get("target_column")
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in DataFrame.")

        # --- Preprocessing ---
        self.scaler = scaler or StandardScaler()
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]

    # --- Data Access ---
    def get_dataframe(self):
        """Return full DataFrame."""
        return self.df

    def preview(self, n=5):
        """Return first n rows of DataFrame."""
        return self.df.head(n)

    def shape(self):
        """Return DataFrame shape."""
        return self.df.shape

    # --- Feature/Target Access ---
    def get_features_target(self, scaled=False):
        """Return X and y. Apply scaling if scaled=True."""
        X = self.X
        if scaled:
            X = self.scaler.fit_transform(X)
        return X, self.y

    # --- Train/Test Splits ---
    def train_test_split(self, test_size=0.2, random_state=None, scaled=False):
        X, y = self.get_features_target(scaled=scaled)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def kfold_split(self, n_splits=5, shuffle=True, random_state=None, scaled=False):
        X, y = self.get_features_target(scaled=scaled)
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            yield X_train, X_test, y_train, y_test
