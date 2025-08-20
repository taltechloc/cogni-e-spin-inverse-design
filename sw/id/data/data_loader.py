# data/data_loader.py
import pandas as pd


class DataLoader:
    """
    A simple data loader for pandas DataFrames.
    Responsible only for loading and holding data.
    """

    def __init__(self, config):
        if config.file_path:
            if config.file_type == 'csv':
                self.df = pd.read_csv(config.file_path)
            elif config.file_type == 'excel':
                self.df = pd.read_excel(config.file_path)
            else:
                raise ValueError("file_type must be 'csv' or 'excel'")
        else:
            raise ValueError("Either df or file_path must be provided.")

    def get_dataframe(self):
        """Returns the loaded DataFrame."""
        return self.df

    def preview(self, n=5):
        """Returns the first n rows of the dataframe"""
        return self.df.head(n)

    def shape(self):
        """Returns the shape of the dataframe"""
        return self.df.shape
