# data/data_loader.py
import pandas as pd


class DataLoader:
    """
    A simple data loader for pandas DataFrames.
    Responsible only for loading and holding data.
    """

    def __init__(self, df=None, file_path=None, file_type='csv', **kwargs):
        """
        Parameters:
            df (pd.DataFrame): Preloaded DataFrame.
            file_path (str): Path to CSV or Excel file.
            file_type (str): 'csv' or 'excel'
            **kwargs: Additional arguments for pandas read_csv/read_excel
        """
        if df is not None:
            self.df = df.copy()
        elif file_path:
            if file_type == 'csv':
                self.df = pd.read_csv(file_path, **kwargs)
            elif file_type == 'excel':
                self.df = pd.read_excel(file_path, **kwargs)
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
