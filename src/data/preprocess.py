import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataStandardizer:
    """
    Standardizes numerical features in a DataFrame using StandardScaler.
    Keeps non-numerical columns and the target column unchanged.
    Fit on training data, then transform train/test/val as needed.
    """
    def __init__(self):
        self.scaler = None
        self.num_cols = None
        self.target_col = None

    def fit(self, df, target_col=None):
        """
        Fit the scaler on the numerical columns of the DataFrame, excluding the target column if provided.
        Args:
            df (pd.DataFrame): The DataFrame to fit the scaler on.
            target_col (str, optional): Name of the target column to exclude from scaling.
        Returns:
            self
        """
        self.target_col = target_col
        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in self.num_cols:
            self.num_cols.remove(target_col)
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.num_cols])
        return self

    def transform(self, df):
        """
        Transform the numerical feature columns of the DataFrame using the fitted scaler.
        Args:
            df (pd.DataFrame): The DataFrame to transform.
        Returns:
            pd.DataFrame: DataFrame with standardized numerical feature columns.
        """
        df_copy = df.copy()
        # Only transform columns that exist in the DataFrame
        cols_to_transform = [col for col in self.num_cols if col in df_copy.columns]
        df_copy[cols_to_transform] = self.scaler.transform(df_copy[cols_to_transform])
        return df_copy

    def fit_transform(self, df, target_col=None):
        """
        Fit the scaler and transform the DataFrame, excluding the target column from scaling.
        Args:
            df (pd.DataFrame): The DataFrame to fit and transform.
            target_col (str, optional): Name of the target column to exclude from scaling.
        Returns:
            pd.DataFrame: DataFrame with standardized numerical feature columns.
        """
        self.fit(df, target_col=target_col)
        return self.transform(df)




class CombinedFrequencyEncoder:
    def __init__(self, columns):
        """
        :param columns: List of columns to frequency encode
        """
        self.columns = columns
        self.freq_maps = {}

    def fit(self, train_df, test_df):
        """
        Fit frequency maps using combined train + test datasets.
        """
        combined = pd.concat([train_df[self.columns], test_df[self.columns]], axis=0)
        for col in self.columns:
            freq = combined[col].value_counts(normalize=True)
            self.freq_maps[col] = freq

    def transform(self, df):
        """
        Apply frequency encoding to a given DataFrame.
        """
        df_encoded = df.copy()
        for col in self.columns:
            df_encoded[f"{col}_encoded"] = df_encoded[col].map(self.freq_maps[col])
        return df_encoded

    def fit_transform(self, train_df, test_df):
        """
        Fit on combined data, then transform both train and test.
        Returns: encoded_train_df, encoded_test_df
        """
        self.fit(train_df, test_df)
        train_encoded = self.transform(train_df)
        test_encoded = self.transform(test_df)
        return train_encoded, test_encoded
