import pandas as pd
import os
from typing import Optional, Dict

def load_data(train_path: Optional[str] = None, 
              test_path: Optional[str] = None, 
              config: Optional[Dict] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test datasets from CSV files, optionally using a config dict.

    Args:
        train_path (str, optional): Path to the training dataset.
        test_path (str, optional): Path to the test dataset.
        config (dict, optional): Config dictionary with data paths.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_df, test_df
    """
    if config is not None:
        train_path = config['data']['train_path']
        test_path = config['data']['test_path']
    else:
        train_path = train_path or "data/raw/train.csv"
        test_path = test_path or "data/raw/test.csv"

    print(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")

    return train_df, test_df
