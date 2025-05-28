import yaml
import os


def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)



def save_dataframe(df, path):
    """
    Save a DataFrame to a CSV file, ensuring the directory exists.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Full path (including filename) to save the CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] DataFrame saved to {path}")