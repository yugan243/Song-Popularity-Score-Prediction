from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def initial_data_inspection(df, name="Dataset"):
    print(f"===== {name} Overview =====")
    print(f"Shape: {df.shape}")
    display(df.head())
    
    print("\nInfo:")
    display(df.info())
    
    print("\nBasic Statistics:")
    display(df.describe(include='all').T)
    
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    print("\nUnique Values per Column (Top 10):")
    unique_vals = df.nunique().sort_values(ascending=False)
    display(unique_vals.head(10))
    
    print("\nMissing Values (Top 10):")
    missing = df.isnull().sum().sort_values(ascending=False)
    display(missing.head(10))
    
    # Display missing value percentages
    print("\nMissing Value Percentages (Top 10):")
    missing_percent = (df.isnull().mean() * 100).sort_values(ascending=False)
    display(missing_percent.head(10))
    
    # Visualize missing values as a heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(f"Missing Values Heatmap: {name}")
    plt.show()
    
    # Visualize missing value counts as a histogram
    plt.figure(figsize=(10, 4))
    missing[missing > 0].plot(kind='bar')
    plt.title(f"Missing Values Count per Column: {name}")
    plt.ylabel("Count of Missing Values")
    plt.xlabel("Column")
    plt.tight_layout()
    plt.show()



def get_highly_correlated_pairs(df, threshold=0.8):
    """
    Finds feature pairs with correlation above `threshold` or below `-threshold`.

    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric features.
        threshold (float): Correlation threshold (default=0.8).
        
    Returns:
        pd.DataFrame: DataFrame of correlated feature pairs with columns:
                      ['Feature 1', 'Feature 2', 'Correlation']
    """
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Mask the upper triangle and diagonal
    mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
    corr_values = corr_matrix.where(mask)
    
    # Stack to long format
    high_corr = corr_values.stack().reset_index()
    high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    # Filter for signed correlations > threshold or < -threshold
    high_corr = high_corr[
        (high_corr['Correlation'] > threshold) | 
        (high_corr['Correlation'] < -threshold)
    ]
    
    # Sort by actual correlation value
    high_corr = high_corr.sort_values(by='Correlation', ascending=False).reset_index(drop=True)
    
    display(high_corr)
    return high_corr
