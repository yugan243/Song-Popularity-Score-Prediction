from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

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

