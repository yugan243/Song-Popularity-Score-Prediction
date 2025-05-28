import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.plot_utils import save_figure


def correlation_matrix(df, config, filename="correlation_matrix.png", figsize=(16, 12), annot=False, cmap="coolwarm"):
    """
    Plot and save the correlation matrix heatmap for the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame for which to compute the correlation matrix.
        config (dict): Configuration dictionary containing output paths.
        filename (str, optional): Name of the file to save the figure as. Defaults to "correlation_matrix.png".
        figsize (tuple, optional): Figure size. Defaults to (16, 12).
        annot (bool, optional): Whether to annotate the heatmap. Defaults to False.
        cmap (str, optional): Colormap for the heatmap. Defaults to "coolwarm".
    """
    # Select only numerical columns
    num_df = df.select_dtypes(include=["number"])
    corr = num_df.corr()
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr, annot=annot, fmt=".2f", cmap=cmap, square=True, cbar_kws={"shrink": .8})
    plt.title("Correlation Matrix", fontsize=18)
    plt.tight_layout()
    fig = plt.gcf()
    save_figure(fig, filename, config)
    plt.show()
    plt.close(fig)


def plot_categorical_count(df, col, top_n=20, figsize=(10, 4)):
    """
    Plots the count of categories for a given column.
    If unique values > top_n, plots only the top_n categories.
    """
    n_unique = df[col].nunique()
    plt.figure(figsize=figsize)
    if n_unique <= top_n:
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order)
        plt.title(f"Frequency of {col}")
    else:
        top_cats = df[col].value_counts().head(top_n)
        sns.barplot(x=top_cats.index, y=top_cats.values)
        plt.title(f"Top {top_n} categories in {col}")
        plt.ylabel("Count")
    plt.xticks(rotation=45 if n_unique <= top_n else 90)
    plt.tight_layout()
    plt.show()

def plot_target_by_category(df, col, target_col, top_n=20, figsize=(10, 4)):
    """
    Plots the distribution of the target variable by category.
    For high-cardinality columns, shows only the top_n categories by frequency.
    """
    n_unique = df[col].nunique()
    plt.figure(figsize=figsize)
    if n_unique <= top_n:
        sns.boxplot(data=df, x=col, y=target_col)
        plt.title(f"{target_col} by {col}")
        plt.xticks(rotation=45)
    else:
        top_cats = df[col].value_counts().head(top_n).index
        means = df[df[col].isin(top_cats)].groupby(col)[target_col].mean().sort_values(ascending=False)
        sns.barplot(x=means.index, y=means.values)
        plt.title(f"Mean {target_col} for top {top_n} {col}")
        plt.ylabel(f"Mean {target_col}")
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    
def plot_violin_by_category(df, col, target_col, figsize=(10, 5)):
    """
    Plots a violin plot of the target variable by a categorical column.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=col, y=target_col, inner='quartile')
    plt.title(f'Violin plot of {target_col} by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_swarm_by_category(df, col, target_col, figsize=(10, 5), size=2):
    """
    Plots a swarm plot of the target variable by a categorical column.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=figsize)
    sns.swarmplot(data=df, x=col, y=target_col, size=size)
    plt.title(f'Swarm plot of {target_col} by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()