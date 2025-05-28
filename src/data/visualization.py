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
