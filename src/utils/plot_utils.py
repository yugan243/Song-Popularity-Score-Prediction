import os



def save_figure(fig, filename, config):
    """
    Save a matplotlib figure to the figures directory specified in the config.

    Args:
        fig (matplotlib.figure.Figure): The matplotlib figure object to save.
        filename (str): The name of the file to save the figure as (e.g., 'plot.png').
        config (dict): Configuration dictionary containing output paths, 
            must include ['output']['figures_dir'].

    Raises:
        KeyError: If 'figures_dir' is not found in the config.
        OSError: If the directory cannot be created or the file cannot be saved.

    Example:
        >>> config = load_config()
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [4, 5, 6])
        >>> save_figure(fig, "example_plot.png", config)
    """
    
    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    save_path = os.path.join(figures_dir, filename)
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Figure saved to {save_path}")