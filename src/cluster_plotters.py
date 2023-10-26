import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def plot_cluster_ellipses(df, ax=None, color=None):
    if ax is None:
        fig, ax  = plt.subplots(figsize=(13,13))

    color_map = False
    if color is None:
        color_map = True
 
    unique_label,cluster_rep_index, counts = np.unique(df.labels, return_index=True, return_counts=True)
    cmap = plt.cm.get_cmap('turbo')
    norm = matplotlib.colors.Normalize(vmin=min(df.labels), vmax=max(df.labels))
    
    for label, rep_id in zip(unique_label, cluster_rep_index):
        if label != -1:
            if color_map:
                color=cmap(norm(label))
            
            
            ax.annotate(label, df.iloc[rep_id][["fx", "fy"]]+[-7,-1],color=color,alpha=1, weight='normal', ha='center', va='center', size=9).draggable()

            cluster_x_y = df[df.labels==label][["fx", "fy"]].to_numpy() 
            confidence_ellipse(cluster_x_y[:, 0], cluster_x_y[:, 1], ax, edgecolor=color, n_std=3)
    return ax
    