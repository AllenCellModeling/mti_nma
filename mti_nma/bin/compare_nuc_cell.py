import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib

# Run matplotlib in the background
matplotlib.use('Agg')


def draw_whist():
    """
    Draws a histogram figure of the eigenvalues

    Parameters
    ----------
    w_nuc: Numpy 1D array
        An array of eigenvalue values from NMA of nucleus
    w_cell: Numpy 1D array
        An array of eigenvalue values from NMA of cell
    Returns
    -------
    fig: Matplotlib Figure object
        Figure containg a histogram of eigenvalues
    """

    w_nuc = np.load("local_staging/nma/nma_data/eigvals_Nuc.npy")
    w_cell = np.load("local_staging/nma/nma_data/eigvals_Cell.npy")

    plt.clf()
    fig = plt.figure()

    # set binning
    w_all = np.concat(w_nuc, w_cell)
    minval = min(w_all) - 0.5
    maxval = max(w_all) + 0.5
    if len(w_nuc) < 20:
        N = int(max(w_all) + 2)
    else:
        N = 30
    bins = np.linspace(minval, maxval, N)

    sb.distplot(w_nuc, kde=True, bins=bins, label="Nuc")
    sb.distplot(w_cell, kde=True, bins=bins, label="Cell")
    plt.xlabel("Eigenvalues (w2*m/k)")
    plt.ylabel("Counts")

    plt.savefig("local_staging/compare_fig")

    return fig
