import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import subprocess


def draw_whist(w):
    """
    Draws a histogram figure of the eigenvalues

    Parameters
    ----------
    w: Numpy 1D array
        An array of eigenvalue values
    Returns
    -------
    fig: Matplotlib Figure object
        Figure containg a histogram of eigenvalues
    """

    plt.clf()
    fig = plt.figure()

    # set binning
    minval = min(w) - 0.5
    maxval = max(w) + 0.5
    if len(w) < 20:
        N = int(max(w) + 2)
    else:
        N = 30
    bins = np.linspace(minval, maxval, N)

    sb.distplot(w, kde=False, bins=bins)
    plt.xlabel("Eigenvalues (w2*m/k)")
    plt.ylabel("Counts")

    return fig


def color_vertices_by_magnitude(path_input_mesh, path_vmags, mode, path_output):
    """
    Creates blender file with mesh vertices colored by eigenvector 
    magnitudes for a given mode and saves as .blend file
    The Python script to do this must be run in Blender, so we open Blender
    in bash and run the python script there.

    If your local copy of Blender is in a different location than the current
    path listed, change the filepath at the start of the `bl` string to 
    your own Blender path.

    The `-b` flag runs Blender headlessly (doesn't open the app GUI) and the
    `-P` flag tells Blender you want to run the python script whose filepath is
    provided after this flag. The `--` indicated to blender that the arguments
    following it are arguments for the python script, not for Blender.

    Parameters
    ----------
    path_input_mesh: str
        Filepath to input mesh to be colored (.stl)
    path_vmags: str
        Filepath to input magnitudes of eigenvector used to color mesh (.npy)
    mode: int
        Index of mode whose eigenvector magnitudes will be used to color mesh
    path_output: str
        Filepath to output file of colored mesh object (.blend)
    """

    bl = "/Applications/Blender.app/Contents/MacOS/Blender -b -P "
    psc = "/Users/juliec/mti/mti_nma/mti_nma/steps/nma/color_vertices.py -- "
    args = f"-i {path_input_mesh} -o {path_output} -m  {mode} -v {path_vmags}"
    cmd = f"{bl} {psc} {args}"
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    p.terminate()
