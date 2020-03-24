import itertools
import logging
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

# Run matplotlib in the background
matplotlib.use('Agg')

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def run_nma(mesh_verts, mesh_faces):
    """
    Runs normal mode analysis on a given mesh by:
    1) Extracting vertices and faces from vtk polydata mesh
    2) Cosntructing Hessian matrix from mesh connectivity
    3) Finding eigenvalues and eigenvectors of hessian
    4) Generating a histogram of the eigenvalues

    Parameters
    ----------
    mesh_verts: numpy array
        Array of locations of all mesh vertices
    mesh_faces: numpy array
        Array of indices of vertices making up each mesh face

    Returns
    -------
    w: Numpy 1D array
        An array of eigenvalue values
    v: Numpy 2D array
        Each array in this array gives the displacement vector for a vertex in the mesh
        The array giving the eigenvectors associated with eigenvalue w[i] are in v[:. i]
    """

    hess = get_hessian_from_mesh(mesh_verts, mesh_faces)
    w, v = get_eigs_from_mesh(hess)
    return w, v


def get_hessian_from_mesh(mesh_verts, mesh_faces):
    """
    Constructs Hessian matrix for this mesh based on mesh connectivity

    Parameters
    ----------
    mesh_verts: Numpy 2D array
        An array of arrays giving the positions of all mesh vertices
        mesh_verts[i][j] gives position of the ith vertex, in the jth spatial dimension
    vmesh_faces: Numpy 2D array
        An array of arrays listing which vertices are connected to form each mesh faces
        mesh_faces[i] gives an array of 3 ints, indicating the indices of the
        mesh_verts indices of vertices connected to form this trimesh faces
    Returns
    -------
    hess: Numpy 2D array
        dNxdN Hessian matrix describing mesh connectivity of N-vertex mesh in d dims
    """

    npts = int(mesh_verts.shape[0])
    ndims = int(mesh_verts[0].shape[0])

    # create hessian matrix of size 3N (xyz)
    hess = np.zeros([ndims * npts, ndims * npts])

    # get all unique pairs of points that are connected in the spring network
    edges = []
    for face in mesh_faces:
        for pair in list(itertools.combinations(face, 2)):
            edges.append(pair)

    # cycle through pairs of x,y,z coordinates
    ind_pairs = list(itertools.combinations_with_replacement(range(ndims), 2))
    for ind_pair in ind_pairs:
        ind1, ind2 = ind_pair

        # cycle through pairs of connected points in mesh
        for edge in edges:
            i, j = edge

            # fill in off-diagonal hessian elements
            if (i != j):
                xyz1 = mesh_verts[i]
                xyz2 = mesh_verts[j]
                R = np.linalg.norm(xyz1 - xyz2)
                if R == 0:
                    log.info('*********Identical vertices found')
                else:
                    val = -(xyz2[ind2] - xyz1[ind2]) * (xyz2[ind1] - xyz1[ind1]) / R**2

                hess[npts * ind1 + i, npts * ind2 + j] = val
                hess[npts * ind2 + j, npts * ind1 + i] = val
                hess[npts * ind1 + j, npts * ind2 + i] = val
                hess[npts * ind2 + i, npts * ind1 + j] = val

    # fill in diagonal and sub-block diagonal elements of hessian
    for ind_pair in ind_pairs:
        ind1, ind2 = ind_pair
        for pt in range(npts):
            hess[ind1 * npts + pt][ind2 * npts + pt] = -np.sum(
                hess[ind1 * npts + pt][ind2 * npts:(ind2 + 1) * npts]
            )
            if ind1 != ind2:
                hess[ind2 * npts + pt][ind1 * npts + pt] = hess[
                    ind1 * npts + pt][ind2 * npts + pt]

    return hess


def get_eigs_from_mesh(hess):
    """
    Finds the eigenvalues and eigenvectors of the Hessian matrix.
    The eigenvalues are related to the frequencies of the normal modes
    The eigenvectors describe the relative displacements of each mesh
    vertex for each normal mode.

    Parameters
    ----------
    hess: Numpy 2D array
        dNxdN Hessian matrix describing mesh connectivity of N-vertex mesh in d dims
    Returns
    -------
    w: Numpy 1D array
        An array of eigenvalue values
    v: Numpy 2D array
        Each array in this array gives the displacement vector for a vertex in the mesh
        The array giving the eigenvectors associated with eigenvalue w[i] are in v[:. i]
    """

    # use solver to get eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eigh(hess)
    return w, v


def get_eigvec_mags(v):
    """
    Returns the magnitudes of the eigenvectors for a given mode.
    Used for visualization.

    Parameters
    ----------
    v: Numpy 2D array
        Each array in this array gives the displacement vector for a vertex in the mesh
        The array giving the eigenvectors associated with eigenvalue w[i] are in v[:. i]
    Returns
    -------
    vmags: 2D Numpy array
        Each array in this array gives the relative displacement magntiude for a vertex
        in the mesh. The array giving the magnitudes associated with eigenvalue w[i]
        are in vmags[:. i]
    """

    nmodes = v.shape[1]
    nverts = int(v.shape[0] / 3)
    vmags = np.empty([nverts, nmodes])
    for j in range(nmodes):
        vecs = [
            [v[i, j],
             v[i + nverts, j],
             v[i + 2 * nverts, j]] for i in range(nverts)
        ]
        mags = [np.linalg.norm(vecs[i], axis=0) for i in range(nverts)]
        vmags[:, j] = mags / max(mags)
    return vmags


def get_vtk_verts_faces(polydata):
    """
    Extracts the mesh vertices and faces encoded in the VTK polydata object

    Parameters
    ----------
    polydata: VTK polydata object
        Mesh extracted from .vtk file
    Returns
    -------
    mesh_verts: Numpy 2D array
        An array of arrays giving the positions of all mesh vertices
        mesh_verts[i][j] gives position of the ith vertex, in the jth spatial dimension
    vmesh_faces: Numpy 2D array
        An array of arrays listing which vertices are connected to form each mesh faces
        mesh_faces[i] gives an array of 3 ints, indicating the indices of the
        mesh_verts indices of vertices connected to form this trimesh faces
    """

    def get_faces(i, polydata):
        """
        Extracts vertices making up the ith face of the polydata mesh

         Parameters
        ----------
        i : int
            Index of the face for which we want to get the vertex points
        polydata: VTK polydata object
            Mesh extracted from .vtk file
        Returns
        -------
        mesh_verts: Numpy 1D array
            An array of indices of vertices making up the three vertices of this face
        """

        cell = polydata.GetCell(i)
        ids = cell.GetPointIds()
        return np.array([ids.GetId(j) for j in range(ids.GetNumberOfIds())])

    # Collect all faces and vertices into arrays
    mesh_faces = np.array(
        [get_faces(i, polydata) for i in range(polydata.GetNumberOfCells())]
    )
    mesh_verts = np.array(
        [np.array(
            polydata.GetPoint(i)) for i in range(polydata.GetNumberOfPoints())]
    )
    return mesh_verts, mesh_faces


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


def color_vertices_by_magnitude(
    path_blender, path_input_mesh, path_vmags, mode, path_output
):
    """
    Creates blender file with mesh vertices colored by eigenvector
    magnitudes for a given mode and saves as .blend file
    The Python script to do this must be run in Blender, so we open Blender
    in bash and run the python script there.

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

    args = f"-i {path_input_mesh} -o {path_output} -m {mode} -v {path_vmags}"
    python_callable = str(
        Path(__file__).parent.parent.parent / "bin" / "color_vertices.py"
    )
    cmd = f"{path_blender} -b -P {python_callable} -- {args}"
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")

    # If an error occurs, log it as output to the terminal
    if p.stderr is not None:
        log.info(p.stderr)

    # Closes subprocess but not until Blender has finished
    p.wait()

    return mode, path_output
