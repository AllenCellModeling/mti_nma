import numpy as np
import itertools
import os
from scipy.sparse.linalg import eigsh
import seaborn as sb
import matplotlib.pyplot as plt


def run_nma(mesh_verts, mesh_faces):
    """
    Runs normal mode analysis on a given mesh.
    :param mesh: mesh to run normal mode analysis on
    :return: eigenvalues and eigenvectors from normal mode analysis
    """
    hess = get_hessian_from_mesh(mesh_verts, mesh_faces)
    w, v = get_eigs_from_mesh(hess)
    return w, v


def get_hessian_from_mesh(mesh_verts, mesh_faces):
    """Find Hessian for mesh defined by input vertices and faces.
    :param mesh_verts: mesh vertices
    :param mesh_faces: mesh faces
    :return: Hessian matrix describing connectivity of mesh
    """

    npts = int(mesh_verts.shape[0])
    ndims = int(mesh_verts[0].shape[0])

    # create hessian matrix of size 3N, allowing each pair of points to have x,y,z components
    hess = np.zeros([ndims*npts, ndims*npts])
    
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
                R = np.linalg.norm(xyz1-xyz2)
                val = -(xyz2[ind2] - xyz1[ind2])*(xyz2[ind1] - xyz1[ind1])/(R**2)

                hess[npts*ind1 + i, npts*ind2 + j] = val
                hess[npts*ind2 + j, npts*ind1 + i] = val
                hess[npts*ind1 + j, npts*ind2 + i] = val
                hess[npts*ind2 + i, npts*ind1 + j] = val

    # fill in diagonal and sub-block diagonal elements of hessian
    for ind_pair in ind_pairs:
        ind1, ind2 = ind_pair
        for pt in range(npts):
            hess[ind1*npts+pt][ind2*npts+pt] = -np.sum(hess[ind1*npts+pt][ind2*npts:(ind2+1)*npts])
            if ind1!=ind2:
                hess[ind2*npts+pt][ind1*npts+pt] = hess[ind1*npts+pt][ind2*npts+pt]

    return hess


def get_eigs_from_mesh(hess):
    """Get eigenvalues and eigenvectors of hessian.
    :param hess: hessian for mesh
    :return: hessian eigenvalues (w) and eigenvectors (v) (v[:,i] correcsponds to w[i])
    """

    # use solver to get eigenvalues (w) and eigenvectors (v)
    w, v = np.linalg.eigh(hess)
    return w, v


def draw_whist(w):
    """Draw histogram of eigenvalues (w2*m/k)
    :param w: list of eigenvalue parameter w values
    """
    minval = min(w)-0.5
    maxval = max(w)+0.5
    if len(w) < 20:
        N = int(max(w)+2)
    else:
        N = 30
    sb.distplot(w, kde=False, bins=N)
    plt.xlabel('Eigenvalues (w2*m/k)')
    plt.ylabel('Counts')
