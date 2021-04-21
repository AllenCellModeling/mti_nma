import logging
from pathlib import Path

import vtk
import matplotlib.pyplot as plt
import numpy as np
from stl import mesh
from aicscytoparam import cytoparam
from aicsimageio import writers
from skimage import filters as skfilters
from vtk.util import numpy_support as vtknp
import pyvista as pv
import pyacvd

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def run_shcoeffs_analysis(df, savedir, struct):
    """
    Extracts the mesh vertices and faces encoded in the VTK polydata object

    Parameters
    ----------
    polydata: VTK polydata object
        Mesh extracted from .vtk file
    Returns
    -------
    df: dataframe
        dataframe containing spherical harmonic decomposition information
    savedir: path or string
        Path to directory where results figures should be saved.
    struct: str
        String giving name of structure to run analysis on.
        Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).
    """

    list_of_scatter_plots = [
        ("shcoeffs_L0M0C", "shcoeffs_L2M0C"),
        ("shcoeffs_L0M0C", "shcoeffs_L2M2C"),
        ("shcoeffs_L0M0C", "shcoeffs_L2M1S"),
        ("shcoeffs_L0M0C", "shcoeffs_L2M1C"),
    ]

    for id_plot, (varx, vary) in enumerate(list_of_scatter_plots):

        fs = 18
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(df[varx], df[vary], "o")
        ax.set_xlabel(varx, fontsize=fs)
        ax.set_ylabel(vary, fontsize=fs)
        plt.tight_layout()
        fig.savefig(
            str(savedir / Path(f"scatter-{id_plot}_{struct}.svg"))
        )
        plt.close(fig)

    list_of_bar_plots = ["shcoeffs_L0M0C"]

    for id_plot, var in enumerate(list_of_bar_plots):

        fs = 18
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.bar(str(df.index), df[var])
        ax.set_xlabel("CellId", fontsize=10)
        ax.set_ylabel(var, fontsize=fs)
        ax.tick_params("x", labelrotation=90)
        plt.tight_layout()
        fig.savefig(
            str(savedir / Path(f"bar-{id_plot}_{struct}.svg"))
        )
        plt.close(fig)


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


def save_mesh_as_stl(polydata, fname):

    # get mesh vertices and faces from polydata object
    verts, faces = get_vtk_verts_faces(polydata)

    # Create the stl Mesh object
    nuc_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            nuc_mesh.vectors[i][j] = verts[f[j], :]

    # Write the mesh to file"
    nuc_mesh.save(fname)


def save_mesh_as_obj(polydata, fname):

    writer = vtk.vtkOBJWriter()
    writer.SetInputData(polydata)
    writer.SetFileName(str(fname))
    writer.Write()


def save_voxelization(polydata, fname):

    domain, _ = cytoparam.voxelize_meshes([polydata])
    with writers.ome_tiff_writer.OmeTiffWriter(fname, overwrite_file=True) as writer:
        writer.save(
            255 * domain,
            dimension_order='ZYX',
            image_name=fname.stem
        )
    return domain


def save_displacement_map(grid, fname):

    grid = grid.reshape(1, *grid.shape).astype(np.float32)

    with writers.ome_tiff_writer.OmeTiffWriter(fname, overwrite_file=True) as writer:
        writer.save(
            grid,
            dimension_order='ZYX',
            image_name=fname.stem
        )


def get_smooth_and_coarse_mesh_from_voxelization(img, sigma, npoints):
    """
    Converts an image into a triangle mesh with even distributed points.
    First we use a Gaussian kernel with size (sigma**3) to smooth the
    input image. Next we apply marching cubes (vtkContourFilter) to obtain
    a first mesh, which is used as input to a Voronoi-based clustering
    that is responsible for remeshing. Details can be found here:
    https://github.com/pyvista/pyacvd

    Parameters
    ----------
    img: np.array
        Input image corresponding to the voxelized version of the original
        average mesh.
    sigma: float
        Gaussian kernel size.
    npoints: int
        Number of points used to create the Voronoi clustering. The larger
        this value the more points the final mesh will have.
    Returns
    -------
    remesh_vtk: vtkPolyData
        Triangle with even distirbuted points.
    """


    rad = 5
    img = np.pad(img, ((rad, rad), (rad, rad), (rad, rad)))
    d, h, w = img.shape
    img = skfilters.gaussian(img > 0, sigma=sigma, preserve_range=True)
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    values = (255 * img).ravel().astype(np.uint8)
    values = vtknp.numpy_to_vtk(values, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    imagedata.GetPointData().SetScalars(values)
    cf = vtk.vtkContourFilter()
    cf.SetInputData(imagedata)
    cf.SetValue(0, 255.0 / np.exp(1.0))
    cf.Update()
    mesh = cf.GetOutput()

    pv_temp = pv.PolyData(mesh)
    cluster = pyacvd.Clustering(pv_temp)
    cluster.cluster(npoints)
    remesh = cluster.create_mesh()
    remesh_vtk = vtk.vtkPolyData()
    remesh_vtk.SetPoints(remesh.GetPoints())
    remesh_vtk.SetVerts(remesh.GetVerts())
    remesh_vtk.SetPolys(remesh.GetPolys())
    return remesh_vtk
