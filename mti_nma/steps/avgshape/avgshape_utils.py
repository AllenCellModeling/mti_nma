import matplotlib.pyplot as plt
from pathlib import Path
from stl import mesh
import numpy as np
import subprocess


def run_shcoeffs_analysis(df, savedir):

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
            str(savedir / Path(f"scatter-{id_plot}.svg"))
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
            str(savedir / Path(f"bar-{id_plot}.svg"))
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


def gen_uniform_trimesh(path_input_mesh, mesh_density, path_output, path_blender):

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

    args = f"-i {path_input_mesh} -o {path_output} -d {mesh_density}"
    cmd = f"{path_blender} -b -P uniform_trimesh -- {args}"
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    p.terminate()


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
