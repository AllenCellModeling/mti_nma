import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aicsshparam import shtools
from datastep import Step, log_run_params
from stl import mesh

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Avgshape(Step):
    def __init__(
        self,
        direct_upstream_tasks,
        filepath_columns=["AvgShapeFilePath", "AvgShapeFilePathStl"]

    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns
        )

    @log_run_params
    def run_avgshape_step(self, sh_df=None, struct="Nuc", **kwargs):
        """
        This step uses the amplitudes of the spherical harmonic components
        of the nuclear shapes in the dataset to construct an average nuclear mesh.

        Parameters
        ----------
        sh_df: dataframe
            dataframe containing results from running Shparam step
            See the construction of the manifest in shparam.py for details

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).
        """

        # If no dataframe is passed in, load manifest from previous step
        if sh_df is None:
            sh_df = pd.read_csv(
                self.step_local_staging_dir.parent / "shparam" / "manifest_"
                f"{struct}.csv"
            )

        # Fix filepaths and use cell id as dataframe index
        sh_df = sh_df.set_index("CellId", drop=True)

        # Load sh coefficients of all samples in manifest
        coeffs_df = pd.DataFrame([])
        for CellId in sh_df.index:
            coeffs_df_path = sh_df["CoeffsFilePath"][CellId]
            coeffs_df = coeffs_df.append(
                pd.read_csv(coeffs_df_path, index_col=["CellId"]), ignore_index=False
            )

        # Create directory to hold results from this step
        avg_data_dir = self.step_local_staging_dir / "avgshape_data"
        avg_data_dir.mkdir(parents=True, exist_ok=True)

        # Perform some per-cell analysis
        run_shcoeffs_analysis(df=coeffs_df, savedir=avg_data_dir, struct=struct)

        # Avg the sh coefficients over all samples and create avg mesh
        coeffs_df_avg = coeffs_df.agg(['mean'])
        coeffs_avg = coeffs_df_avg.values

        # Number of columns = 2*lmax*lmax
        lmax = int(np.sqrt(0.5 * coeffs_avg.size))

        coeffs_avg = coeffs_avg.reshape(-2, lmax, lmax)

        mesh_avg, _ = shtools.get_reconstruction_from_coeffs(coeffs=coeffs_avg)

        shtools.save_polydata(
            mesh=mesh_avg,
            filename=str(avg_data_dir / f"avgshape_{struct}.vtk")
        )

        # Save mesh as stl file for blender import
        save_mesh_as_stl(mesh_avg, str(avg_data_dir / f"avgshape_{struct}.stl"))

        # Save avg coeffs to csv file
        coeffs_df_avg.to_csv(
            str(avg_data_dir / f"avgshape_{struct}.csv")
        )

        # Save path to avg shape in the manifest
        self.manifest = pd.DataFrame({
            "Label": "Average_mesh",
            "AvgShapeFilePath": avg_data_dir / f"avgshape_{struct}.vtk",
            "AvgShapeFilePathStl": avg_data_dir / f"avgshape_{struct}.stl",
            "Structure": struct,
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path(f"manifest_{struct}.csv"), index=False
        )
        return self.manifest


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
