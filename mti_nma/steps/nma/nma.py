import logging
from sys import platform
from typing import Optional, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk
from vtk.util import numpy_support

from datastep import Step, log_run_params
import aics_dask_utils
from .nma_utils import get_vtk_verts_faces, run_nma, draw_whist, get_eigvec_mags
from .nma_utils import color_vertices_by_magnitude

# Run matplotlib in the background
matplotlib.use('Agg')

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Nma(Step):
    """
    This step is used to run normal mode analysis on a given mesh by:
    1) Extracting vertices and faces from vtk polydata mesh
    2) Cosntructing Hessian matrix from mesh connectivity
    3) Finding eigenvalues and eigenvectors of hessian
    4) Generating a histogram of the eigenvalues

    Parameters
    ----------
    direct_upstream_tasks: List containing a Class name
        Lists the class which is directly prior to this one in the worflow
    """

    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [],
        filepath_columns=["w_FilePath", "v_FilePath", "vmag_FilePath", "fig_FilePath"],
        **kwargs
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            **kwargs
        )

    @log_run_params
    def run(
        self,
        mode_list=list(range(6)),
        avg_df=None,
        struct="Nuc",
        norm_vecs=True,
        n_revs=4,
        n_frames=64,
        path_blender=None,
        distributed_executor_address: Optional[str] = None,
        **kwargs
    ):
        """
        This function will run normal mode analysis on an average mesh.
        It will then create heatmaps of the average mesh for the desired set of modes,
        where the coloring shows the relative amplitudes of vertex oscillation in
        the mode.

        Parameters
        ----------
        mode_list: list
            List of indices of modes to create heatmap files for.
            Defaults the six lowest energy modes (i.e. 0-5).

        avg_df: dataframe
            dataframe containing results from running Avgshape step
            See the construction of the manifest in avgshape.py for details

        path_blender: str
            Path to your local download of the Blender Application.
            If on Mac, the default Blender Mac download location is used.

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        norm_vecs: bool
            Choose whether to set all eigenvectors to the same length or not.

        n_revs: int
            Number of revolutions for results visualization to make.

        n_frames: int
            Number of frames to split visualization into

        distributed_executor_address: Optional[str]
            An optional distributed executor address to use for job distribution.
            Default: None (no distributed executor, use local threads)
        """

        # If no dataframe is passed in, load manifest from previous step
        if avg_df is None:
            avg_df = pd.read_csv(
                self.step_local_staging_dir.parent / "avgshape_"
                f"{struct}" / "manifest.csv"
            )

        # Create directory to hold NMA results
        nma_data_dir = self.step_local_staging_dir / "nma_data"
        nma_data_dir.mkdir(parents=True, exist_ok=True)

        reader = vtk.vtkPLYReader()
        reader.SetFileName(str(avg_df["AvgShapeFilePath"].iloc[0]))
        reader.Update()
        polydata = reader.GetOutput()

        verts, faces = get_vtk_verts_faces(polydata)
        w, v = run_nma(verts, faces)
        draw_whist(w)
        vmags = get_eigvec_mags(v)

        # Working the visualization of eigenvectors on VTK
        n = polydata.GetNumberOfPoints()
        writer = vtk.vtkPolyDataWriter()

        for id_mode in range(9):

            # 1st Get eigenvector of interest as a Nx3 array
            arr_eigenvec = v.T[id_mode, :].reshape(3, -1).T

            if norm_vecs:

                # Calculate eigenvector norm
                arr_eigenvec_norm = np.repeat(np.sqrt(
                    np.power(arr_eigenvec, 2).sum(axis=1, keepdims=True)), 3, axis=1)

                # Normalize eigenvectory to unit
                arr_eigenvec /= arr_eigenvec_norm

            # Convert numpy array to vtk
            eigenvec = numpy_support.numpy_to_vtk(
                num_array=arr_eigenvec,
                deep=True,
                array_type=vtk.VTK_DOUBLE)
            eigenvec.SetName('Eigenvector')

            # Assign eigenvectors as mesh points
            polydata.GetPointData().AddArray(eigenvec)

            for id_theta, theta in enumerate(
                np.linspace(0, n_revs * 2 * np.pi, n_frames)
            ):

                # Update mesh points according to eigenvector
                for i in range(n):
                    xo, yo, zo = polydata.GetPoints().GetPoint(i)
                    x = xo + arr_eigenvec[i, 0] * np.sin(theta)
                    y = yo + arr_eigenvec[i, 1] * np.sin(theta)
                    z = zo + arr_eigenvec[i, 2] * np.sin(theta)
                    polydata.GetPoints().SetPoint(i, x, y, z)

                # Write mesh with new coordinates
                writer.SetInputData(polydata)
                writer.SetFileName(str(
                    nma_data_dir / f"avgshape_{struct}_M{id_mode}_T{id_theta:03d}.vtk"))
                writer.Write()

        fig_path = nma_data_dir / f"w_fig_{struct}.pdf"
        plt.savefig(fig_path, format="pdf")
        w_path = nma_data_dir / f"eigvals_{struct}.npy"
        np.save(w_path, w)
        v_path = nma_data_dir / f"eigvecs_{struct}.npy"
        np.save(v_path, v)
        vmags_path = nma_data_dir / f"eigvecs_mags_{struct}.npy"
        np.save(vmags_path, vmags)

        # Create manifest with eigenvectors, eigenvalues, and hist of eigenvalues
        self.manifest = pd.DataFrame({
            "Label": "nma_avg_mesh",
            "w_FilePath": w_path,
            "v_FilePath": v_path,
            "vmag_FilePath": vmags_path,
            "fig_FilePath": fig_path,
            "Structure": struct
        }, index=[0])

        # If no blender path passed: use default for mac and throw error otherwise
        if path_blender is None:
            if platform == "darwin":
                log.info(
                    "Run on Mac with no Blender path provided. Using default path.")
                path_blender = "/Applications/Blender.app/Contents/MacOS/Blender"
            else:
                raise NotImplementedError(
                    "If using any OS except Mac you must pass in the path to your"
                    "Blender download. For example: "
                    "mti_nma all run --path_blender <path_to_blender_application>")

        # Generate heatmap colored mesh
        heatmap_dir = nma_data_dir / "mode_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        path_input_mesh = avg_df["AvgShapeFilePathStl"].iloc[0]

        # Distribute blender mode heatmap generation
        with aics_dask_utils.DistributedHandler(distributed_executor_address) as handler:
            futures = handler.client.map(
                color_vertices_by_magnitude,
                [path_blender for i in range(len(mode_list))],
                [path_input_mesh for i in range(len(mode_list))],
                [vmags_path for i in range(len(mode_list))],
                mode_list,
                [heatmap_dir / f"mode_{mode}_{struct}.blend" for mode in mode_list]
            )

            # Block until all complete
            results = handler.gather(futures)

            # Set manifest with results
            for mode, output_path in results:
                self.manifest[f"mode_{mode}_FilePath"] = output_path
                self.filepath_columns.append(output_path)

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / f"manifest.csv", index=False
        )
