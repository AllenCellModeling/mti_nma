import logging
from sys import platform
from typing import Optional, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import vtk

from datastep import Step, log_run_params
from .. import dask_utils
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

        distributed_executor_address: Optional[str]
            An optional distributed executor address to use for job distribution.
            Default: None (no distributed executor, use local threads)
        """

        # If no dataframe is passed in, load manifest from previous step
        if avg_df is None:
            avg_df = pd.read_csv(
                self.step_local_staging_dir.parent / "avgshape" / "manifest_"
                f"{struct}.csv"
            )

        # Create directory to hold NMA results
        nma_data_dir = self.step_local_staging_dir / "nma_data"
        nma_data_dir.mkdir(parents=True, exist_ok=True)

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(avg_df["AvgShapeFilePath"].iloc[0]))
        reader.Update()
        polydata = reader.GetOutput()

        verts, faces = get_vtk_verts_faces(polydata)
        w, v = run_nma(verts, faces)
        draw_whist(w)
        vmags = get_eigvec_mags(v)

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
            "Label": "nma_avg_nuc_mesh",
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
        with dask_utils.DistributedHandler(distributed_executor_address) as handler:
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
