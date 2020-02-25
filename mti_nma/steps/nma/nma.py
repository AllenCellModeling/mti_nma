#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
from typing import List, Optional
import vtk

from datastep import Step, log_run_params

from ..avgshape import Avgshape
from .nma_utils import run_nma, get_eigvec_mags, get_vtk_verts_faces
from .nma_viz import draw_whist, color_vertices_by_magnitude

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
        direct_upstream_tasks: Optional[List["Step"]] = [Avgshape],
        filepath_columns=["w_FilePath", "v_FilePath", "vmag_FilePath", "fig_FilePath"]
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns
        )

    @log_run_params
    def run(self, mode_list=list(range(6)), avg_df=None, path_blender=None, **kwargs):
        """
        This function will run normal mode analysis on an average mesh.
        It will then create heatmaps of the average mesh for the desired set of modes,
        where the coloring shows the relative amplitudes of vertex oscillation in 
        the mode.

        Parameters
        ----------
        mode_list: list
            List of indices of modes to create heatmap files for

        avg_df: dataframe
            dataframe containing results from running Avgshape step
            See the construction of the manifest in avgshape.py for details

        path_blender: str
            Path to your local download of the Blender Application.
            If on Mac, the default Blender Mac download location is used.
        """

        # if no dataframe is passed in, load manifest from previous step
        if avg_df is None:
            avg_df = pd.read_csv(
                self.step_local_staging_dir.parent / "avgshape" / "manifest.csv"
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

        fig_path = nma_data_dir / "w_fig.pdf"
        plt.savefig(fig_path, format="pdf")
        w_path = nma_data_dir / "eigvals.npy"
        np.save(w_path, w)
        v_path = nma_data_dir / "eigvecs.npy"
        np.save(v_path, v)
        vmags_path = nma_data_dir / "eigvecs_mags.npy"
        np.save(vmags_path, vmags)

        # Create manifest with eigenvectors, eigenvalues, and hist of eigenvalues
        self.manifest = pd.DataFrame({
            "Label": "nma_avg_nuc_mesh",
            "w_FilePath" : w_path,
            "v_FilePath": v_path,
            "vmag_FilePath" : vmags_path,
            "fig_FilePath": fig_path,
        }, index=[0])

        # Get Blender app download filepath
        if platform == "darwin":
            path_blender = "/Applications/Blender.app/Contents/MacOS/Blender"
        else:
            raise NotImplementedError(
                "If using any OS except Mac you must pass in the path to your"
                "Blender download. For example: "
                "mti_nma all run --path_blender <path_to_blender_application_download>"
            )

        # Generate heatmap colored mesh
        heatmap_dir = nma_data_dir / "mode_heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        path_input_mesh = avg_df["AvgShapeFilePathStl"].iloc[0]
        for mode in mode_list:
            path_output = heatmap_dir / f"mode_{mode}.blend"
            color_vertices_by_magnitude(
                path_blender, path_input_mesh, vmags_path, mode, path_output
            )
            self.manifest[f"mode_{mode}_FilePath"] = path_output

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / "manifest.csv", index=False
        )
