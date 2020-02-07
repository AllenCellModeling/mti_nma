#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vtk

from datastep import Step, log_run_params
from datastep.file_utils import manifest_filepaths_rel2abs

from ..avgshape import Avgshape
from .nma_utils import run_nma

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
        clean_before_run=False,
        direct_upstream_tasks=[Avgshape],
        filepath_columns=["w_FilePath", "v_FilePath", "fig_FilePath"],
        config=None
    ):
        super().__init__(
            clean_before_run=clean_before_run,
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            config=config
        )

    @log_run_params
    def run(self, **kwargs):

        # Load avg shape manifest and read avg mesh file out
        avgshape = Avgshape()
        manifest_filepaths_rel2abs(avgshape)
        df = avgshape.manifest.copy()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(
            df[df['label'] == 'Average_nuclear_mesh']['AvgShapeFilePath'].iloc[0])
        reader.Update()
        polydata = reader.GetOutput()

        # Create directory to hold NMA results
        nma_data_dir = self.step_local_staging_dir / Path("nma_data")
        nma_data_dir.mkdir(parents=True, exist_ok=True)

        # run NMA on avg mesh and save results to local stagings
        w, v, wfig = run_nma(polydata)
        w_path = nma_data_dir / Path('eigvals.npy')
        np.save(w_path, w)
        v_path = nma_data_dir / Path('eigvecs.npy')
        np.save(v_path, v)
        fig_path = nma_data_dir / Path('w_fig.pdf')
        plt.savefig(fig_path, format='pdf')

        # Create manifest with eigenvectors, eigenvalues, and hist of eigenvalues
        self.manifest = pd.DataFrame({
            'label': 'nma_avg_nuc_mesh',
            'w_FilePath' : w_path,
            "v_FilePath": v_path,
            "fig_FilePath": [fig_path],
        })

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
