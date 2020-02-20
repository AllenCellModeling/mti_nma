#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from skimage import io as skio
from aicsshparam import aicsshparam, aicsshtools

from datastep import Step, log_run_params
from datastep.file_utils import manifest_filepaths_rel2abs

from ..singlecell import Singlecell

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Shparam(Step):
    def __init__(
        self,
        clean_before_run=False,
        direct_upstream_tasks: Optional[List["Step"]] = Singlecell,
        filepath_columns=[
            "InitialMeshFilePath", "ShparamMeshFilePath", "CoeffsFilePath"],
        config: Optional[Union[str, Path, Dict[str, str]]] = None
    ):
        super().__init__(
            clean_before_run=clean_before_run,
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            config=config
        )

    @log_run_params
    def run(self, **kwargs):
        """
        This function loads the seg images we want to perform sh parametrization on
        and calculate the sh coefficients. Results are saved as csv files.
        """

        # Get manifest from previous step
        singlecells = Singlecell()
        manifest_filepaths_rel2abs(singlecells)
        sc_df = singlecells.manifest.copy()
        sc_df = sc_df.set_index('CellId')

        # create directory to save data for this step in local staging
        sh_data_dir = self.step_local_staging_dir / Path('shparam_data')
        sh_data_dir.mkdir(parents=True, exist_ok=True)

        # Get spherical harmonic set for segmentation, save and record in manifest
        self.manifest = pd.DataFrame([])
        for CellId in sc_df.index:

            # Read segmentation image
            impath = sc_df['SegFilePath'][CellId]
            seg = skio.imread(impath)

            # Get spherical harmonic decomposition of segmentation
            # Here is the place where I need someone taking a look at the
            # aicsshparam package to see what is the best way to return
            # the outputs
            (coeffs, grid), (_, mesh_initial, _, grid_initial) = aicsshparam.get_shcoeffs(
                image = seg,
                lmax = 8,
                sigma = 1)

            # Compute reconstruction error
            mean_sq_error = aicsshtools.get_reconstruction_error(
                grid_input = grid_initial,
                grid_rec = grid)

            # Store spherical harmonic coefficients in dataframe by cell id
            df_coeffs = pd.DataFrame(coeffs, index=[CellId])
            df_coeffs.index = df_coeffs.index.rename('CellId')

            # Mesh reconstructed with the sh coefficients
            mesh_shparam = aicsshtools.get_reconstruction_from_grid(grid=grid)

            # Save meshes as VTK file
            aicsshtools.save_polydata(
                mesh = mesh_initial, 
                filename = str(
                    sh_data_dir / Path(f'{CellId}.initial.vtk'))
            )
            aicsshtools.save_polydata(
                mesh = mesh_shparam, 
                filename = str(
                    sh_data_dir / Path(f'{CellId}.shparam.vtk'))
            )

            # Save coeffs into a csv file in local staging
            df_coeffs.to_csv(
                str(sh_data_dir / Path(f'{CellId}.shparam.csv'))
            )

            # Build dataframe of saved files to store in manifest
            pdSerie = pd.Series(
                {
                    'InitialMeshFilePath': str(
                        sh_data_dir / Path(f'{CellId}.initial.vtk')
                    ),
                    'ShparamMeshFilePath': str(
                        sh_data_dir / Path(f'{CellId}.shparam.vtk')),
                    'CoeffsFilePath': str(sh_data_dir / Path(f'{CellId}.shparam.csv')),
                    'MeanSqError': mean_sq_error,
                    'CellId': CellId,
                }, name=CellId)
            self.manifest = self.manifest.append(pdSerie)

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
