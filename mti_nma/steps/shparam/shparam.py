#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params

from ..singlecell import Singlecell

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Shparam(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [Singlecell],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks, config)

    @log_run_params
    def run(self, **kwargs):

        '''
            Get singlecell manifest
        '''

        singlecells = Singlecell()
        df = singlecells.manifest.copy()
        df = df.set_index('CellId')

        '''
            This function loads the seg images we want to perform sh parametrization on
            and calculate the sh coefficients. Results are saved as csv files.
        '''

        import os
        import pandas as pd
        from skimage import io as skio
        from aicsshparam import aicsshparam, aicsshtools

        self.manifest = pd.DataFrame([])

        for CellId in df.index:

            impath = str(self.step_local_staging_dir / Path(f'../singlecell/{df.SegFilePath[CellId]}'))
            seg = skio.imread(impath)

            df_coeffs, extras = aicsshparam.get_shcoeffs(
                seg = seg,
                params = {'sigma': 1, 'lmax': 8})

            mesh_initial = extras[1]

            df_coeffs = pd.DataFrame(df_coeffs, index=[CellId])
            df_coeffs.index = df_coeffs.index.rename('CellId')

            # Mesh reconstructed with the sh coefficients

            mesh_shparam = aicsshtools.get_reconstruction_from_dataframe(df=df_coeffs)

            # Save meshes as VTK file

            aicsshtools.save_polydata(mesh = mesh_initial,
                filename = str(self.step_local_staging_dir / Path(f'{CellId}.initial.vtk')))
            aicsshtools.save_polydata(mesh = mesh_shparam,
                filename = str(self.step_local_staging_dir / Path(f'{CellId}.shparam.vtk')))
            
            # Save coeffs into a csv file

            df_coeffs.to_csv(
                str(self.step_local_staging_dir / Path(f'{CellId}.shparam.csv'))
            )

            pdSerie = pd.Series(
                {
                    'InitialMeshFilePath': f'{CellId}.initial.vtk',
                    'ShparamMeshFilePath': f'{CellId}.shparam.vtk',
                    'CoeffsFilePath': f'{CellId}.shparam.csv',
                    'CellId': CellId,
                }, name=CellId)
            
            self.manifest = self.manifest.append(pdSerie)

        # save manifest as csv

        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )


