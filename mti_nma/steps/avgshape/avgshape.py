#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
from aicsshparam import aicsshtools

from datastep import Step, log_run_params
from datastep.file_utils import manifest_filepaths_rel2abs

from ..shparam import Shparam
from .avgshape_utils import run_shcoeffs_analysis, save_mesh_as_stl

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Avgshape(Step):
    def __init__(
        self,
        clean_before_run=False,
        direct_upstream_tasks: Optional[List["Step"]] = Shparam,
        filepath_columns=["AvgShapeFilePath", "AvgShapeFilePathStl"],
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
        This step uses the amplitudes of the spherical harmonic components
        of the nuclear shapes in the dataset to construct an average nuclear mesh.
        """

        # Get shparam manifest
        shparam = Shparam()
        manifest_filepaths_rel2abs(shparam)
        df_sh = shparam.manifest.copy()
        df_sh = df_sh.set_index('CellId', drop=True)

        # Load sh coefficients of all samples in manifest
        df_coeffs = pd.DataFrame([])
        for CellId in df_sh.index:
            df_coeffs_path = df_sh['CoeffsFilePath'][CellId]
            df_coeffs = df_coeffs.append(
                pd.read_csv(df_coeffs_path, index_col=['CellId']), ignore_index=False
            )

        # Create directory to hold results from this step
        avg_data_dir = self.step_local_staging_dir / Path('avgshape_data')
        avg_data_dir.mkdir(parents=True, exist_ok=True)

        # Perform some per-cell analysis
        run_shcoeffs_analysis(df=df_coeffs, savedir=avg_data_dir)

        # Avg the sh coefficients over all samples and create avg mesh
        df_coeffs_avg = df_coeffs.agg(['mean'])
        mesh_avg = aicsshtools.get_reconstruction_from_dataframe(df=df_coeffs_avg)
        aicsshtools.save_polydata(
            mesh=mesh_avg,
            filename=str(avg_data_dir / Path('avgshape.vtk'))
        )

        # Save mesh as stl file for blender import
        save_mesh_as_stl(mesh_avg, str(avg_data_dir / Path('avgshape.stl')))

        # Save avg coeffs to csv file
        df_coeffs_avg.to_csv(
            str(avg_data_dir / Path('avgshape.csv'))
        )

        # Save path to avg shape in the manifest
        self.manifest = pd.DataFrame({
            'Label': 'Average_nuclear_mesh',
            'AvgShapeFilePath': str(avg_data_dir / Path('avgshape.vtk')),
            'AvgShapeFilePathStl': str(avg_data_dir / Path('avgshape.stl'))
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
