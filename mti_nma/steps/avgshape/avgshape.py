#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params

from ..shparam import Shparam

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

def run_shcoeffs_analysis(self, df):

    import matplotlib.pyplot as plt

    list_of_scatter_plots = [
        ('shcoeffs_L0M0C','shcoeffs_L2M0C'),
        ('shcoeffs_L0M0C','shcoeffs_L2M2C'),
        ('shcoeffs_L0M0C','shcoeffs_L2M1S'),
        ('shcoeffs_L0M0C','shcoeffs_L2M1C'),
    ]

    for id_plot, (varx,vary) in enumerate(list_of_scatter_plots):

        fs = 18
        fig, ax = plt.subplots(1,1,figsize=(6,4))
        ax.plot(df[varx],df[vary],'o')
        ax.set_xlabel(varx, fontsize=fs)
        ax.set_ylabel(vary, fontsize=fs)
        plt.tight_layout()
        fig.savefig(
            str(self.step_local_staging_dir / Path(f'scatter-{id_plot}.svg'))
        )
        plt.close(fig)

    list_of_bar_plots = [
        'shcoeffs_chi2',
        'shcoeffs_L0M0C',
    ]

    for id_plot, var in enumerate(list_of_bar_plots):

        fs = 18
        fig, ax = plt.subplots(1,1,figsize=(12,8))
        ax.bar(df.index, df[var])
        ax.set_xlabel('CellId', fontsize=10)
        ax.set_ylabel(var, fontsize=fs)
        ax.tick_params('x', labelrotation=90)
        plt.tight_layout()
        fig.savefig(
            str(self.step_local_staging_dir / Path(f'bar-{id_plot}.svg'))
        )
        plt.close(fig)


class Avgshape(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [Shparam],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks, config)

    @log_run_params
    def run(self, **kwargs):

        # Get shparam manifest

        shparam = Shparam()
        df = shparam.manifest.copy()
        df = df.set_index('CellId', drop=True)

        # Load sh coefficients of all samples in manifest

        import pandas as pd
        from aicsshparam import aicsshtools

        df_coeffs = pd.DataFrame([])

        for CellId in df.index:
            df_coeffs_path = str(self.step_local_staging_dir / Path(f'../shparam/{df.CoeffsFilePath[CellId]}'))
            df_coeffs = df_coeffs.append(
                pd.read_csv(df_coeffs_path, index_col=['CellId']), ignore_index=False
            )

        # At this point we are able to perform some per cell analysis

        run_shcoeffs_analysis(self,df=df_coeffs)

        # Avg the sh coefficients over all samples

        df_coeffs_avg = df_coeffs.agg(['mean'])

        mesh_avg = aicsshtools.get_reconstruction_from_dataframe(df=df_coeffs_avg)

        aicsshtools.save_polydata(mesh = mesh_avg,
            filename = str(self.step_local_staging_dir / Path('avgshape.vtk')))

        # Save avg coeffs to csv file

        df_coeffs_avg.to_csv(
            str(self.step_local_staging_dir / Path('avgshape.csv'))
        )
           
        # Save path to avg shape in the manifest

        self.manifest = pd.DataFrame([])

        pdSerie = pd.Series(
            {
                'AvgShapeFilePath': 'avgshape.vtk'
            })
            
        self.manifest = self.manifest.append(pdSerie, ignore_index=True)

        # save manifest as csv

        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )


