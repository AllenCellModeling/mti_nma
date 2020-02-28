#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

from aicsimageio import AICSImage
from aicsshparam import aicsshparam, aicsshtools

from datastep import Step, log_run_params

from ..singlecell import Singlecell

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Shparam(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [Singlecell],
        filepath_columns=[
            "InitialMeshFilePath", "ShparamMeshFilePath", "CoeffsFilePath"]
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns
        )

    @log_run_params
    def run(self, sc_df=None, struct="Nuc", lmax=8, **kwargs):

        """
        This function loads the seg images we want to perform sh parametrization on
        and calculate the sh coefficients. Results are saved as csv files.

        Parameters
        ----------
        sc_df: dataframe
            dataframe containing results from running Singlecell step
            See the construction of the manifest in singlecell.py for details

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).
        """

        # If no dataframe is passed in, load manifest from previous step
        if sc_df is None:
            sc_df = pd.read_csv(
                self.step_local_staging_dir.parent / "singlecell" / f"manifest_"
                f"{struct}.csv"
            )

        # Use cell ID as dataframe index
        sc_df = sc_df.set_index("CellId")

        # Create directory to save data for this step in local staging
        sh_data_dir = self.step_local_staging_dir / "shparam_data"
        sh_data_dir.mkdir(parents=True, exist_ok=True)

        # Get spherical harmonic set for segmentation, save and record in manifest
        self.manifest = pd.DataFrame([])

        for CellId in tqdm(sc_df.index):

            # Read segmentation image
            impath = sc_df[f"SegFilePath"][CellId]
            seg = AICSImage(impath).get_image_data("ZYX", S=0, T=0, C=0)

            # Get spherical harmonic decomposition of segmentation
            # Here is the place where I need someone taking a look at the
            # aicsshparam package to see what is the best way to return
            # the outputs
            (coeffs, grid), (_, mesh_init, _, grid_init) = aicsshparam.get_shcoeffs(
                image=seg,
                lmax=lmax,
                sigma=1)

            # Compute reconstruction error
            mean_sq_error = aicsshtools.get_reconstruction_error(
                grid_input=grid_init,
                grid_rec=grid)

            # Store spherical harmonic coefficients in dataframe by cell id
            df_coeffs = pd.DataFrame(coeffs, index=[CellId])
            df_coeffs.index = df_coeffs.index.rename("CellId")

            # Mesh reconstructed with the sh coefficients
            mesh_shparam = aicsshtools.get_reconstruction_from_grid(grid=grid)

            # Save meshes as VTK file
            aicsshtools.save_polydata(
                mesh=mesh_init, 
                filename=str(
                    sh_data_dir / f"{CellId}.initial_{struct}.vtk")
            )
            aicsshtools.save_polydata(
                mesh=mesh_shparam, 
                filename=str(
                    sh_data_dir / f"{CellId}.shparam_{struct}.vtk")
            )

            # Save coeffs into a csv file in local staging
            df_coeffs.to_csv(
                str(sh_data_dir / f"{CellId}.shparam_{struct}.csv")
            )

            # Build dataframe of saved files to store in manifest
            pdSerie = pd.Series(
                {
                    "InitialMeshFilePath": 
                        sh_data_dir / f"{CellId}.initial_{struct}.vtk",
                    "ShparamMeshFilePath":
                        sh_data_dir / f"{CellId}.shparam_{struct}.vtk",
                    "CoeffsFilePath":
                        sh_data_dir / f"{CellId}.shparam_{struct}.csv",
                    "MeanSqError": mean_sq_error,
                    "Structure": struct,
                    "CellId": CellId,

                }, name=CellId)
            self.manifest = self.manifest.append(pdSerie)

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / f"manifest_{struct}.csv", index=False
        )
        return self.manifest
