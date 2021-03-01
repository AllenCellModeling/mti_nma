#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, List

import pandas as pd
from aicsshparam import shparam, shtools
from aicsimageio import AICSImage

from datastep import Step, log_run_params

import aics_dask_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class CellProcessResult(NamedTuple):
    cell_id: int
    data: Optional[Dict[str, Any]]

###############################################################################


class Shparam(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [],
        filepath_columns=[
            "InitialMeshFilePath", "ShparamMeshFilePath", "CoeffsFilePath"],
        **kwargs
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            **kwargs
        )

    @staticmethod
    def _process_cell(
        cell_id: str,
        cell_details: pd.Series,
        struct: str,
        lmax: int,
        save_dir: Path
    ) -> CellProcessResult:
        # Alert of which cell we are processing
        log.info(f"Beginning processing of cell: {cell_id}")

        # Read segmentation image
        impath = cell_details.SegFilePath
        seg = AICSImage(impath).get_image_data("ZYX", S=0, T=0, C=0)

        # Get spherical harmonic decomposition of segmentation
        (coeffs, grid_rec), (_, mesh_init, grid_init, transform) = shparam.get_shcoeffs(
            image=seg,
            lmax=lmax,
            sigma=1
        )

        # Compute reconstruction error
        mean_sq_error = shtools.get_reconstruction_error(
            grid_input=grid_init,
            grid_rec=grid_rec
        )

        # Store spherical harmonic coefficients in dataframe by cell id
        df_coeffs = pd.DataFrame(coeffs, index=[cell_id])
        df_coeffs.index = df_coeffs.index.rename("CellId")

        # Mesh reconstructed with the sh coefficients
        mesh_shparam = shtools.get_reconstruction_from_grid(grid=grid_rec)

        # Save meshes as PLY files compatible with both Blender and Paraview
        shtools.save_polydata(
            mesh=mesh_init,
            filename=str(save_dir / f"{cell_id}.initial_{struct}.ply")
        )
        shtools.save_polydata(
            mesh=mesh_shparam,
            filename=str(save_dir / f"{cell_id}.shparam_{struct}.ply")
        )

        # Save coeffs into a csv file in local staging
        df_coeffs.to_csv(
            str(save_dir / f"{cell_id}.shparam_{struct}.csv")
        )

        # Build dataframe of saved files to store in manifest
        data = {
            "InitialMeshFilePath":
                save_dir / f"{cell_id}.initial_{struct}.ply",
            "ShparamMeshFilePath":
                save_dir / f"{cell_id}.shparam_{struct}.ply",
            "CoeffsFilePath":
                save_dir / f"{cell_id}.shparam_{struct}.csv",
            "MeanSqError": mean_sq_error,
            "Structure": struct,
            "CellId": cell_id,
        }

        # Alert completed
        log.info(f"Completed processing for cell: {cell_id}")
        return CellProcessResult(cell_id, data)

    @log_run_params
    def run(
        self,
        sc_df=None,
        struct="Nuc",
        lmax=16,
        distributed_executor_address: Optional[str] = None,
        **kwargs
    ):
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

        lmax: int
            The lmax passed to spherical harmonic generation.

        distributed_executor_address: Optional[str]
            An optional distributed executor address to use for job distribution.
            Default: None (no distributed executor, use local threads)
        """

        # If no dataframe is passed in, load manifest from previous step
        if sc_df is None:
            sc_df = pd.read_csv(
                self.step_local_staging_dir.parent / "single_" 
                f"{struct}" / "manifest.csv", index_col='CellId'
            )

        # Create directory to save data for this step in local staging
        sh_data_dir = self.step_local_staging_dir / "shparam_data"
        sh_data_dir.mkdir(parents=True, exist_ok=True)

        # Get spherical harmonic set for segmentation, save and record in manifest
        self.manifest = pd.DataFrame([])

        # Process each cell in the dataframe
        with aics_dask_utils.DistributedHandler(distributed_executor_address) as handler:
            futures = handler.client.map(
                self._process_cell,
                # Convert dataframe iterrows into two lists of items to iterate over
                # One list will be the index (cell ids)
                # One list will be the pandas series of every row
                *zip(*list(sc_df.iterrows())),
                # Pass the other parameters as lists with the same length as the
                # dataframe but with the same value for every item in the list
                [struct for i in range(len(sc_df))],
                [lmax for i in range(len(sc_df))],
                [sh_data_dir for i in range(len(sc_df))]
            )

            # Block until all complete
            results = handler.gather(futures)

            # Set manifest with results
            for result in results:
                self.manifest = self.manifest.append(
                    pd.Series(result.data), ignore_index=True
                )

        self.manifest = self.manifest.set_index('CellId')
        self.manifest.index = self.manifest.index.astype(int)
                
        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / f"manifest.csv"
        )
        return self.manifest
