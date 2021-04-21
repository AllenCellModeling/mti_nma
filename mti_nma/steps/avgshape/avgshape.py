import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from aicsshparam import shtools

from datastep import Step, log_run_params
from .avgshape_utils import run_shcoeffs_analysis, save_mesh_as_stl
from .avgshape_utils import save_mesh_as_obj, save_voxelization, save_displacement_map

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Avgshape(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [],
        filepath_columns=["AvgShapeFilePath", "AvgShapeFilePathStl"],
        **kwargs

    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            **kwargs
        )

    @log_run_params
    def run(self, sh_df=None, struct="Nuc", **kwargs):
        """
        This step uses the amplitudes of the spherical harmonic components
        of the nuclear shapes in the dataset to construct an average nuclear mesh.

        Parameters
        ----------
        sh_df: dataframe
            dataframe containing results from running Shparam step
            See the construction of the manifest in shparam.py for details

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).
        """

        # If no dataframe is passed in, load manifest from previous step
        if sh_df is None:
            sh_df = pd.read_csv(
                self.step_local_staging_dir.parent / "shparam" /
                f"shparam_{struct}" / "manifest.csv", index_col="CellId"
            )

        # Load sh coefficients of all samples in manifest
        coeffs_df = pd.DataFrame([])
        for CellId in sh_df.index:
            coeffs_df_path = sh_df["CoeffsFilePath"][CellId]
            coeffs_df = coeffs_df.append(
                pd.read_csv(coeffs_df_path, index_col=["CellId"]), ignore_index=False
            )

        # Create directory to hold results from this step
        struct_dir = self.step_local_staging_dir / f"avgshape_{struct}"
        struct_dir.mkdir(parents=True, exist_ok=True)

        avg_data_dir = struct_dir / f"avgshape_data"
        avg_data_dir.mkdir(parents=True, exist_ok=True)

        # Perform some per-cell analysis
        run_shcoeffs_analysis(df=coeffs_df, savedir=avg_data_dir, struct=struct)

        # Avg the sh coefficients over all samples and create avg mesh
        coeffs_df_avg = coeffs_df.agg(['mean'])
        coeffs_avg = coeffs_df_avg.values

        # Number of columns = 2*lmax*lmax
        lmax = int(np.sqrt(0.5 * coeffs_avg.size))

        coeffs_avg = coeffs_avg.reshape(-2, lmax, lmax)

        # Here we use the new meshing implementation for a more evenly distributed mesh
        '''
        mesh_avg, _ = shtools.get_even_reconstruction_from_coeffs(
            coeffs=coeffs_avg,
            npoints=1024
        )
        '''

        mesh_avg, grid_avg = shtools.get_reconstruction_from_coeffs(
            coeffs=coeffs_avg,
        )

        shtools.save_polydata(
            mesh=mesh_avg,
            filename=str(avg_data_dir / f"avgshape_{struct}.ply")
        )

        # Save mesh as obj
        save_mesh_as_obj(mesh_avg, avg_data_dir / f"avgshape_{struct}.obj")

        # Save displacement map
        save_displacement_map(grid_avg, avg_data_dir / f"avgshape_dmap_{struct}.tif")

        # Save mesh as image
        save_voxelization(mesh_avg, avg_data_dir / f"avgshape_{struct}.tif")

        # Save mesh as stl file for blender import
        save_mesh_as_stl(mesh_avg, avg_data_dir / f"avgshape_{struct}.stl")

        # Save avg coeffs to csv file
        coeffs_df_avg.to_csv(
            str(avg_data_dir / f"avgshape_{struct}.csv")
        )

        # Save path to avg shape in the manifest
        self.manifest = pd.DataFrame({
            "Label": "Average_mesh",
            "AvgShapeFilePath": avg_data_dir / f"avgshape_{struct}.ply",
            "AvgShapeFilePathStl": avg_data_dir / f"avgshape_{struct}.stl",
            "AvgShapeFilePathObj": avg_data_dir / f"avgshape_{struct}.obj",
            "AvgShapeFilePathTif": avg_data_dir / f"avgshape_{struct}.tif",
            "AvgShapeDMapFilePathTif": avg_data_dir / f"avgshape_dmap_{struct}.tif",
            "Structure": struct,
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            struct_dir / Path(f"manifest.csv"), index=False
        )
        return self.manifest
