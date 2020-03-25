import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from aicsshparam import shtools

from datastep import Step
from ..shparam import Shparam
from .avgshape_utils import run_shcoeffs_analysis, save_mesh_as_stl

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Avgshape(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [Shparam],
        filepath_columns=["AvgShapeFilePath", "AvgShapeFilePathStl"],
        **kwargs

    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            **kwargs
        )

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
                self.step_local_staging_dir.parent / "shparam" / "manifest_"
                f"{struct}.csv"
            )

        # Fix filepaths and use cell id as dataframe index
        sh_df = sh_df.set_index("CellId", drop=True)

        # Load sh coefficients of all samples in manifest
        coeffs_df = pd.DataFrame([])
        for CellId in sh_df.index:
            coeffs_df_path = sh_df["CoeffsFilePath"][CellId]
            coeffs_df = coeffs_df.append(
                pd.read_csv(coeffs_df_path, index_col=["CellId"]), ignore_index=False
            )

        # Create directory to hold results from this step
        avg_data_dir = self.step_local_staging_dir / "avgshape_data"
        avg_data_dir.mkdir(parents=True, exist_ok=True)

        # Perform some per-cell analysis
        run_shcoeffs_analysis(df=coeffs_df, savedir=avg_data_dir, struct=struct)

        # Avg the sh coefficients over all samples and create avg mesh
        coeffs_df_avg = coeffs_df.agg(['mean'])
        coeffs_avg = coeffs_df_avg.values

        # Number of columns = 2*lmax*lmax
        lmax = int(np.sqrt(0.5 * coeffs_avg.size))

        coeffs_avg = coeffs_avg.reshape(-2, lmax, lmax)

        mesh_avg, _ = shtools.get_reconstruction_from_coeffs(coeffs=coeffs_avg)

        shtools.save_polydata(
            mesh=mesh_avg,
            filename=str(avg_data_dir / f"avgshape_{struct}.vtk")
        )

        # Save mesh as stl file for blender import
        save_mesh_as_stl(mesh_avg, str(avg_data_dir / f"avgshape_{struct}.stl"))

        # Save avg coeffs to csv file
        coeffs_df_avg.to_csv(
            str(avg_data_dir / f"avgshape_{struct}.csv")
        )

        # Save path to avg shape in the manifest
        self.manifest = pd.DataFrame({
            "Label": "Average_mesh",
            "AvgShapeFilePath": avg_data_dir / f"avgshape_{struct}.vtk",
            "AvgShapeFilePathStl": avg_data_dir / f"avgshape_{struct}.stl",
            "Structure": struct,
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path(f"manifest.csv"), index=False
        )
        return self.manifest
