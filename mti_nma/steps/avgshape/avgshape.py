#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
from pathlib import Path
from typing import List, Optional
import pandas as pd
from aicsshparam import aicsshtools
import platform

from datastep import Step, log_run_params
from datastep.file_utils import manifest_filepaths_rel2abs

from ..shparam import Shparam
from .avgshape_utils import run_shcoeffs_analysis, save_mesh_as_stl, uniform_trimesh

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Avgshape(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [Shparam],
        filepath_columns=[
            "AvgShapeFilePath",
            "AvgShapeFilePathStl",
            "UniformMeshFilePathStl",
            "UniformMeshFilePathBlend",
            "UniformMeshVertices",
            "UniformMeshFaces"
        ]

    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns
        )

    @log_run_params
    def run(self, mesh_density=5, sh_df=None, path_blender=None, **kwargs):
        """
        This step uses the amplitudes of the spherical harmonic components
        of the nuclear shapes in the dataset to construct an average nuclear mesh.

        Parameters
        ----------
        mesh_density: int (1-10)
            Mesh density parameter used in Blender 
        """

        # fix filepaths and use cell id as dataframe index
        manifest_filepaths_rel2abs(sh_df)
        sh_df = sh_df.set_index("CellId", drop=True)

        # Load sh coefficients of all samples in manifest
        df_coeffs = pd.DataFrame([])
        for CellId in sh_df.index:
            df_coeffs_path = sh_df["CoeffsFilePath"][CellId]
            df_coeffs = df_coeffs.append(
                pd.read_csv(df_coeffs_path, index_col=["CellId"]), ignore_index=False
            )

        # Create directory to hold results from this step
        avg_data_dir = self.step_local_staging_dir / "avgshape_data"
        avg_data_dir.mkdir(parents=True, exist_ok=True)

        # Perform some per-cell analysis
        run_shcoeffs_analysis(df=df_coeffs, savedir=avg_data_dir)

        # Avg the sh coefficients over all samples and create avg mesh
        df_coeffs_avg = df_coeffs.agg(['mean'])
        coeffs_avg = df_coeffs_avg.values

        # Number of columns = 2*lmax*lmax
        lmax = int(np.sqrt(0.5*coeffs_avg.size))

        coeffs_avg = coeffs_avg.reshape(-2,lmax,lmax)

        mesh_avg, _ = aicsshtools.get_reconstruction_from_coeffs(coeffs=coeffs_avg)

        aicsshtools.save_polydata(
            mesh=mesh_avg,
            filename=str(avg_data_dir / "avgshape.vtk")
        )

        # Save mesh as stl file for blender import
        save_mesh_as_stl(mesh_avg, str(avg_data_dir / "avgshape.stl"))

        # Save avg coeffs to csv file
        df_coeffs_avg.to_csv(
            str(avg_data_dir / "avgshape.csv")
        )

        # Set the blender application download filepath
        if path_blender is None and platform == "darwin":
            path_blender = "/Applications/Blender.app/Contents/MacOS/Blender"
        else:
            raise NotImplementedError(
                "If using Linux you must pass in the path to your Blender download."
            )

        # Make new version of the mesh which is more uniform using Blender
        remesh_dir = avg_data_dir / "remesh"
        remesh_dir.mkdir(parents=True, exist_ok=True)
        path_input_mesh = str(avg_data_dir / "avgshape.stl")
        path_output = remesh_dir + "uniform_mesh"
        uniform_trimesh(path_input_mesh, mesh_density, path_output, path_blender)

        # Save path to avg shape in the manifest
        self.manifest = pd.DataFrame({
            "Label": "Average_nuclear_mesh",
            "AvgShapeFilePath": avg_data_dir / "avgshape.vtk",
            "AvgShapeFilePathStl": avg_data_dir / "avgshape.stl",
            "UniformMeshFilePathStl" : f"{path_output}.stl",
            "UniformMeshFilePathBlend" : f"{path_output}.blend",
            "UniformMeshVertices" : f"{path_output}_verts.npy",
            "UniformMeshFaces" : f"{path_output}_faces.npy"
        }, index=[0])

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
        return self.manifest
