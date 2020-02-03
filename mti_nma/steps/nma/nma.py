#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from datastep import Step, log_run_params
from datastep.file_utils import manifest_filepaths_rel2abs

from ..mesh import Mesh
from .nma_utils import run_nma

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

class Nma(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = ["Mesh"],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks, config)

    @log_run_params
    def run(self, mesh_list=['default']):

        # set default list of meshes to use
        if 'default' in mesh_list:
            mesh_list = ['1D_3masses', '2D_triangle',  # include two prescribed models
                         '2D_polygon', '2D_polygon_x', # include a cross-connected and edge-connected polygon
                         '3D_sphere', '3D_sphere_S3']   # include two resolutions of marching cube spheres

        # make new manifest for nma step
        N = len(mesh_list)
        col = ["label", "filepath"]
        self.manifest = pd.DataFrame(index=range(2 * N), columns=col)

        # get mesh manifest
        meshes = Mesh()
        manifest_filepaths_rel2abs(meshes)
        mesh_df = meshes.manifest.copy()

        # create directory to hold meshes
        nma_data_dir = self.step_local_staging_dir / Path("nma_data")
        nma_data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(N):
            mesh_verts = np.load(mesh_df[mesh_df['label'] == mesh_list[i]+'_verts']['filepath'].iloc[0])
            mesh_faces = np.load(mesh_df[mesh_df['label'] == mesh_list[i]+'_faces']['filepath'].iloc[0])
            w, v = run_nma(mesh_verts, mesh_faces)

            # create new directory for each mesh containing the mesh vertices and faces
            this_nma_data_dir = nma_data_dir / Path(mesh_list[i])
            this_nma_data_dir.mkdir(parents=True, exist_ok=True)
            w_path = this_nma_data_dir / Path('eigvals.npy')
            v_path = this_nma_data_dir / Path('eigvecs.npy')
            np.save(w_path, w)
            np.save(v_path, v)

            # add mesh vertices and faces to manifest
            self.manifest.at[2 * i, "filepath"] = w_path
            self.manifest.at[2 * i, "label"] = mesh_list[i] + '_eigvals'
            self.manifest.at[2 * i + 1, "filepath"] = v_path
            self.manifest.at[2 * i + 1, "label"] = mesh_list[i] + '_eigvecs'


        # save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )