#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datastep import Step, log_run_params
from .mesh_utils import polygon_mesh, mesh_from_models, volume_trimesh, model_verts, draw_mesh

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Mesh(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = None,
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks, config)

    @log_run_params
    def run(self, mesh_list=['default']):
        """
        Generate meshes to apply NMA to, saving each with a label and filepath.
        :param meshes: string indicating which meshes to generate, or default set
        """

        # generate default list of meshes to create if no input list given
        if 'default' in mesh_list:
            mesh_list = ['1D_3masses', '2D_triangle',  # include two prescribed models
                         '2D_polygon', '2D_polygon_x', # include a cross-connected and edge-connected polygon
                         '3D_sphere', '3D_sphere_S3']   # include two resolutions of marching cube spheres

        # create manifest to document generated/saved mesh vertices and faces
        N = len(mesh_list)
        col = ["label", "filepath"]
        self.manifest = pd.DataFrame(index=range(3 * N), columns=col)

        # create directory to hold meshes
        mesh_data_dir = self.step_local_staging_dir / Path("mesh_data")
        mesh_data_dir.mkdir(parents=True, exist_ok=True)

        # cycle through mesh labels and use them to generate mesh objects
        for i in range(N):
            mesh_label = mesh_list[i]
            if mesh_label in model_verts.keys():
                mesh = mesh_from_models(mesh_label)
            elif 'polygon' in mesh_label:
                mesh = polygon_mesh(mesh_label)
            elif 'sphere' in mesh_label:
                mesh = volume_trimesh(mesh_label)
            else:
                raise('No mesh generation found for this mesh label: ' + mesh_label)

            # create new directory for each mesh containing the mesh vertices and faces
            this_mesh_data_dir = mesh_data_dir / Path(mesh_label)
            this_mesh_data_dir.mkdir(parents=True, exist_ok=True)
            vert_path = this_mesh_data_dir / Path('verts.npy')
            face_path = this_mesh_data_dir / Path('faces.npy')
            np.save(vert_path, mesh.verts)
            np.save(face_path, mesh.faces)

            # add mesh vertices and faces to manifest
            self.manifest.at[3 * i, "filepath"] = vert_path
            self.manifest.at[3 * i, "label"] = mesh_label + '_verts'
            self.manifest.at[3 * i + 1, "filepath"] = face_path
            self.manifest.at[3 * i + 1, "label"] = mesh_label + '_faces'

            mesh_fig = draw_mesh(mesh)
            fig_path = this_mesh_data_dir / Path('fig.pdf')
            plt.savefig(fig_path, format='pdf')
            self.manifest.at[3 * i + 2, "filepath"] = fig_path
            self.manifest.at[3 * i + 2, "label"] = mesh_label + '_fig'

        # save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
