#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List
from ..avgshape_nuc import AvgshapeNuc
from ...utils.nma_utils import Nma
from datastep import log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class NmaNuc(Nma):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [AvgshapeNuc],
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @log_run_params
    def run(self, **kwargs):
        """
        Run the normal mode analysis step of the pipeline on nucleus data.

        Protected Parameters
        --------------------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)

        Parameters
        ----------
        mode_list: list
            List of indices of modes to create heatmap files for.
            Defaults the six lowest energy modes (i.e. 0-5).

        avg_df: dataframe
            dataframe containing results from running Avgshape step
            See the construction of the manifest in avgshape.py for details

        path_blender: str
            Path to your local download of the Blender Application.
            If on Mac, the default Blender Mac download location is used.

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        Returns
        -------
        result: Dataframe
            Dataframe containing results of this step to be saved in the step manifest
        """
        self.manifest = self.run_nma_step(**kwargs)
        return self.manifest
