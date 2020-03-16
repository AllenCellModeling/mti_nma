#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List
from ..shparam_cell import ShparamCell
from ...utils.avgshape_utils import Avgshape
from datastep import log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class AvgshapeCell(Avgshape):
    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [ShparamCell],
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks)

    @log_run_params
    def run(self, **kwargs):
        """
        Run the shape-averaging step of the pipeline on cell data.

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
        sh_df: dataframe
            dataframe containing results from running Shparam step
            See the construction of the manifest in shparam.py for details

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        Returns
        -------
        result: Dataframe
            Dataframe containing results of this step to be saved in the step manifest
        """
        self.manifest = self.run_avgshape_step(**kwargs)
        return self.manifest
