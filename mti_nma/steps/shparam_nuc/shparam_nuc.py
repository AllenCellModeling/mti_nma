#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Optional

from datastep import Step, log_run_params

from ...utils.shparam_utils import Shparam
from ..single_nuc import SingleNuc

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ShparamNuc(Step, Shparam):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = [SingleNuc]
    ):
        super().__init__(
            direct_upstream_tasks=direct_upstream_tasks
        )

    @log_run_params
    def run(self, **kwargs):
        """
        Run the shparam step of the pipeline on nucleus data.

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
        sc_df: dataframe
            dataframe containing results from running Singlecell step
            See the construction of the manifest in singlecell.py for details

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        lmax: int
            The lmax passed to spherical harmonic generation.


        Returns
        -------
        result: Dataframe
            Dataframe containing results of this step to be saved in the step manifest
        """
        self.manifest = self.run_shparam_step(**kwargs)
        return self.manifest
