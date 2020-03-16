#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from ...utils.singlestruct_utils import SingleStruct
from datastep import log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class SingleCell(SingleStruct):

    @log_run_params
    def run(self, **kwargs):
        """
        Run the single structure data processing step of the pipeline
        for the cell membrane.

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
        cell_line_id: str
            Name of the cell line where nuclei are going to be sampled from
            AICS-13 = Lamin

        nsamples: int
            Number of cells to sample randomly (but with set seed) from dataset

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        Returns
        -------
        result: Dataframe
            Dataframe containing results of this step to be saved in the step manifest
        """
        self.manifest = self.run_singlestruct_step(**kwargs)
        return self.manifest
