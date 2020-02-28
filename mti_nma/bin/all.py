#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.
When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging
from typing import Optional
import os

from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor

from mti_nma import steps
from .compare_nuc_cell import draw_whist

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not running.
        """
        self.step_list = [
            steps.Singlecell(),
            steps.Shparam(),
            steps.Avgshape(),
            steps.Nma(),
        ]

    def run(
        self,
        distributed_executor_address: Optional[str] = None,
        clean: bool = False,
        debug: bool = False,
        cell_flag: bool = False,
        **kwargs,
    ):
        """
        Run a flow with your steps.
        Parameters
        ----------
        distributed_executor_address: Optional[str]
            An optional executor address to pass to some computation engine.
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)
        cell_flag: bool
            Flag for wether to include cell membrane in analysis. The nucleus is always
            analyzed, and this flag allow you to either additionally analyze the cell
            membrane (True) or not (False).
        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/
        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Initalize steps
        singlecell = steps.Singlecell()
        shparam = steps.Shparam()
        avgshape = steps.Avgshape()
        nma = steps.Nma()

        # Choose executor
        if debug:
            exe = LocalExecutor()
        else:
            if distributed_executor_address is not None:
                exe = DaskExecutor(distributed_executor_address)
            else:
                # Stop conflicts between Dask and OpenBLAS
                # Info here:
                # https://stackoverflow.com/questions/45086246/too-many-memory-regions-error-with-dask
                os.environ["OMP_NUM_THREADS"] = "1"

                # Start local dask cluster
                exe = DaskExecutor()

        # Configure your flow
        with Flow("mti_nma") as flow:
            # If your step utilizes a secondary flow with dask pass the executor address
            # If you want to clean the local staging directories pass clean
            # If you want to utilize some debugging functionality pass debug
            # If you don't utilize any of these, just pass the parameters you need.

            if cell_flag:
                structs = ["Nuc", "Cell"]
            else:
                structs = ["Nuc"]

            for struct in structs:

                sc_df = singlecell(
                    distributed_executor_address=distributed_executor_address,
                    clean=clean,
                    debug=debug,
                    struct=struct,
                    **kwargs
                )
                sh_df = shparam(
                    sc_df=sc_df,
                    distributed_executor_address=distributed_executor_address,
                    clean=clean,
                    debug=debug,
                    struct=struct,
                    **kwargs
                )
                avg_df = avgshape(
                    sh_df=sh_df,
                    distributed_executor_address=distributed_executor_address,
                    clean=clean,
                    debug=debug,
                    struct=struct,
                    **kwargs
                )
                nma(
                    avg_df=avg_df,
                    distributed_executor_address=distributed_executor_address,
                    clean=clean,
                    debug=debug,
                    struct=struct,
                    **kwargs
                )

        # Run flow and get ending state
        flow.run(executor=exe)
        # state = flow.run(executor=exe)

        if cell_flag:
            draw_whist()

    def pull(self):
        """
        Pull all steps.
        """
        for step in self.step_list:
            step.pull()

    def checkout(self):
        """
        Checkout all steps.
        """
        for step in self.step_list:
            step.checkout()

    def push(self):
        """
        Push all steps.
        """
        for step in self.step_list:
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()
