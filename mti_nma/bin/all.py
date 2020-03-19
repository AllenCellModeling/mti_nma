#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.
When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging
from datetime import datetime
from pathlib import Path

from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
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
            steps.SingleNuc(),
            steps.ShparamNuc(),
            steps.AvgshapeNuc(),
            steps.NmaNuc(),
            steps.SingleCell(),
            steps.ShparamCell(),
            steps.AvgshapeCell(),
            steps.NmaCell()
        ]

    def run(
        self,
        distributed: bool = False,
        clean: bool = False,
        debug: bool = False,
        structs: list = ['Nuc'],
        **kwargs,
    ):
        """
        Run a flow with your steps.
        Parameters
        ----------
        distributed: bool
            A boolean option to determine if the jobs should be distributed to a remote
            cluster when possible.
            Default: False (Do not distribute)
        clean: bool
            Should the local staging directory be cleaned prior to this run.
            Default: False (Do not clean)
        debug: bool
            A debug flag for the developer to use to manipulate how much data runs,
            how it is processed, etc.
            Default: False (Do not debug)
        structs: List
            List of structure data to run pipeline on. Currently, only
            'Nuc' (nuclear membrane) and 'Cell' (cell membrane) are supported.

        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/
        Basic prefect example:
        https://docs.prefect.io/core/
        """
        # Initalize steps
        if "Nuc" in structs:
            single_nuc = steps.SingleNuc()
            shparam_nuc = steps.ShparamNuc()
            avgshape_nuc = steps.AvgshapeNuc()
            nma_nuc = steps.NmaNuc()

        if "Cell" in structs:
            single_cell = steps.SingleCell()
            shparam_cell = steps.ShparamCell()
            avgshape_cell = steps.AvgshapeCell()
            nma_cell = steps.NmaCell()

        # Choose executor
        if debug:
            exe = LocalExecutor()
            distributed_executor_address = None
        else:
            if distributed:
                # Create or get log dir
                # Do not include ms
                log_dir_name = datetime.now().isoformat().split(".")[0]
                log_dir = Path(f"~/.dask_logs/mti_nma/{log_dir_name}").expanduser()
                # Log dir settings
                log_dir.mkdir(parents=True)

                # Create cluster
                log.info("Creating SLURMCluster")
                cluster = SLURMCluster(
                    cores=2,
                    memory="24GB",
                    queue="aics_cpu_general",
                    walltime="10:00:00",
                    local_directory=str(log_dir),
                    log_directory=str(log_dir)
                )
                log.info("Created SLURMCluster")

                # Scale workers
                cluster.adapt(minimum_jobs=1, maximum_jobs=40)

                # Use the port from the created connector to set executor address
                distributed_executor_address = cluster.scheduler_address

                # Log dashboard URI
                log.info(f"Dask dashboard available at: {cluster.dashboard_link}")
            else:
                # Create local cluster
                log.info("Creating LocalCluster")
                cluster = LocalCluster()
                log.info("Created LocalCluster")

                # Set distributed_executor_address
                distributed_executor_address = cluster.scheduler_address

                # Log dashboard URI
                log.info(f"Dask dashboard available at: {cluster.dashboard_link}")

            # Use dask cluster
            exe = DaskExecutor(distributed_executor_address)

        try:
            # Configure your flow
            with Flow("mti_nma") as flow:
                # If your step utilizes dask pass the executor address
                # If you want to clean the local staging directories pass clean
                # If you want to utilize some debugging functionality pass debug
                # If you don't utilize any of these, just pass the parameters you need.

                if "Nuc" in structs:
                    struct = "Nuc"

                    sc_df = single_nuc(
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    sh_df = shparam_nuc(
                        sc_df=sc_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    avg_df = avgshape_nuc(
                        sh_df=sh_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    nma_nuc(
                        avg_df=avg_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )

                if "Cell" in structs:
                    struct = "Cell"

                    sc_df = single_cell(
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    sh_df = shparam_cell(
                        sc_df=sc_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    avg_df = avgshape_cell(
                        sh_df=sh_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    nma_cell(
                        avg_df=avg_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )

            # Run flow and get ending state
            flow.run(executor=exe)

            # If nucleus and cell membrane were anlyzed, draw comparison plot
            if "Nuc" and "Cell" in structs:
                draw_whist()

        # Catch any error and kill the remote dask cluster
        except Exception as err:
            log.error(f"Something went wrong during pipeline run: {err}")

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
