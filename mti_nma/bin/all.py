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

import dask
from dask_jobqueue import SLURMCluster
from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor

from mti_nma import steps

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class All:
    def __init__(self):
        """
        Set all of your available steps here.
        This is only used for data logging operations, not running.
        """
        step_list = [
            [
                steps.LoadData(),
                steps.Shparam(step_name=f"shparam_{name}"),
                steps.Avgshape(step_name=f"avgshape_{name}"),
                steps.Nma(step_name=f"nma_{name}")
            ]
            for name in ["nuc"]
        ]
        self.step_list = [step for sublist in step_list for step in sublist]
        self.step_list.append(steps.CompareNucCell())

    def run(
        self,
        distributed: bool = False,
        clean: bool = False,
        debug: bool = False,
        structs: list = ["Nuc"],
        flow_viz: bool = False,
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
        flow_viz: bool
            Make flow chart to visualize pipeline - requires conda install of graphviz.

        Notes
        -----
        Documentation on prefect:
        https://docs.prefect.io/core/
        Basic prefect example:
        https://docs.prefect.io/core/
        """

        # Initalize steps
        if "Nuc" in structs:
            loaddata_nuc = steps.LoadData()
            shparam_nuc = steps.Shparam(step_name="shparam_nuc")
            avgshape_nuc = steps.Avgshape(step_name="avgshape_nuc")
            nma_nuc = steps.Nma(step_name="nma_nuc")

        if "Cell" in structs:
            single_cell = steps.Singlecell(step_name="single_cell")
            shparam_cell = steps.Shparam(step_name="shparam_cell")
            avgshape_cell = steps.Avgshape(step_name="avgshape_cell")
            nma_cell = steps.Nma(step_name="nma_cell")

        if "Nuc" in structs and "Cell" in structs:
            compare_nuc_cell = steps.CompareNucCell()

        # Choose executor
        if debug:
            exe = LocalExecutor()
            distributed_executor_address = None
            log.info(f"Debug flagged. Will use threads instead of Dask.")
        else:
            if distributed:
                # Create or get log dir
                # Do not include ms
                log_dir_name = datetime.now().isoformat().split(".")[0]
                log_dir = Path(f".dask_logs/{log_dir_name}").expanduser()
                # Log dir settings
                log_dir.mkdir(parents=True, exist_ok=True)

                # Configure dask config
                dask.config.set(
                    {
                        "scheduler.work-stealing": False,
                        "logging.distributed.worker": "info",
                    }
                )

                # Create cluster
                log.info("Creating SLURMCluster")
                cluster = SLURMCluster(
                    cores=4,
                    memory="20GB",
                    queue="aics_cpu_general",
                    walltime="10:00:00",
                    local_directory=str(log_dir),
                    log_directory=str(log_dir),
                )
                log.info("Created SLURMCluster")

                # Scale cluster
                cluster.scale(60)

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

                    ld_nuc_df = loaddata_nuc(
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    sh_nuc_df = shparam_nuc(
                        sc_df=ld_nuc_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    avg_nuc_df = avgshape_nuc(
                        sh_df=sh_nuc_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    nma_nuc_df = nma_nuc(
                        avg_df=avg_nuc_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )

                if "Cell" in structs:
                    struct = "Cell"

                    sc_cell_df = single_cell(
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    sh_cell_df = shparam_cell(
                        sc_df=sc_cell_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    avg_cell_df = avgshape_cell(
                        sh_df=sh_cell_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )
                    nma_cell_df = nma_cell(
                        avg_df=avg_cell_df,
                        distributed_executor_address=distributed_executor_address,
                        clean=clean,
                        debug=debug,
                        struct=struct,
                        **kwargs
                    )

                # If nucleus and cell membrane were anlyzed, draw comparison plot
                if "Nuc" in structs and "Cell" in structs:
                    compare_nuc_cell(nma_nuc_df, nma_cell_df)

            # Run flow, get ending state, and visualize pipeline
            flow.run(executor=exe)

            # Create pipeline visualization if flag is True
            # Note:
            # Flag False by default as a required package is not pip-installable
            # To use this feature, first `conda install graphviz`
            if flow_viz:
                flow.visualize()

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
            log.info(f"Pushing data for step: {step.step_name}")
            step.push()

    def clean(self):
        """
        Clean all steps.
        """
        for step in self.step_list:
            step.clean()
