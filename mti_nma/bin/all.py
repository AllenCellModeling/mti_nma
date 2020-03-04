#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run all tasks in a prefect Flow.
When you add steps to you step workflow be sure to add them to the step list
and configure their IO in the `run` function.
"""

import logging
from getpass import getuser
from pathlib import Path

from distributed import LocalCluster
from prefect import Flow
from prefect.engine.executors import DaskExecutor, LocalExecutor
from scheduler_tools import Connector

from mti_nma import steps

from .compare_nuc_cell import draw_whist

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

REMOTE_DASK_PREFS = {
    "cluster_obj_name": {},
    "cluster_conf": {},
    "worker_conf": {},
    "remote_conf": {},
}

REMOTE_DASK_PREFS["cluster_obj_name"] = {
    "module": "dask_jobqueue",
    "object": "SLURMCluster",
}

REMOTE_DASK_PREFS["cluster_conf"]["queue"] = "aics_cpu_general"
REMOTE_DASK_PREFS["cluster_conf"]["cores"] = 2
REMOTE_DASK_PREFS["cluster_conf"]["memory"] = "4GB"
REMOTE_DASK_PREFS["cluster_conf"]["walltime"] = "240:00:00"
REMOTE_DASK_PREFS["worker_conf"]["minimum_jobs"] = 1
REMOTE_DASK_PREFS["worker_conf"]["maximum_jobs"] = 40
REMOTE_DASK_PREFS["remote_conf"]["env"] = "mti_nma"
REMOTE_DASK_PREFS["remote_conf"]["command"] = "setup_and_spawn.bash"
REMOTE_DASK_PREFS["remote_conf"]["path"] = f"/home/{getuser()}/.slurm_dask_cpu"


REMOTE_SSH_PREFS = {
    "localfolder": str(Path("~/.aics_dask").expanduser()),
    "gateway": {
        "url": "slurm-master",
        "user": getuser(),
        "identityfile": str(Path("~/.ssh/id_rsa").expanduser())
    },
    "dask_port": 34000,
    "dashboard_port": 8787
}

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
        distributed: bool = False,
        clean: bool = False,
        debug: bool = False,
        cell_flag: bool = False,
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
            if distributed:
                # Create connection to remote cluster
                conn = Connector(
                    dask_prefs=REMOTE_DASK_PREFS,
                    prefs=REMOTE_SSH_PREFS
                )

                # Start the connection
                conn.run_command()
                conn.stop_forward_if_running()
                conn.forward_ports()

                # Use the port from the created connector to set executor address
                distributed_executor_address = f"tcp://localhost:{conn.local_dask_port}"

                # Log dashboard URI
                log.info(f"Dask dashboard available at: {conn.local_dashboard_port}")
            else:
                # Create local cluster
                cluster = LocalCluster()

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

            if cell_flag:
                draw_whist()

            # Stop the cluster if distributed
            if distributed:
                conn.stop_dask()

        # Catch any error and kill the remote dask cluster
        except Exception:
            if distributed:
                conn.stop_dask()

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
