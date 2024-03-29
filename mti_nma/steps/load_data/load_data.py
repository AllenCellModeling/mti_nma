#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params
import os

from .load_data_tools import download_data

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class LoadData(Step):

    def __init__(
        self,
        direct_upstream_tasks: List["Step"] = [],
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks=direct_upstream_tasks, config=config)

    @log_run_params
    def run(
        self,
        struct="Nuc",
        nsamples=0,
        distributed_executor_address: Optional[str] = None,
        debug=False,
        **kwargs
    ):

        struct_dir = self.step_local_staging_dir / f"loaddata_{struct}"
        struct_dir.mkdir(parents=True, exist_ok=True)

        # Move init and run parameters to structure dir to avoid overwriting
        for filetype in ["init", "run"]:
            filename = f"{filetype}_parameters.json"
            os.rename(
                self.step_local_staging_dir / filename,
                struct_dir / filename
            )

        df = download_data(
            save_dir=struct_dir / "singlecell_data",
            nsamples=nsamples,
            struct=struct,
            distributed_executor_address=distributed_executor_address
        )

        self.manifest = df
        manifest_save_path = struct_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)

        return self.manifest
