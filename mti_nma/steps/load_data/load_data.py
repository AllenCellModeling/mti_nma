#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params

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
        nsamples=0,
        debug=False,
        distributed_executor_address: Optional[str]=None,
        **kwargs
    ):
                
        df = download_data(
            save_dir=self.step_local_staging_dir / "singlecell_data",
            nsamples=nsamples,
            distributed_executor_address=distributed_executor_address
        )
        
        self.manifest = df
        manifest_save_path = self.step_local_staging_dir / "manifest.csv"
        self.manifest.to_csv(manifest_save_path)
            
        return manifest_save_path