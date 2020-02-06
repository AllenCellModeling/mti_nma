#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import shutil
import numpy as np
import pandas as pd

from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Singlecell(Step):

    def __init__(
        self,
        clean_before_run=False,
        direct_upstream_tasks: Optional[List["Step"]] = None,
        filepath_columns=["RawFilePath", "SegFilePath"],
        config: Optional[Union[str, Path, Dict[str, str]]] = None
    ):
        super().__init__(
            clean_before_run=clean_before_run,
            direct_upstream_tasks=direct_upstream_tasks,
            filepath_columns=filepath_columns,
            config=config
        )

    @log_run_params
    def run(self, nsamples=16, **kwargs):
        """

        This function will gather the data for which we want to
        perform the normal mode analysis on. For now we will be
        working on a very small dataset as a proof of concept.
        The sample dataset is from the colony dataset that
        Matheus used in the ASCB-2019 talk.

        Data was imaged at 20X and tranfered to 100X for
        segmentation. The segmented images were downsampled to
        40X for tracking. Next we proceed with the single cell
        croping that creates single cell isotropic volumes with
        pixel size of 0.135um.

        Parameters
        ----------
        nsamples: int
            Number of cells to sample randomly (but with set seed) from dataset
        """

        # Set reproducible random seed
        np.random.seed(666)

        # Set local directory to access raw data
        raw_data_dir = Path('/Users/juliec/mti/raw_data')

        # Load data from local directory, drawing random samples from among the cells
        df = pd.read_csv(raw_data_dir / Path('ColonyMovie-ASCB2019.csv'), index_col=0)
        df = df[['CellId', 'crop_raw', 'crop_seg']].sample(n=nsamples)
        df = df.set_index('CellId')

        # Create directory to save data for this step in local staging
        sc_data_dir = self.step_local_staging_dir / Path('singlecell_data')
        sc_data_dir.mkdir(parents=True, exist_ok=True)

        # Copy files to self.step_local_staging_dir and record in manifest
        self.manifest = pd.DataFrame([])
        for CellId in df.index:
            shutil.copy(
                src=raw_data_dir / Path('raw.ome.tif'),
                dst=sc_data_dir / Path(f'{CellId}.raw.tif')
            )
            shutil.copy(
                src=raw_data_dir / Path('segmentation.ome.tif'),
                dst=sc_data_dir / Path(f'{CellId}.seg.tif')
            )
            pdSerie = pd.Series(
                {
                    'RawFilePath': str(sc_data_dir / Path(f'{CellId}.raw.tif')),
                    'SegFilePath': str(sc_data_dir / Path(f'{CellId}.seg.tif')),
                    'CellId': CellId,
                }, name=CellId)
            self.manifest = self.manifest.append(pdSerie)

        # Save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
