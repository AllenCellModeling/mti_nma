#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Singlecell(Step):
    def __init__(
        self,
        direct_upstream_tasks: Optional[List["Step"]] = None,
        config: Optional[Union[str, Path, Dict[str, str]]] = None,
    ):
        super().__init__(direct_upstream_tasks, config)

    @log_run_params
    def run(self, nsamples=16, **kwargs):

        '''
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
        '''

        print(self.step_local_staging_dir)

        import os
        import shutil
        import numpy as np
        np.random.seed(666)
        import pandas as pd

        df = pd.read_csv('/allen/aics/assay-dev/MicroscopyOtherData/Viana/forJulieTimelapseNucleus/NewColonyDataset/ColonyMovie-ASCB2019.csv', index_col=0)
        df = df[['CellId','crop_raw','crop_seg']].sample(n=nsamples) # Paths to raw and seg images
        df = df.set_index('CellId')

        # Copy files from original location to self.step_local_staging_dir

        self.manifest = pd.DataFrame([])

        for CellId in df.index:
            shutil.copy(
                src = df.crop_raw[CellId],
                dst = self.step_local_staging_dir / Path(f'{CellId}.raw.tif')
                )
            shutil.copy(
                src = df.crop_seg[CellId],
                dst = self.step_local_staging_dir / Path(f'{CellId}.seg.tif')
                )
            pdSerie = pd.Series(
                {
                    'RawFilePath': f'{CellId}.raw.tif',
                    'SegFilePath': f'{CellId}.seg.tif',
                    'CellId': CellId,
                }, name=CellId)
            self.manifest = self.manifest.append(pdSerie)

        # save manifest as csv
        self.manifest.to_csv(
            self.step_local_staging_dir / Path("manifest.csv"), index=False
        )
