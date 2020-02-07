#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from datastep import Step, log_run_params

from .singlecell_utils import query_data_from_labkey, crop_object

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

        import uuid
        import numpy as np
        np.random.seed(666)
        import pandas as pd
        from tqdm import tqdm
        from pathlib import Path
        from aicsimageio import AICSImage, writers

        if nsamples > 0:

            self.manifest = pd.DataFrame([])

            print("Loading data from LabKey...")

            df = query_data_from_labkey(CellLineId='AICS-13') # 13 = Lamin

            print(f"Number of FOVs available = {df.shape[0]}. Sampling {nsamples} FOVs now.")

            df = df.sample(n=nsamples)

            for FOVId in tqdm(df.index):

                sx = df.PixelScaleX[FOVId]
                sy = df.PixelScaleY[FOVId]
                sz = df.PixelScaleZ[FOVId]

                # C=-2 may not return the DNA channel for other cell lines. Need validation.
                raw = AICSImage(df.ReadPathRaw[FOVId]).get_image_data('ZYX', S=0, T=0, C=-2)
                seg = AICSImage(df.ReadPathSeg[FOVId]).get_image_data('ZYX', S=0, T=0, C= 0)

                # Select one label from seg image at random
                obj_label = np.random.randint(low=1, high=1+seg.max())

                raw, seg = crop_object(
                    raw = raw,
                    seg = seg,
                    obj_label = obj_label,
                    isotropic = (sx,sy,sz))

                # Create an unique id for this object
                CellId = uuid.uuid4().hex[:8]

                # Save images
                writer = writers.OmeTiffWriter(self.step_local_staging_dir.as_posix() + f'/{CellId}.raw.tif')
                writer.save(raw)

                writer = writers.OmeTiffWriter(self.step_local_staging_dir.as_posix() + f'/{CellId}.seg.tif')
                writer.save(seg)

                pdSerie = pd.Series({
                        'RawFilePath': f'{CellId}.raw.tif',
                        'SegFilePath': f'{CellId}.seg.tif',
                        'OriginalFOVPathRaw': df.ReadPathRaw[FOVId],
                        'OriginalFOVPathSeg': df.ReadPathSeg[FOVId],
                        'FOVId': FOVId,
                        'CellId': CellId}, name=CellId)

                self.manifest = self.manifest.append(pdSerie)

            # save manifest as csv

            self.manifest = self.manifest.astype({'FOVId': 'int64'})

            self.manifest.to_csv(
                self.step_local_staging_dir / Path("manifest.csv"), index=False
            )

