#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
import uuid
from tqdm import tqdm
from aicsimageio import AICSImage, writers

from datastep import Step, log_run_params

from .singlecell_utils import query_data_from_labkey, crop_object

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Singlecell(Step):

    def __init__(
        self,
        filepath_columns=["RawFilePath", "SegFilePath"]
    ):
        super().__init__(
            filepath_columns=filepath_columns
        )

    @log_run_params
    def run(self, cell_line_id="AICS-13", nsamples=3, **kwargs):

        """
            This function will collect all FOVs of a particular cell
            line from LabKey and their corresponding nuclear segmentations.
            Results are returned as a dataframe where each row
            corresponds to one FOV.

            The dataframe is sampled according to the input parameter
            `nsamples`.

            Next we select at random one nucleus from each of the remaining
            FOVs to perform nma analysis on.

            Raw and segmented images of selected nuclei are cropped and resized
            to isotropic volume with pixel size 0.135um (compatitle with 40X).
            Images are also stored in the usual `ZYX` order.


        Parameters
        ----------
        cell_line_id: str
            Name of the cell line where nuclei are going to be sampled from
            AICS-13 = Lamin

        nsamples: int
            Number of cells to sample randomly (but with set seed) from dataset
        """

        np.random.seed(666)

        if nsamples > 0:

            self.manifest = pd.DataFrame([])

            # Load data from labkey and store in local dataframe
            log.info("Loading data from LabKey...")
            df = query_data_from_labkey(cell_line_id=cell_line_id)  # 13 = Lamin
            log.info(
                f"Number of FOVs available = {df.shape[0]}."
                f"Sampling {nsamples} FOVs now.")
            df = df.sample(n=nsamples)

            # create directory to save data for this step in local staging
            sc_data_dir = self.step_local_staging_dir / "singlecell_data"
            sc_data_dir.mkdir(parents=True, exist_ok=True)

            # Process each FOV in dataframe
            for fov_id in tqdm(df.index):

                sx = df.PixelScaleX[fov_id]
                sy = df.PixelScaleY[fov_id]
                sz = df.PixelScaleZ[fov_id]

                import pdb
                pdb.set_trace()

                # Use H3342 for nuclear channel
                ch_ind = AICSImage(
                    df.ReadPathRaw[fov_id]).get_channel_names().index('H3342')
                raw = AICSImage(
                    df.ReadPathRaw[fov_id]).get_image_data("ZYX", S=0, T=0, C=ch_ind)
                seg_nuc = AICSImage(
                    df.ReadPathSegNuc[fov_id]).get_image_data("ZYX", S=0, T=0, C=0)
                seg_cell = AICSImage(
                    df.ReadPathSegCell[fov_id]).get_image_data("ZYX", S=0, T=0, C=0)

                # Select one label from seg image at random
                obj_label = np.random.randint(low=1, high=1 + seg_nuc.max())

                # Center and crop raw and images to set size
                raw, seg_nuc, seg_cell = crop_object(
                    raw=raw,
                    seg_nuc=seg_nuc,
                    seg_cell=seg_cell,
                    obj_label=obj_label,
                    isotropic=(sx, sy, sz))

                # Only proceed with this cell if image isn't empty
                if raw is not None:

                    # Create an unique id for this object
                    cell_id = uuid.uuid4().hex[:8]

                    # Save images and write to manifest
                    rawpath = sc_data_dir.as_posix() + f"/{cell_id}.raw.tif"
                    with writers.OmeTiffWriter(rawpath) as writer:
                        writer.save(raw, dimension_order="ZYX")

                    segpath_nuc = sc_data_dir.as_posix() + f"/{cell_id}.seg_nuc.tif"
                    with writers.OmeTiffWriter(segpath_nuc) as writer:
                        writer.save(seg_nuc, dimension_order="ZYX")

                    segpath_cell = sc_data_dir.as_posix() + f"/{cell_id}.seg_cell.tif"
                    with writers.OmeTiffWriter(segpath_cell) as writer:
                        writer.save(seg_cell, dimension_order="ZYX")

                    series = pd.Series({
                        "RawFilePath": sc_data_dir / f"{cell_id}.raw.tif",
                        "SegNucFilePath": sc_data_dir / f"{cell_id}.seg_nuc.tif",
                        "SegCellFilePath": sc_data_dir / f"{cell_id}.seg_cell.tif",
                        "OriginalFOVPathRaw": df.ReadPathRaw[fov_id],
                        "OriginalFOVPathSegNuc": df.ReadPathSegNuc[fov_id],
                        "OriginalFOVPathSegCell": df.ReadPathSegCell[fov_id],
                        "FOVId": fov_id,
                        "CellId": cell_id}, name=cell_id)

                    self.manifest = self.manifest.append(series)
                else:
                    log.info("Rejected FOV: {fov_id} for empty images.")

            import pdb
            pdb.set_trace()

            # save manifest as csv
            self.manifest = self.manifest.astype({"FOVId": "int64"})
            self.manifest.to_csv(
                self.step_local_staging_dir / "manifest.csv", index=False
            )
            return self.manifest
