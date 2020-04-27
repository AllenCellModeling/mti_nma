#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

import numpy as np
import pandas as pd

from aicsimageio import AICSImage, writers
from datastep import Step, log_run_params

import aics_dask_utils
from .singlecell_utils import crop_object, query_data_from_labkey

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class FOVProcessResult(NamedTuple):
    fov_id: int
    cell_id: int
    data: Optional[Dict[str, Any]]

###############################################################################


class Singlecell(Step):

    def __init__(
        self,
        filepath_columns=[
            "RawFilePath",
            "SegFilePath"],
        **kwargs
    ):
        super().__init__(
            filepath_columns=filepath_columns, **kwargs
        )

    @staticmethod
    def _process_fov(
        fov_id: int,
        fov_details: pd.Series,
        struct: str,
        save_dir: Path
    ) -> FOVProcessResult:
        # Alert of which FOV we are processing
        log.info(f"Beginning processing for FOV: {fov_id}")

        # Get pixel scales out
        sx = fov_details.PixelScaleX
        sy = fov_details.PixelScaleY
        sz = fov_details.PixelScaleZ

        # Set channel numbers for this structure
        if struct == "Nuc":
            ch = 405
            full = "Nucleus"
        elif struct == "Cell":
            ch = 638
            full = "Membrane"
        else:
            raise(f"Analysis of structure {struct} is not currently supported."
                  "Please pass Nuc or Cell for the struct paramter.")

        # Get structure raw and seg images
        raw = AICSImage(fov_details.SourceReadPath).get_image_data(
            "ZYX", S=0, T=0, C=fov_details[f"ChannelNumber{ch}"]
        )

        seg = AICSImage(fov_details[f"{full}SegmentationReadPath"]).get_image_data(
            "ZYX", S=0, T=0, C=0
        )

        # Select one label from seg image at random
        obj_label = np.random.randint(low=1, high=1 + seg.max())

        # Center and crop raw and images to set size
        raw, seg, = crop_object(
            raw=raw,
            seg=seg,
            obj_label=obj_label,
            isotropic=(sx, sy, sz))

        # Only proceed with this cell if image isn't empty
        if raw is not None:
            # Create an unique id for this object
            cell_id = uuid.uuid4().hex[:8]

            # Save images and write to manifest
            raw_path = save_dir.as_posix() + f"/{cell_id}.raw_{struct}.tif"
            with writers.OmeTiffWriter(raw_path) as writer:
                writer.save(raw, dimension_order="ZYX")

            seg_path = save_dir.as_posix() + f"/{cell_id}.seg_{struct}.tif"
            with writers.OmeTiffWriter(seg_path) as writer:
                writer.save(seg, dimension_order="ZYX")

            data = {
                "RawFilePath": raw_path,
                "SegFilePath": seg_path,
                "OrigFOVPathRaw": fov_details.SourceReadPath,
                "OrigFOVPathSeg": fov_details[f"{full}SegmentationReadPath"],
                "Structure": struct,
                "FOVId": fov_id,
                "CellId": cell_id}

            result = FOVProcessResult(fov_id, cell_id, data)

        # Return None result as indication of rejected FOV
        else:
            result = FOVProcessResult(fov_id, None, None)

        # Alert completed
        log.info(f"Completed processing for FOV: {fov_id}")
        return result

    @log_run_params
    def run(
        self,
        cell_line_id="AICS-13",
        nsamples=3,
        struct="Nuc",
        distributed_executor_address: Optional[str] = None,
        **kwargs
    ):
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

        struct: str
            String giving name of structure to run analysis on.
            Currently, this must be "Nuc" (nucleus) or "Cell" (cell membrane).

        distributed_executor_address: Optional[str]
            An optional distributed executor address to use for job distribution.
            Default: None (no distributed executor, use local threads)
        """
        np.random.seed(666)

        if nsamples > 0:
            # Create manifest and directory to save data for this step in local staging
            self.manifest = pd.DataFrame([])
            sc_data_dir = self.step_local_staging_dir / "singlecell_data"
            sc_data_dir.mkdir(parents=True, exist_ok=True)

            # Load data from labkey and store in local dataframe
            log.info("Loading data from LabKey...")
            df = query_data_from_labkey(cell_line_id=cell_line_id)  # 13 = Lamin
            log.info(
                f"{len(df)} FOVs available. "
                f"Sampling {nsamples} FOVs.")
            df = df.sample(n=nsamples)

            # Process each FOV in dataframe
            with aics_dask_utils.DistributedHandler(distributed_executor_address) as handler:
                futures = handler.client.map(
                    self._process_fov,
                    # Convert dataframe iterrows into two lists of items to iterate over
                    # One list will be fov_ids
                    # One list will be the pandas series of every row
                    *zip(*list(df.iterrows())),
                    # Pass the other parameters as list of the same thing for each
                    # mapped function call
                    [struct for i in range(len(df))],
                    [sc_data_dir for i in range(len(df))]
                )

                # Block until all complete
                results = handler.gather(futures)

                # Set manifest with results
                for result in results:
                    if result.data is not None:
                        self.manifest = self.manifest.append(
                            pd.Series(result.data, name=result.cell_id)
                        )
                    else:
                        log.info(f"Rejected FOV: {result.fov_id} for empty images.")

            # Save manifest as csv
            self.manifest = self.manifest.astype({"FOVId": "int64"})
            self.manifest.to_csv(
                self.step_local_staging_dir / f"manifest.csv", index=False
            )
            return self.manifest
