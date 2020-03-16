#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import platform

import numpy as np
import pandas as pd
from lkaccess import LabKey, contexts
from skimage import transform as sktrans

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional

from aicsimageio import AICSImage, writers
from datastep import Step, log_run_params
from . import dask_utils

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class FOVProcessResult(NamedTuple):
    fov_id: int
    cell_id: int
    data: Optional[Dict[str, Any]]

###############################################################################


class SingleStruct(Step):

    def __init__(
        self,
        filepath_columns=[
            "RawFilePath",
            "SegFilePath",
            "OrigFOVPathRaw",
            "OrigFOVPathSeg"]
    ):
        super().__init__(
            filepath_columns=filepath_columns
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
    def run_singlestruct_step(
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
            with dask_utils.DistributedHandler(distributed_executor_address) as handler:
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
                self.step_local_staging_dir / f"manifest_{struct}.csv", index=False
            )
            return self.manifest


def query_data_from_labkey(cell_line_id):
    """
        This function returns a dataframe containing all the FOVs available
        on LabKey for a particular cell line.


    Parameters
    ----------
    cell_line_id: str
        Name of the cell line where nuclei are going to be sampled from
        AICS-13 = Lamin

    Returns
    -------
    df: pandas dataframe
        Indexed by FOV id
    """

    # Query for labkey data
    db = LabKey(contexts.PROD)

    # Get production data for cell line
    data = db.dataset.get_pipeline_4_production_cells([("CellLine", cell_line_id)])
    data = pd.DataFrame(data)

    # Because we are querying the `cells` dataset and not the `fovs` dataset
    # We need to clean up just a tiny bit

    # NOTE: Tyler is looking into this
    # The only reason we query the `cells` dataset is for the `PixelScale` numbers
    # But those _should_ be exposed on the `fovs` dataset so he is looking into
    # why they aren't. In the future this query should be much simpler.

    # Select down to just the columns we want
    data = data[[
        "FOVId", "CellLine", "Gene", "Protein",
        "PixelScaleX", "PixelScaleY", "PixelScaleZ",
        "SourceReadPath", "ChannelNumber405",
        "ChannelNumber638", "ChannelNumberBrightfield",
        "NucleusSegmentationReadPath",
        "MembraneSegmentationReadPath",
        "StructureSegmentationReadPath"
    ]]

    # Drop duplicates because this dataset will have a row for every cell
    # instead of per-FOV
    data = data.drop_duplicates("FOVId")
    data = data.set_index("FOVId")

    # Fix all filepaths
    data = fix_filepaths(data)

    return data


def fix_filepaths(df):
    """
    Checks the OS and fixes filepaths for Mac.
    Mac users should have created a "data" directory in the repo and mounted
    the data there using:
        mount_smbfs //<YOUR_USERNAME>@allen/programs/allencell/data ./data/

    Parameters
    ----------
    df: dataframe
        dataframe containing filepaths from Labkey
    Returns
    -------
    df: dataframe
        the input dataframe with filepaths truncated for Mac users
    """

    if platform in ["linux", "linux2", "darwin"]:
        pass
    else:
        raise NotImplementedError(
            "OSes other than Linux and Mac are currently not supported."
        )
    return df


def crop_object(raw, seg, obj_label, isotropic=None):
    """
        This function returns a cropped area around an object of interest
        given the raw data and its corresponding segmentation.


    Parameters
    ----------
    raw: zyx numpy.array
        Representing the raw FOV data
    seg: zyx numpy.array
        Representing the corresponding segmentation of raw
    obj_label: int
        Label of the object of interest in the segmented image
    isotropic: integer tuple or None
        Original scale of x, y and z axes. Images are not scaled
        if None is used.

    Returns
    -------
    raw: zyx numpy.array
        isotropic and cropped version of input raw around the obj of interest
    seg: zyx numpy.array
        isotropic and cropped version of input seg around the obj of interest
    """

    offset = 16
    raw = np.pad(raw, ((0, 0), (offset, offset), (offset, offset)), "constant")
    seg = np.pad(seg, ((0, 0), (offset, offset), (offset, offset)), "constant")

    _, y, x = np.where(seg == obj_label)

    if x.shape[0] > 0:

        xmin = x.min() - offset
        xmax = x.max() + offset
        ymin = y.min() - offset
        ymax = y.max() + offset

        raw = raw[:, ymin:ymax, xmin:xmax]
        seg = seg[:, ymin:ymax, xmin:xmax]

        # Resize to isotropic volume
        if isotropic is not None:

            dim = raw.shape

            (sx, sy, sz) = isotropic

            # We fix the target scale to 0.135um. Compatible with 40X

            target_scale = 0.135

            output_shape = np.array([
                sz / target_scale * dim[0],
                sy / target_scale * dim[1],
                sx / target_scale * dim[2]], dtype=np.int)

            raw = sktrans.resize(
                image=raw,
                output_shape=output_shape,
                preserve_range=True,
                anti_aliasing=True).astype(np.uint16)

            seg = sktrans.resize(
                image=seg,
                output_shape=output_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False).astype(np.uint8)

        seg = (seg == obj_label).astype(np.uint8)

        return raw, seg
    else:
        return None, None
