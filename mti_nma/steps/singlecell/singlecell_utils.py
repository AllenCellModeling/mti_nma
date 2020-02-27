#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sys import platform

import numpy as np
import pandas as pd
from lkaccess import LabKey, contexts
from skimage import transform as sktrans


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

    # Fix all filepaths
    data = fix_filepaths(data)

    return data


def process_seg_df(df):
    for index in df.index:
        fov_id = df.FOVIdList[index]
        # Some FOVs on labkey do not have segmentation available and
        # therefore fov_id is an empty list.
        if len(fov_id):
            fov_id = fov_id[0]
        else:
            fov_id = None
        df.loc[index, "FOVId"] = fov_id

    df = df.dropna()
    df = df.astype({"FOVId": "int64"})
    df = df.drop(columns=["FOVIdList"])
    return df


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

    if platform == "linux" or platform == "linux2":
        pass
    elif platform == "darwin":
        # if we're in osx, we change all the read paths from
        # /allen/programs/allencell/data/...
        # to
        # ./data/...
        for column in df.columns:
            if "Path" in column:
                df[column] = [
                    readpath.replace("/allen/programs/allencell/", "./")
                    for readpath in df[column]
                ]
    else:
        raise NotImplementedError(
            "OSes other than Linux and Mac are currently not supported."
        )
    return df


def crop_object(raw, seg_nuc, seg_cell, obj_label, isotropic=None):

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
    seg_nuc = np.pad(seg_nuc, ((0, 0), (offset, offset), (offset, offset)), "constant")
    seg_cell = np.pad(seg_cell, ((0, 0), (offset, offset), (offset, offset)), "constant")

    _, y, x = np.where(seg_cell == obj_label)

    if x.shape[0] > 0:

        xmin = x.min() - offset
        xmax = x.max() + offset
        ymin = y.min() - offset
        ymax = y.max() + offset

        raw = raw[:, ymin:ymax, xmin:xmax]
        seg_nuc = seg_nuc[:, ymin:ymax, xmin:xmax]
        seg_cell = seg_cell[:, ymin:ymax, xmin:xmax]
        raw = raw[:, ymin:ymax, xmin:xmax]

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

            seg_nuc = sktrans.resize(
                image=seg_nuc,
                output_shape=output_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False).astype(np.uint8)

            seg_cell = sktrans.resize(
                image=seg_cell,
                output_shape=output_shape,
                order=0,
                preserve_range=True,
                anti_aliasing=False).astype(np.uint8)

        seg_nuc = (seg_nuc == obj_label).astype(np.uint8)
        seg_cell = (seg_cell == obj_label).astype(np.uint8)

        return raw, seg_nuc, seg_cell
    else:
        return None, None, None
