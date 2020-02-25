import numpy as np
import pandas as pd
import labkey as lk
from lkaccess import LabKey as lka
from skimage import transform as sktrans
from sys import platform


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

    filters = [
        # 100X FOVs
        lk.query.QueryFilter("Objective", "100.0", "eq"),
        # Production data
        lk.query.QueryFilter(
            "WellId/PlateId/PlateTypeId/Name", "Production", "contains"),
        # Pipeline 4
        lk.query.QueryFilter("WellId/PlateId/Workflow/Name", "4", "contains"),
        # Not celigo images
        lk.query.QueryFilter("InstrumentId/Name", "Celigo", "neqornull"),
        # Passed QC
        lk.query.QueryFilter("QCStatusId/Name", "Passed", "eq"),
        # Cell line (13 = Lamin)
        lk.query.QueryFilter(
            "SourceImageFileId/cell_line_id/Name", cell_line_id, "contains")
    ]

    query_raw = lka(host="aics").select_rows_as_list(

        schema_name="microscopy",
        query_name="FOV",
        filter_array=filters,
        columns=[
            "FOVId",
            "PixelScaleX",
            "PixelScaleY",
            "PixelScaleZ",
            "SourceImageFileId/LocalFilePath"
        ]

    )

    query_seg_nuc = lka(host="aics").select_rows_as_list(               

        schema_name="fms",
        query_name="FileUI",
        filter_array=filters + [
            # Nuclear segmentations
            lk.query.QueryFilter("ReadPath", "nucWholeIndexImageScale", "contains")
        ],
        columns=[
            "FOVId",
            "ReadPath",
        ]

    )

    query_seg_cell = lka(host="aics").select_rows_as_list(               

        schema_name="fms",
        query_name="FileUI",
        filter_array=filters + [
            # Cell segmentations
            lk.query.QueryFilter("ReadPath", "cellWholeIndexImageScale", "contains")
        ],
        columns=[
            "FOVId",
            "ReadPath",
        ]

    )

    df_seg_nuc = fix_filepaths(pd.DataFrame(query_seg_nuc))
    df_seg_nuc = df_seg_nuc.rename(
        columns={"FOVId": "FOVIdList", "ReadPath": "ReadPathSegNuc"})

    df_seg_cell = fix_filepaths(pd.DataFrame(query_seg_cell))
    df_seg_cell = df_seg_cell.rename(
        columns={"FOVId": "FOVIdList", "ReadPath": "ReadPathSegCell"})

    df_raw = fix_filepaths(pd.DataFrame(query_raw))
    df_raw = df_raw.rename(columns={"SourceImageFileId/LocalFilePath": "ReadPathRaw"})

    # Merging raw and seg tables
    df_seg_nuc = process_seg_df(df_seg_nuc).set_index("FOVId")
    df_seg_cell = process_seg_df(df_seg_cell).set_index("FOVId")
    df_raw = df_raw.set_index("FOVId")

    df_tmp = df_raw.merge(df_seg_nuc, how="inner", left_index=True, right_index=True)
    df = df_tmp.merge(df_seg_cell, how="inner", left_index=True, right_index=True)

    return df


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
