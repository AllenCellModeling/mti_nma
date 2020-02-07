import numpy as np
import pandas as pd
import labkey as lk
from lkaccess import LabKey as lka
from skimage import transform as sktrans

def query_data_from_labkey(CellLineId):

    # Query for labkey data

    filters = [
            # 100X FOVs
            lk.query.QueryFilter('Objective', '100.0', 'eq'),
            # Production data
            lk.query.QueryFilter('WellId/PlateId/PlateTypeId/Name', 'Production', 'contains'),
            # Pipeline 4
            lk.query.QueryFilter('WellId/PlateId/Workflow/Name', '4', 'contains'),
            # Not celigo images
            lk.query.QueryFilter('InstrumentId/Name', 'Celigo', 'neqornull'),
            # Passed QC
            lk.query.QueryFilter('QCStatusId/Name', 'Passed', 'eq'),
            # Cell line (13 = Lamin)
            lk.query.QueryFilter('SourceImageFileId/CellLineId/Name', CellLineId, 'contains')
    ]

    query_raw = lka(host="aics").select_rows_as_list(

        schema_name = "microscopy",
        query_name="FOV",
        filter_array = filters,
        columns = [
            'FOVId',
            'PixelScaleX',
            'PixelScaleY',
            'PixelScaleZ',
            'SourceImageFileId/LocalFilePath'
        ]

    )

    query_seg = lka(host="aics").select_rows_as_list(               

        schema_name = 'fms',
        query_name = 'FileUI',
        filter_array = filters + [
            # Nuclei segmentation only
            lk.query.QueryFilter('ReadPath', 'nucWholeIndexImageScale', 'contains')
        ],
        columns = [
            'FOVId',
            'ReadPath',
        ]

    )

    df_seg = pd.DataFrame(query_seg) # FOVIds are returned as list
    df_seg = df_seg.rename(columns={'FOVId': 'FOVIdList', 'ReadPath': 'ReadPathSeg'})

    df_raw = pd.DataFrame(query_raw)
    df_raw = df_raw.rename(columns={'SourceImageFileId/LocalFilePath': 'ReadPathRaw'})

    # Merging raw and seg tables

    for index in df_seg.index:
        fov_id = df_seg.FOVIdList[index]
        if len(fov_id):
            fov_id = fov_id[0]
        else:
            fov_id = None
        df_seg.loc[index,'FOVId'] = fov_id
    df_seg = df_seg.dropna()
    df_seg = df_seg.astype({'FOVId': 'int64'})
    df_seg = df_seg.drop(columns=['FOVIdList'])

    df_seg = df_seg.set_index('FOVId')
    df_raw = df_raw.set_index('FOVId')

    df = df_raw.merge(df_seg, how='inner', left_index=True, right_index=True)

    return df

def crop_object(raw, seg, obj_label, isotropic=None):

    offset = 16
    raw = np.pad(raw, ((0,0),(offset,offset),(offset,offset)), 'constant')
    seg = np.pad(seg, ((0,0),(offset,offset),(offset,offset)), 'constant')

    _, y, x = np.where(seg==obj_label)

    xmin = x.min() - offset
    xmax = x.max() + offset
    ymin = y.min() - offset
    ymax = y.max() + offset

    seg = seg[:,ymin:ymax,xmin:xmax]
    raw = raw[:,ymin:ymax,xmin:xmax]

    # Resize to isotropic volume
    if isotropic is not None:

        dim = raw.shape

        (sx, sy, sz) = isotropic

        # We fix the target scale to 0.135um. Compatible with 40X

        target_scale = 0.135

        output_shape = np.array([
            sz/target_scale*dim[0],
            sy/target_scale*dim[1],
            sx/target_scale*dim[2]], dtype=np.int)

        raw = sktrans.resize(
            image = raw,
            output_shape = output_shape,
            preserve_range = True,
            anti_aliasing = True).astype(np.uint16)

        seg = sktrans.resize(
            image = seg,
            output_shape = output_shape,
            order = 0,
            preserve_range = True,
            anti_aliasing = False).astype(np.uint8)

    return raw, (seg==obj_label).astype(np.uint8)