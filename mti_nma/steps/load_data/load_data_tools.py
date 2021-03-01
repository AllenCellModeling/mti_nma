import quilt3
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure as skmeasure
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler

def _keep_nucleus_only(fpath):
    seg = AICSImage(fpath).data.squeeze()
    with writers.ome_tiff_writer.OmeTiffWriter(fpath, overwrite_file=True) as writer:
        writer.save(
            seg[0],
            dimension_order = 'ZYX',
            image_name = fpath.stem,
        )

def _fetch_data(index, row, pkg, save_dir):
    ''' Downlaods single cell seg only. '''
    fpath = save_dir/row["crop_seg"]
    pkg[row["crop_seg"]].fetch(fpath)
    _keep_nucleus_only(fpath)

def download_data(
    save_dir: Path,
    nsamples: int=0,
    distributed_executor_address: Optional[str] = None
):
    
    package_name="aics/hipsc_single_cell_image_dataset"
    registry="s3://allencell"
    data_save_loc="quilt_data"

    pkg = quilt3.Package.browse(package_name, registry)
    df = pkg["metadata.csv"]()
    df = df.set_index('CellId', drop=True)

    print(f"Dataset size: {df.shape[0]} elements.")

    df = df.loc[df.cell_stage=='M0']
    
    print(f"Dataset size after removing mitotic cells: {df.shape[0]} elements.")
    
    if nsamples > 0:
        df = df.sample(n=nsamples, random_state=42)
        print(f"Test dataset size: {df.shape[0]} elements.")

    save_dir.mkdir(parents=True, exist_ok=True)
    
    nrows = df.shape[0]
    with DistributedHandler(distributed_executor_address) as handler:
        handler.batched_map(
            _fetch_data,
            *zip(*list(df.iterrows())),
            [pkg]*nrows,
            [save_dir]*nrows
        )
    
    # Rename columns according to the rest of the repo    
    columns_to_keep = {
        'FOVId': 'FOVId',
        'fov_path': 'OriginalFOVPathRaw',
        'fov_seg_path': 'OriginalFOVPathSeg',
        'crop_raw': 'RawFilePath',
        'crop_seg': 'SegFilePath'
    }
        
    df = df[[k for k in columns_to_keep.keys()]]
    df = df.rename(columns = columns_to_keep)
    
    df['SegFilePath'] = save_dir / df['SegFilePath']
    
    return df

