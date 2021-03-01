import quilt3
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import measure as skmeasure
from aicsimageio import AICSImage, writers
from typing import Dict, List, Optional, Union
from aics_dask_utils import DistributedHandler

def _fetch_data(index, row, pkg, save_dir):
    pkg[row["crop_seg"]].fetch(save_dir/row["crop_seg"])

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

    print(f"Dataset size: {df.shape[0]} elements.")

    df = df.loc[df.cell_stage=='M0']
    
    print(f"Dataset size after removing mitotic cells: {df.shape[0]} elements.")
    
    if nsamples > 0:
        df = df.sample(n=nsamples, random_state=42)
        print(f"Test dataset size: {df.shape[0]} elements.")

    # Download the data
    save_dir.mkdir(parents=True, exist_ok=True)
    
    nrows = df.shape[0]
    with DistributedHandler(distributed_executor_address) as handler:
        handler.batched_map(
            _fetch_data,
            *zip(*list(df.iterrows())),
            [pkg]*nrows,
            [save_dir]*nrows
        )
    
    return df

