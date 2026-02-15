import os
import pytest
import torch
import xarray as xr
import numpy as np
import pandas as pd
from fme.core.distributed import Distributed
from fme.core.dataset.xarray import XarrayDataset, XarrayDataConfig
from fme.core.dataset.schedule import IntSchedule

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_spatial_data_loading():
    dist = Distributed.get_instance()

    # Create dummy data
    # H=4, W=8
    # We assume we run with H_PARALLEL_SIZE=2, W_PARALLEL_SIZE=1 (2 spatial ranks)
    # Total ranks = 2.

    if dist.rank == 0:
        ds = xr.Dataset(
            {
                "a": (("time", "lat", "lon"), np.random.rand(5, 4, 8)),
            },
            coords={
                "time": pd.date_range("2000-01-01", periods=5, freq="D"),
                "lat": np.arange(4),
                "lon": np.arange(8),
            }
        )
        ds.to_netcdf("test_data_spatial.nc")

    dist.barrier()

    config = XarrayDataConfig(
        data_path=".",
        file_pattern="test_data_spatial.nc",
        spatial_dimensions="latlon",
    )

    dataset = XarrayDataset(config, ["a"], IntSchedule.from_constant(1))

    # Check shape
    # If H_PARALLEL=2, local H should be 2.
    item = dataset[0]
    local_shape = item[0]["a"].shape # (1, H_loc, W_loc)

    assert local_shape == (1, 2, 8)

    # Check content consistency?
    # We can gather and check?

    dist.barrier()
    if dist.rank == 0:
        os.remove("test_data_spatial.nc")
