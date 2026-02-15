import os
import datetime
import pytest
import torch
import xarray as xr
import numpy as np
import pandas as pd
from fme.core.distributed import Distributed
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.core.dataset.data_typing import VariableMetadata

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_spatial_data_writing_netcdf():
    dist = Distributed.get_instance()

    # Global shapes
    H, W = 4, 8
    n_samples = 2
    n_timesteps = 1

    # Coords
    coords = {
        "lat": np.arange(H, dtype=np.float32),
        "lon": np.arange(W, dtype=np.float32),
    }

    # Metadata
    variable_metadata = {"a": VariableMetadata("units", "long_name")}
    dataset_metadata = DatasetMetadata(history="test")

    config = DataWriterConfig(
        save_prediction_files=True,
        save_monthly_files=False,
    )

    writer = config.build(
        experiment_dir=".",
        n_initial_conditions=n_samples,
        n_timesteps=n_timesteps,
        timestep=datetime.timedelta(hours=1),
        variable_metadata=variable_metadata,
        coords=coords,
        dataset_metadata=dataset_metadata,
    )

    # Create local data
    # H_loc = 2 (if H_PARALLEL=2)
    # n_samples_loc = 1 (if 2 data ranks? No, here we test spatial only)
    # If 2 ranks total, and they are spatial split (H=2), then batch is NOT split (D=1).
    # So n_samples_loc = n_samples = 2.

    slices = dist.get_local_slices((H, W))
    h_slice, w_slice = slices
    H_loc = h_slice.stop - h_slice.start
    W_loc = w_slice.stop - w_slice.start

    data = {
        "a": torch.rand(n_samples, n_timesteps, H_loc, W_loc, device="cuda")
    }
    batch_time = xr.DataArray(
        np.array([
            datetime.datetime(2000, 1, 1),
            datetime.datetime(2000, 1, 2)
        ]),
        dims=["sample"]
    )

    writer.append_batch(data, batch_time)
    writer.finalize()

    dist.barrier()

    if dist.rank == 0:
        # Check output file
        ds = xr.open_dataset("autoregressive_predictions.nc")
        assert ds["a"].shape == (n_samples, n_timesteps, H, W)
        os.remove("autoregressive_predictions.nc")
