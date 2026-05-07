"""End-to-end inference tests under spatial parallelism.

Locks in:

* ``InferenceGriddedData`` scatters the initial condition (Stage A.1)
* ``PairedDataWriter`` / ``DataWriter`` gather predictions to spatial-rank 0
  before writing (Stage A.2)
* Aggregators that hold per-pixel buffers (video, seasonal, ENSO coefficient
  maps) gather correctly before producing logs / datasets (Stage D)

Baselines for any regression checks here are produced by single-rank
``python -m pytest`` runs and compared against under ``torchrun``.
"""

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import (  # noqa: F401
    BatchData,
    PairedData,
    PrognosticState,
)
from fme.core.distributed.distributed import Distributed


@pytest.mark.parallel
def test_batch_data_gather_spatial_to_root_round_trip():
    """``BatchData.scatter_spatial`` followed by
    ``BatchData.gather_spatial_to_root`` round-trips back to the global
    tensor on rank 0 (and ``None`` elsewhere).
    """
    dist = Distributed.get_instance()
    img_shape = (8, 16)
    n_samples = 2
    device = fme.get_device()

    torch.manual_seed(0)
    global_data = {
        "x": torch.randn(n_samples, 3, *img_shape, device=device),
        "y": torch.randn(n_samples, 1, *img_shape, device=device),
    }
    time = xr.DataArray(
        np.zeros((n_samples, 1), dtype="datetime64[ns]"),
        dims=("sample", "time"),
    )

    batch = BatchData.new_on_device(data=global_data, time=time)
    local = batch.scatter_spatial(img_shape)
    gathered = local.gather_spatial_to_root(img_shape)

    if dist.has_spatial_parallelism():
        is_spatial_root = (
            torch.distributed.get_rank(group=dist._distributed._spatial_group)  # type: ignore[attr-defined]
            == 0
        )
    else:
        is_spatial_root = True

    if is_spatial_root:
        assert gathered is not None
        for k in global_data:
            torch.testing.assert_close(gathered.data[k], global_data[k])
    else:
        assert gathered is None


@pytest.mark.parallel
def test_paired_data_gather_spatial_to_root_round_trip():
    dist = Distributed.get_instance()
    img_shape = (8, 16)
    n_samples = 2
    device = fme.get_device()

    torch.manual_seed(1)
    prediction = {"a": torch.randn(n_samples, 1, *img_shape, device=device)}
    reference = {"a": torch.randn(n_samples, 1, *img_shape, device=device)}
    time = xr.DataArray(
        np.zeros((n_samples, 1), dtype="datetime64[ns]"),
        dims=("sample", "time"),
    )
    # Manually scatter via the scatter helper used by InferenceGriddedData.
    pred_local = dist.scatter_spatial(prediction, img_shape)
    ref_local = dist.scatter_spatial(reference, img_shape)
    paired_local = PairedData(prediction=pred_local, reference=ref_local, time=time)

    gathered = paired_local.gather_spatial_to_root(img_shape)

    if dist.has_spatial_parallelism():
        is_spatial_root = (
            torch.distributed.get_rank(group=dist._distributed._spatial_group)  # type: ignore[attr-defined]
            == 0
        )
    else:
        is_spatial_root = True

    if is_spatial_root:
        assert gathered is not None
        torch.testing.assert_close(gathered.prediction["a"], prediction["a"])
        torch.testing.assert_close(gathered.reference["a"], reference["a"])
    else:
        assert gathered is None


@pytest.mark.parallel
def test_prognostic_state_scatter_then_local_shape():
    """``PrognosticState.scatter_spatial`` gives every rank the local
    spatial slice — used in InferenceGriddedData to wrap an unscattered IC.
    """
    dist = Distributed.get_instance()
    img_shape = (12, 24)
    n_samples = 2
    device = fme.get_device()

    torch.manual_seed(2)
    global_data = {
        "x": torch.randn(n_samples, 1, 1, *img_shape, device=device),
    }
    time = xr.DataArray(
        np.zeros((n_samples, 1), dtype="datetime64[ns]"),
        dims=("sample", "time"),
    )
    batch = BatchData.new_on_device(data=global_data, time=time)
    state = PrognosticState(batch)

    local_state = state.scatter_spatial(img_shape)
    local_batch = local_state.as_batch_data()
    expected_h = (
        dist.get_local_slices(img_shape)[0].stop
        - dist.get_local_slices(img_shape)[0].start
    )
    expected_w = (
        dist.get_local_slices(img_shape)[1].stop
        - dist.get_local_slices(img_shape)[1].start
    )
    assert local_batch.data["x"].shape[-2:] == (expected_h, expected_w)


@pytest.mark.parallel
def test_video_aggregator_returns_global_dataset_on_root():
    """Per-pixel ``VideoAggregator`` should gather across the spatial group
    and produce a global-shape xr.Dataset on world-rank 0; non-root ranks
    return an empty Dataset.
    """
    from fme.ace.aggregator.inference.data import InferenceBatchData
    from fme.ace.aggregator.inference.video import VideoAggregator

    dist = Distributed.get_instance()
    img_shape = (8, 16)
    n_timesteps = 4
    n_samples = 2
    device = fme.get_device()

    aggregator = VideoAggregator(
        n_timesteps=n_timesteps,
        enable_extended_videos=False,
        global_img_shape=img_shape,
    )

    torch.manual_seed(0)
    target_global = {
        "a": torch.randn(n_samples, n_timesteps, *img_shape, device=device)
    }
    pred_global = {"a": torch.randn(n_samples, n_timesteps, *img_shape, device=device)}
    target_local = dist.scatter_spatial(target_global, img_shape)
    pred_local = dist.scatter_spatial(pred_global, img_shape)
    time = xr.DataArray(
        np.zeros((n_samples, n_timesteps), dtype="datetime64[ns]"),
        dims=("sample", "time"),
    )

    aggregator.record_batch(
        InferenceBatchData(
            target=target_local,
            prediction=pred_local,
            time=time,
            i_time_start=0,
        )
    )

    ds = aggregator.get_dataset()
    if dist.is_root():
        assert "a" in ds
        # Variable should be at the *global* spatial extent.
        assert ds["a"].shape[-2:] == img_shape
    else:
        assert len(ds.data_vars) == 0
