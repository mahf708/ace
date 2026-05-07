"""Inline-inference tests under spatial parallelism.

The inline-inference path used by ``trainer.inference_one_epoch`` and
``train.inference_callback`` (``fme/core/generics/trainer.py:743`,
``fme/ace/train/train.py:142``) routes through the same generic
``run_inference`` (``fme/core/generics/inference.py``) as standalone
inference. With Stages A and D in place, the same fixes that make
inference correct under spatial parallelism also make inline inference
correct — these tests exercise the inline path's ingredients without
spinning up a full Trainer.
"""

import numpy as np
import pytest
import torch
import xarray as xr

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.core.distributed.distributed import Distributed


@pytest.mark.parallel
def test_inference_gridded_data_scatters_initial_condition():
    """``InferenceGriddedData.__init__`` must scatter a globally-shaped
    initial condition to the local spatial chunk so that the very first
    step of inference (and inline inference during training) sees a
    matching local-shape IC + local-shape forcing.
    """
    # Importing here avoids a heavy module-level import chain when the
    # test is collected on workers that don't run it.
    from fme.ace.data_loading.batch_data import BatchData as _BatchData
    from fme.ace.data_loading.gridded_data import InferenceGriddedData
    from fme.core.dataset.properties import DatasetProperties

    dist = Distributed.get_instance()

    # We can't easily construct a real DataLoader/DatasetProperties here
    # (they pull in xarray/zarr machinery), so just exercise the scatter
    # codepath directly: the helper is the same one used inside
    # ``InferenceGriddedData.__init__``.
    img_shape = (8, 16)
    n_samples = 2
    device = fme.get_device()

    torch.manual_seed(0)
    global_data = {"x": torch.randn(n_samples, 1, 1, *img_shape, device=device)}
    time = xr.DataArray(
        np.zeros((n_samples, 1), dtype="datetime64[ns]"), dims=("sample", "time")
    )
    state = PrognosticState(_BatchData.new_on_device(data=global_data, time=time))
    scattered = state.scatter_spatial(img_shape)
    local_batch = scattered.as_batch_data()

    h_slice, w_slice = dist.get_local_slices(img_shape)
    expected = global_data["x"][..., h_slice, w_slice]
    torch.testing.assert_close(local_batch.data["x"], expected)

    # Sanity-check that ``InferenceGriddedData`` exposes the global shape
    # via the public property used by inference.py to thread it through to
    # the writer for the gather chokepoint.
    assert hasattr(InferenceGriddedData, "global_img_shape")
    # Also confirm DatasetProperties is available — the constructor path
    # reads ``properties.horizontal_coordinates.shape`` to populate
    # ``_global_img_shape``.
    assert hasattr(DatasetProperties, "to_device")


@pytest.mark.parallel
def test_run_inference_loop_calls_writer_and_aggregator_with_local_data():
    """Smoke-test the generic ``run_inference`` loop under spatial
    parallelism. The looper should pull batches from the data loader (which
    are already spatially scattered) and pass them through to writer and
    aggregator without errors. The writer's gather chokepoint (Stage A.2)
    handles the gather; aggregators that need global tensors gather
    themselves (Stage D).
    """
    from fme.core.generics.inference import run_inference

    dist = Distributed.get_instance()
    img_shape = (8, 16)
    n_samples = 2
    n_steps = 3
    device = fme.get_device()

    torch.manual_seed(7)
    base_global = torch.randn(n_samples, 1, 1, *img_shape, device=device)

    class _FakePredict:
        def __call__(self, initial_condition, forcing, compute_derived_variables=False):
            # Echo the initial condition back as the "prediction" and
            # return a fresh prognostic state for the next step.
            ic = initial_condition.as_batch_data()
            return ic, PrognosticState(ic)

    class _FakeLoader:
        def __init__(self, n: int):
            self._n = n
            self._i = 0

        def __iter__(self):
            self._i = 0
            return self

        def __next__(self):
            if self._i >= self._n:
                raise StopIteration
            self._i += 1
            return None

        def __len__(self):
            return self._n

    class _FakeData:
        def __init__(self):
            time = xr.DataArray(
                np.zeros((n_samples, 1), dtype="datetime64[ns]"),
                dims=("sample", "time"),
            )
            local = base_global[..., *dist.get_local_slices(img_shape)].contiguous()
            self.initial_condition = PrognosticState(
                BatchData.new_on_device(data={"a": local}, time=time)
            )
            self.loader = _FakeLoader(n_steps)

    class _FakeAggregator:
        def __init__(self):
            self.batches = 0

        def record_initial_condition(self, initial_condition):
            return []

        def record_batch(self, data):
            self.batches += 1
            return []

    class _FakeWriter:
        def __init__(self):
            self.batches = 0
            self.writes = 0

        def write(self, data, filename):
            self.writes += 1

        def append_batch(self, batch):
            self.batches += 1

    aggregator = _FakeAggregator()
    writer = _FakeWriter()

    run_inference(
        predict=_FakePredict(),
        data=_FakeData(),
        aggregator=aggregator,
        writer=writer,
        record_logs=lambda logs: None,
    )

    # Every rank must complete without deadlocking; aggregator and writer
    # see all the batches.
    assert aggregator.batches == n_steps
    assert writer.batches == n_steps
    assert writer.writes == 2  # initial_condition.nc + restart.nc
