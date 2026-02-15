import numpy as np
import pytest
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.distributed.parallel_tests._helpers import WORLD_SIZE, requires_parallel
from fme.core.gridded_ops import LatLonOperations

@requires_parallel
@pytest.mark.parametrize(
    "h_parallel,w_parallel",
    [
        (2, 1),
        (1, 2),
    ],
)
def test_area_weighted_ops_spatial(h_parallel, w_parallel, monkeypatch):
    """Test area weighted operations with spatial parallelism."""
    world_size = WORLD_SIZE
    spatial_size = h_parallel * w_parallel
    if world_size % spatial_size != 0:
        pytest.skip(f"world_size={world_size} not divisible by spatial_size={spatial_size}")

    nx = 8
    ny = 8

    monkeypatch.setenv("H_PARALLEL_SIZE", str(h_parallel))
    monkeypatch.setenv("W_PARALLEL_SIZE", str(w_parallel))

    # Setup data
    torch.manual_seed(0)
    data_host = torch.randn(1, 2, nx, ny).to("cpu")
    weights_host = torch.rand(nx, ny).to("cpu")

    # Serial execution
    with Distributed.force_non_distributed():
        ops_serial = LatLonOperations(weights_host)
        sum_serial = ops_serial.area_weighted_sum(data_host)
        mean_serial = ops_serial.area_weighted_mean(data_host)
        zonal_mean_serial = ops_serial.zonal_mean(data_host)

    # Parallel execution
    dist = Distributed.get_instance()
    # LatLonOperations handles slicing of weights in __init__
    # But it expects global weights as input (which it slices)
    ops_parallel = LatLonOperations(weights_host)

    # We must manually slice data for the parallel ops
    # get_local_slices returns slices for the global shape.
    # Note: data_host is (batch, chan, lat, lon).
    # get_local_slices might return 2 slices for (lat, lon).
    # We need to apply them to last 2 dims.
    local_slices = dist.get_local_slices((nx, ny))
    data_local = data_host[..., local_slices[0], local_slices[1]].to(get_device())

    sum_parallel = ops_parallel.area_weighted_sum(data_local)
    mean_parallel = ops_parallel.area_weighted_mean(data_local)
    zonal_mean_parallel = ops_parallel.zonal_mean(data_local)

    # Comparison
    # Sum and mean are global scalars (reduced)
    assert torch.allclose(sum_serial.to(get_device()), sum_parallel, atol=1e-5), \
        f"Sum mismatch: serial={sum_serial}, parallel={sum_parallel}"
    assert torch.allclose(mean_serial.to(get_device()), mean_parallel, atol=1e-5), \
        f"Mean mismatch: serial={mean_serial}, parallel={mean_parallel}"

    # Zonal mean returns tensor of shape (..., lat).
    # If H is split, we compare with the slice of serial result.
    zonal_mean_serial_slice = zonal_mean_serial[..., local_slices[0]]
    assert torch.allclose(zonal_mean_serial_slice.to(get_device()), zonal_mean_parallel, atol=1e-5), \
        f"Zonal mean mismatch"
