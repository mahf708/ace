import pytest
import torch
from fme.core.distributed import Distributed

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_gather_spatial():
    dist = Distributed.get_instance()

    # Assuming 2 ranks, H=2, W=1.
    # Global H=4, W=8.

    H, W = 4, 8
    slices = dist.get_local_slices((H, W))
    h_slice, w_slice = slices
    H_loc = h_slice.stop - h_slice.start

    # Create tensor: rank index
    tensor = torch.full((1, H_loc, W), dist.rank, device="cuda", dtype=torch.float32)

    # Gather
    gathered = dist.gather_global(tensor, (1, H, W))

    if dist.is_root():
        assert gathered.shape == (1, H, W)
        # Check content
        # Rank 0 should have 0s in top half
        # Rank 1 should have 1s in bottom half
        assert torch.all(gathered[0, 0:2, :] == 0)
        assert torch.all(gathered[0, 2:4, :] == 1)
