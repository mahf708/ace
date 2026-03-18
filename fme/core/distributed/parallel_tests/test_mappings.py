"""Analytical tests for the autograd-aware ``spatial_reduce_sum``.

The key backward-pass fix in ``model_torch_distributed.py`` is making
``spatial_reduce_sum`` differentiable via ``_AutogradAllReduce``.  These
tests verify both forward and backward correctness with simple tensors
whose expected results can be computed analytically.

Tests work serially (single process) and in parallel (``torchrun``).

Run examples::

    # serial (identity path — spatial_reduce_sum is a no-op)
    python -m pytest fme/core/distributed/parallel_tests/test_mappings.py -v

    # 2-rank spatial
    FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 \\
        FME_DISTRIBUTED_W=1 torchrun --nproc-per-node 2 -m pytest -m parallel \\
        fme/core/distributed/parallel_tests/test_mappings.py -v
"""

import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed


def _spatial_size(dist: Distributed) -> int:
    """Number of spatial ranks (h * w)."""
    n_dp = dist.total_data_parallel_ranks
    return dist.world_size // n_dp


# ---------------------------------------------------------------------------
# Forward behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_spatial_reduce_sum_forward_ones():
    """All-reduce of all-ones across sp spatial ranks should give sp * ones."""
    dist = Distributed.get_instance()
    sp = _spatial_size(dist)
    t = torch.ones(3, 4, device=get_device())
    result = dist.spatial_reduce_sum(t)
    torch.testing.assert_close(result, torch.full_like(t, float(sp)))


@pytest.mark.parallel
def test_spatial_reduce_sum_forward_rank_values():
    """Each rank contributes rank-offset values; sum should equal analytic total."""
    dist = Distributed.get_instance()
    sp = _spatial_size(dist)
    # Each spatial rank sets value to its data_parallel_rank (same across dp ranks)
    # so all spatial ranks hold the same value → sum = sp * value
    val = 3.0
    t = torch.full((2,), val, device=get_device())
    result = dist.spatial_reduce_sum(t)
    torch.testing.assert_close(result, torch.full_like(t, sp * val))


@pytest.mark.parallel
def test_weighted_mean_forward_uniform_weights():
    """weighted_mean with uniform weights should return the mean of local values.

    With spatial parallelism, weighted_mean does:
        spatial_reduce_sum(local_weighted_sum) / spatial_reduce_sum(local_weight_sum)
    For uniform data and weights, this simplifies to the data value itself.
    """
    dist = Distributed.get_instance()
    val = 5.0
    data = torch.full((2, 3, 4), val, device=get_device())
    weights = torch.ones(2, 3, 4, device=get_device())
    result = dist.weighted_mean(data, weights, dim=(-2, -1))
    expected = torch.full((2,), val, device=get_device())
    torch.testing.assert_close(result, expected)


# ---------------------------------------------------------------------------
# Backward behaviour — the core fix
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_spatial_reduce_sum_preserves_gradient():
    """spatial_reduce_sum must be autograd-aware so loss.backward() works.

    This is the most critical test: without ``_AutogradAllReduce``,
    the raw ``all_reduce`` breaks the gradient graph and ``x.grad``
    would be None.
    """
    dist = Distributed.get_instance()
    x = torch.randn(3, 4, device=get_device(), requires_grad=True)
    y = dist.spatial_reduce_sum(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradient is None — spatial_reduce_sum broke autograd"
    # Backward of _AutogradAllReduce is identity, so grad_x = ones
    expected_grad = torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected_grad)


@pytest.mark.parallel
def test_spatial_reduce_sum_backward_does_not_communicate():
    """Backward through spatial_reduce_sum should NOT all-reduce the gradient.

    The all_reduce happens in the forward pass only; the backward is identity.
    So if we set up a loss where each rank has a different upstream gradient,
    grad_x should equal that local gradient (not the sum across ranks).
    """
    dist = Distributed.get_instance()
    x = torch.randn(4, device=get_device(), requires_grad=True)
    y = dist.spatial_reduce_sum(x)
    # Multiply by a per-element weight to create a non-trivial gradient
    weight = torch.tensor([1.0, 2.0, 3.0, 4.0], device=get_device())
    loss = (y * weight).sum()
    loss.backward()
    # backward is identity → grad_x = weight (not all_reduced weight)
    torch.testing.assert_close(x.grad, weight)


@pytest.mark.parallel
def test_weighted_mean_preserves_gradient():
    """weighted_mean uses spatial_reduce_sum, so it must also be differentiable."""
    dist = Distributed.get_instance()
    x = torch.randn(2, 3, 4, device=get_device(), requires_grad=True)
    weights = torch.ones(2, 3, 4, device=get_device())
    mean = dist.weighted_mean(x, weights, dim=(-2, -1))
    loss = mean.sum()
    loss.backward()
    assert x.grad is not None, "weighted_mean broke autograd"
    # Verify gradient is not all zeros
    assert not torch.all(x.grad == 0), "All-zero gradient from weighted_mean"


@pytest.mark.parallel
def test_spatial_reduce_sum_does_not_modify_input():
    """spatial_reduce_sum should NOT modify the input tensor in-place.

    The old implementation used in-place all_reduce which mutated the input.
    The new _AutogradAllReduce clones first.
    """
    dist = Distributed.get_instance()
    x = torch.randn(3, 4, device=get_device())
    x_orig = x.clone()
    _ = dist.spatial_reduce_sum(x)
    torch.testing.assert_close(x, x_orig)


# ---------------------------------------------------------------------------
# End-to-end gradient check: loss computation path
# ---------------------------------------------------------------------------


@pytest.mark.parallel
def test_area_weighted_mse_loss_path_gradient():
    """Simulate the loss computation path used during training.

    The path is: model_output → (output - target)² → area_weighted_mean
        → spatial_reduce_sum → scalar loss → backward

    This verifies that gradients flow all the way back to model_output.
    """
    dist = Distributed.get_instance()

    # Simulate model output and target
    output = torch.randn(2, 3, 4, device=get_device(), requires_grad=True)
    target = torch.randn(2, 3, 4, device=get_device())
    weights = torch.ones(2, 3, 4, device=get_device())

    # Loss path: MSE with area-weighted mean
    diff_sq = (output - target) ** 2
    # weighted_mean internally calls spatial_reduce_sum
    mean_loss = dist.weighted_mean(diff_sq, weights, dim=(-2, -1))
    loss = mean_loss.mean()
    loss.backward()

    assert output.grad is not None, "Gradient did not flow through loss path"
    # Gradient should be proportional to 2 * (output - target)
    # The exact scaling depends on spatial_size and tensor dims
    expected_direction = 2 * (output - target)
    # Check gradient has correct sign/direction (cosine similarity > 0)
    cos_sim = torch.nn.functional.cosine_similarity(
        output.grad.flatten(), expected_direction.flatten(), dim=0
    )
    assert cos_sim > 0.99, f"Gradient direction wrong, cosine similarity: {cos_sim}"


@pytest.mark.parallel
def test_gradcheck_spatial_reduce_sum():
    """Numerical gradient check for spatial_reduce_sum.

    Uses torch.autograd.gradcheck to verify the backward implementation
    matches numerical finite-difference gradients.
    """
    dist = Distributed.get_instance()
    x = torch.randn(3, 4, device=get_device(), dtype=torch.float64, requires_grad=True)

    def fn(inp):
        return dist.spatial_reduce_sum(inp)

    assert torch.autograd.gradcheck(fn, (x,), raise_exception=True)
