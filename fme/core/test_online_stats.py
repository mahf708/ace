import math

import pytest
import torch

from fme.core.online_stats import StatsAccumulator, compute_normalization_stats


class _FakeBatch:
    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data


class _FakeDist:
    """Stand-in for the distributed handle so finalize() can run."""

    world_size = 1

    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def test_stats_accumulator_matches_numpy():
    torch.manual_seed(0)
    tensors = [torch.randn(4, 3, 5, 7) for _ in range(5)]
    acc = StatsAccumulator()
    for t in tensors:
        acc.update("a", t)
    means, stds = acc.finalize(dist=_FakeDist())
    all_values = torch.cat([t.reshape(-1) for t in tensors]).to(torch.float64)
    expected_mean = float(all_values.mean().item())
    # population std (divide by N), not sample std
    expected_std = float(
        torch.sqrt(((all_values - all_values.mean()) ** 2).mean()).item()
    )
    assert math.isclose(means["a"], expected_mean, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(stds["a"], expected_std, rel_tol=1e-10, abs_tol=1e-10)


def test_stats_accumulator_ignores_nans():
    t = torch.tensor([1.0, 2.0, float("nan"), 3.0, float("inf"), 4.0])
    acc = StatsAccumulator()
    acc.update("a", t)
    means, stds = acc.finalize(dist=_FakeDist())
    finite = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    expected_mean = float(finite.mean().item())
    expected_std = float(torch.sqrt(((finite - finite.mean()) ** 2).mean()).item())
    assert math.isclose(means["a"], expected_mean)
    assert math.isclose(stds["a"], expected_std)


def test_stats_accumulator_raises_when_all_nonfinite():
    acc = StatsAccumulator()
    acc.update("a", torch.tensor([1.0, 2.0]))
    # name 'b' is never updated → not present and not finalized; only fail
    # for tracked names with zero count.
    acc._sum_x["b"] = torch.tensor(0.0, dtype=torch.float64)
    acc._sum_x2["b"] = torch.tensor(0.0, dtype=torch.float64)
    acc._count["b"] = torch.tensor(0.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="No finite values"):
        acc.finalize(dist=_FakeDist())


def test_compute_normalization_stats_over_loader():
    torch.manual_seed(1)
    batches = [
        _FakeBatch({"a": torch.randn(2, 3, 4), "b": torch.randn(2, 3, 4) * 5 + 7})
        for _ in range(4)
    ]
    means, stds = compute_normalization_stats(batches, dist=_FakeDist())
    assert set(means.keys()) == {"a", "b"}
    # b ~ N(7, 5^2) with limited samples — looser tolerance
    assert abs(means["b"] - 7.0) < 1.0
    assert abs(stds["b"] - 5.0) < 1.0


def test_compute_normalization_stats_respects_max_batches():
    torch.manual_seed(2)
    batches = [_FakeBatch({"a": torch.ones(2, 3)}) for _ in range(10)]
    means, stds = compute_normalization_stats(batches, max_batches=3, dist=_FakeDist())
    assert math.isclose(means["a"], 1.0)
    assert math.isclose(stds["a"], 0.0)


def test_compute_normalization_stats_filters_names():
    batches = [_FakeBatch({"a": torch.ones(2), "b": torch.zeros(2)})]
    means, stds = compute_normalization_stats(batches, names=["a"], dist=_FakeDist())
    assert set(means.keys()) == {"a"}
    assert set(stds.keys()) == {"a"}
