"""Online (single-pass) normalization statistics from a data loader."""

import logging
from collections.abc import Iterable
from typing import Protocol

import torch

from fme.core.distributed import Distributed
from fme.core.typing_ import TensorMapping


class _BatchLike(Protocol):
    data: TensorMapping


class StatsAccumulator:
    """Per-variable running totals for mean/std, accumulated in float64.

    Tracks sum, sum-of-squares, and count of finite values. NaNs and infs
    are ignored, matching the ``fill_nans_on_normalize`` semantics where a
    NaN in the source data should not poison the global statistic.
    """

    def __init__(self):
        self._sum_x: dict[str, torch.Tensor] = {}
        self._sum_x2: dict[str, torch.Tensor] = {}
        self._count: dict[str, torch.Tensor] = {}

    def update(self, name: str, tensor: torch.Tensor) -> None:
        x = tensor.detach()
        mask = torch.isfinite(x)
        if not bool(mask.any()):
            return
        x64 = x.to(torch.float64)
        zero = torch.zeros((), dtype=torch.float64, device=x64.device)
        x_masked = torch.where(mask, x64, zero)
        s = x_masked.sum()
        s2 = (x_masked * x_masked).sum()
        n = mask.sum().to(torch.float64)
        if name not in self._sum_x:
            self._sum_x[name] = s
            self._sum_x2[name] = s2
            self._count[name] = n
        else:
            self._sum_x[name] = self._sum_x[name] + s
            self._sum_x2[name] = self._sum_x2[name] + s2
            self._count[name] = self._count[name] + n

    @property
    def names(self) -> list[str]:
        return list(self._sum_x.keys())

    def finalize(
        self, dist: Distributed | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Reduce across ranks (if distributed) and return (means, stds)."""
        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for name in sorted(self._sum_x):
            sum_x = self._sum_x[name].cpu().clone()
            sum_x2 = self._sum_x2[name].cpu().clone()
            count = self._count[name].cpu().clone()
            if dist is not None and dist.world_size > 1:
                sum_x = dist.reduce_sum(sum_x)
                sum_x2 = dist.reduce_sum(sum_x2)
                count = dist.reduce_sum(count)
            if float(count.item()) == 0.0:
                raise ValueError(
                    f"No finite values seen for variable {name!r} while "
                    "computing online normalization stats."
                )
            mean = sum_x / count
            var = (sum_x2 / count) - mean * mean
            var = var.clamp(min=0.0)
            means[name] = float(mean.item())
            stds[name] = float(torch.sqrt(var).item())
        return means, stds


def compute_normalization_stats(
    loader: Iterable[_BatchLike],
    names: Iterable[str] | None = None,
    max_batches: int | None = None,
    dist: Distributed | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Single-pass per-variable mean and std over batches from ``loader``.

    Args:
        loader: Iterable of batch objects with a ``data`` mapping of
            variable name to tensor.
        names: If given, only accumulate these variables; otherwise every
            variable observed is tracked.
        max_batches: Optional cap on the number of batches consumed.
        dist: Distributed instance used to all-reduce partial sums. Defaults
            to ``Distributed.get_instance()``.
    """
    if dist is None:
        dist = Distributed.get_instance()
    wanted: set[str] | None = set(names) if names is not None else None
    acc = StatsAccumulator()
    n_batches = 0
    for batch in loader:
        for name, tensor in batch.data.items():
            if wanted is not None and name not in wanted:
                continue
            acc.update(name, tensor)
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    logging.info(
        "Online stats: accumulated %d batches over %d variables",
        n_batches,
        len(acc.names),
    )
    return acc.finalize(dist=dist)
