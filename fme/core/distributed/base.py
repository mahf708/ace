from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar("T")


def _pad_along_dim(
    tensor: torch.Tensor, dim: int, left: int, right: int, mode: str = "replicate"
) -> torch.Tensor:
    """Pad *tensor* along a single *dim* using ``F.pad``.

    ``F.pad`` expects pairs from the last dimension backwards, so we build
    the pad tuple accordingly.  Supports any mode accepted by ``F.pad``
    (``'replicate'``, ``'circular'``, ``'constant'``, …).
    """
    if left == 0 and right == 0:
        return tensor
    ndim = tensor.dim()
    dim_idx = dim if dim >= 0 else ndim + dim
    # F.pad pairs: (last_left, last_right, …, dim_left, dim_right)
    rev_idx = ndim - 1 - dim_idx
    # For non-constant modes, F.pad requires at least 2*(ndim-2) pad entries.
    min_pairs = max(rev_idx + 1, ndim - 2) if mode != "constant" else rev_idx + 1
    pad_sizes = [0] * (2 * min_pairs)
    pad_sizes[2 * rev_idx] = left
    pad_sizes[2 * rev_idx + 1] = right
    return F.pad(tensor, pad_sizes, mode=mode)


class DistributedBackend(ABC):
    """
    Interface that TorchDistributed / NonDistributed must implement.
    """

    @property
    @abstractmethod
    def rank(self) -> int:
        """Global rank of this process."""
        ...

    @property
    @abstractmethod
    def data_parallel_rank(self) -> int: ...

    @property
    @abstractmethod
    def total_ranks(self) -> int:
        """Total number of processes."""
        ...

    @property
    @abstractmethod
    def total_data_parallel_ranks(self) -> int:
        """
        Total number of rank splits along the data parallel dimension.

        For example, 8 ranks using 2 ranks of model parallelism would have
        only 4 ranks of data paralellism.
        """

    @abstractmethod
    def local_batch_size(self, batch_size: int) -> int: ...

    @abstractmethod
    def get_local_slices(self, tensor_shape, data_parallel_dim: int | None = None): ...

    @abstractmethod
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_min(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def reduce_max(self, tensor: torch.Tensor) -> torch.Tensor | None: ...

    @abstractmethod
    def gather(
        self, tensor: torch.Tensor, gather_list: list[torch.Tensor] | None
    ) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process.

        Note: tensor shape is assumed to be equal across all processes; data will
            reshaped/filled/dropped to coerce non-root tensors to the shape
            of the root tensor if not. To avoid this behavior, use
            "gather_irregular" instead.

        Args:
            tensor: The tensor to gather.
            gather_list: A list of tensor buffers to gather into,
                one for each rank.

        Returns:
            A list of tensors, where the i-th element is the tensor
                from the i-th process.
        """
        ...

    @abstractmethod
    def gather_object(self, obj: T) -> list[T] | None: ...

    @abstractmethod
    def scatter_object(self, obj: T) -> T: ...

    @abstractmethod
    def gather_irregular(self, tensor: torch.Tensor) -> list[torch.Tensor] | None:
        """
        Gather a tensor from all processes to the root process. The rank tensors
        may have diferent dimension lengths, but must have the same number of
        dimensions.

        Args:
            tensor: The tensor to gather.

        Returns:
            A list of tensors of consistent shape, where the i-th element is the tensor
                from the i-th process.
        """
        ...

    @abstractmethod
    def wrap_module(self, module: torch.nn.Module) -> torch.nn.Module:
        """
        Wrap a module in for distributed training, if required.

        The wrapped module must follow the module structure of DistributedDataParallel,
        with the passed module's state contained under "module".
        """
        ...

    @abstractmethod
    def barrier(self): ...

    @abstractmethod
    def shutdown(self): ...

    @abstractmethod
    def get_sht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        """Create a forward SHT (possibly distributed)."""
        ...

    @abstractmethod
    def get_isht(
        self,
        nlat: int,
        nlon: int,
        lmax: int | None = None,
        mmax: int | None = None,
        grid: str = "legendre-gauss",
    ) -> nn.Module:
        """Create an inverse SHT (possibly distributed)."""
        ...

    @abstractmethod
    def get_disco_conv_s2(self, *args, **kwargs) -> nn.Module:
        """Create a disco conv S2 instance (possibly distributed)."""
        ...

    @abstractmethod
    def spatial_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce sum across spatial (h, w) ranks. Identity for non-spatial."""
        ...

    @abstractmethod
    def weighted_mean(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
        keepdim: bool = False,
    ) -> torch.Tensor:
        """Compute a weighted mean, correctly handling spatial parallelism."""
        ...

    @abstractmethod
    def zonal_mean(self, data: torch.Tensor) -> torch.Tensor:
        """Compute the zonal mean (mean over longitude dimension)."""
        ...

    @abstractmethod
    def halo_exchange(
        self,
        tensor: torch.Tensor,
        dim: int,
        width: int,
        periodic: bool = False,
    ) -> tuple[torch.Tensor, int, int]:
        """Pad *tensor* with ghost cells from neighboring spatial ranks.

        Args:
            tensor: Input tensor.
            dim: Dimension to exchange along (``-1`` for w, ``-2`` for h).
            width: Number of ghost cells requested on each side.
            periodic: Whether the dimension wraps around (e.g. longitude).

        Returns:
            ``(padded_tensor, left_width, right_width)`` — *left_width* /
            *right_width* may be less than *width* at non-periodic domain
            boundaries.
        """
        ...

    def rolling(
        self,
        tensor: torch.Tensor,
        dim: int,
        window_size: int,
        periodic: bool = False,
    ) -> torch.Tensor:
        """Distributed rolling mean along *dim*.

        Args:
            tensor: Input tensor.
            dim: Dimension to roll along.
            window_size: Size of the rolling window (must be odd).
            periodic: Whether the dimension wraps around.

        Returns:
            Tensor of the same shape as *tensor* with the rolling mean applied.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        half = window_size // 2
        if half == 0:
            return tensor
        padded, left, right = self.halo_exchange(tensor, dim, half, periodic)
        missing_left = half - left
        missing_right = half - right
        if missing_left > 0 or missing_right > 0:
            padded = _pad_along_dim(padded, dim, missing_left, missing_right)
        return padded.unfold(dim, window_size, 1).mean(dim=-1)

    @abstractmethod
    def gradient_magnitude_percent_diff(
        self,
        truth: torch.Tensor,
        predicted: torch.Tensor,
        weights: torch.Tensor,
        dim: tuple[int, ...],
    ) -> torch.Tensor:
        """Compute percent difference of weighted mean gradient magnitude."""
        ...
