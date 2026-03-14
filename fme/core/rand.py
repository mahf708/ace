import contextlib
import os
import random

import numpy as np
import torch

from fme.core.distributed import Distributed

USE_CPU_RANDN = False
_GLOBAL_IMG_SHAPE: tuple[int, int] | None = None


def set_seed(seed: int):
    """
    Set the seed for all random number generators, including numpy, random, torch,
    and Distributed.

    Args:
        seed: The seed to set.
    """
    # https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed + 1)
    random.seed(seed + 2)
    torch.manual_seed(seed + 3)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 4)
    dist = Distributed.get_instance()
    dist.set_seed(seed + 5)


def _generate_and_slice(global_shape, local_slices, **kwargs):
    """Generate a random tensor at global shape, then sub-select the local slice."""
    if USE_CPU_RANDN:
        device = kwargs.pop("device", None)
        full = torch.randn(global_shape, device="cpu", **kwargs)
        return full[(..., *local_slices)].contiguous().to(device)
    else:
        full = torch.randn(global_shape, **kwargs)
        return full[(..., *local_slices)].contiguous()


def randn_like(x: torch.Tensor, **kwargs):
    if _GLOBAL_IMG_SHAPE is not None:
        dist = Distributed.get_instance()
        local_slices = dist.get_local_slices(_GLOBAL_IMG_SHAPE)
        global_shape = list(x.shape)
        global_shape[-2] = _GLOBAL_IMG_SHAPE[0]
        global_shape[-1] = _GLOBAL_IMG_SHAPE[1]
        return _generate_and_slice(
            global_shape, local_slices, dtype=x.dtype, device=x.device, **kwargs
        )
    if USE_CPU_RANDN:
        device = kwargs.pop("device", x.device)
        return torch.randn_like(x, device="cpu", **kwargs).to(device)
    else:
        return torch.randn_like(x, **kwargs)


def randn(shape: torch.Size, **kwargs) -> torch.Tensor:
    if USE_CPU_RANDN:
        device = kwargs.pop("device", None)
        return torch.randn(shape, device="cpu", **kwargs).to(device)
    else:
        return torch.randn(shape, **kwargs)


def randn_spatial(shape: list[int] | tuple[int, ...], **kwargs) -> torch.Tensor:
    """Generate a random tensor, respecting global random state if active.

    Like ``randn``, but when ``use_global_random_state`` is active the last two
    dimensions are treated as spatial: the full global tensor is generated and
    then sub-selected to the local slice for this rank.

    Use this instead of ``torch.randn`` when the tensor has spatial (H, W)
    dimensions that may be decomposed across ranks.
    """
    if _GLOBAL_IMG_SHAPE is not None:
        dist = Distributed.get_instance()
        local_slices = dist.get_local_slices(_GLOBAL_IMG_SHAPE)
        global_shape = list(shape)
        global_shape[-2] = _GLOBAL_IMG_SHAPE[0]
        global_shape[-1] = _GLOBAL_IMG_SHAPE[1]
        return _generate_and_slice(global_shape, local_slices, **kwargs)
    if USE_CPU_RANDN:
        device = kwargs.pop("device", None)
        return torch.randn(shape, device="cpu", **kwargs).to(device)
    else:
        return torch.randn(list(shape), **kwargs)


def log_normal_sample(
    p_mean: float, p_std: float, shape: torch.Size, dtype: torch.dtype
) -> torch.Tensor:
    rnd = randn(shape, dtype=dtype)
    return (rnd * p_std + p_mean).exp()


def log_uniform_sample(
    p_min: float, p_max: float, shape: torch.Size, dtype: torch.dtype
) -> torch.Tensor:
    return torch.exp(
        torch.empty(shape, dtype=dtype).uniform_(np.log(p_min), np.log(p_max))
    )


@contextlib.contextmanager
def use_cpu_randn():
    """
    Context manager to use CPU when generating random numbers for
    randn and randn_like.

    This is likely less performant than generating them directly on the GPU,
    but it allows comparing regression outputs between machines.
    """
    global USE_CPU_RANDN
    old_use_cpu_randn = USE_CPU_RANDN
    USE_CPU_RANDN = True
    yield
    USE_CPU_RANDN = old_use_cpu_randn


@contextlib.contextmanager
def use_global_random_state(img_shape: tuple[int, int]):
    """Generate noise for the full global spatial domain, then sub-select locally.

    When active, ``randn_like`` and ``randn_spatial`` generate the full
    ``(H_global, W_global)`` tensor on every rank and return only the local
    slice.  Because all ranks draw from the same global shape (and therefore
    advance the torch random state identically), the resulting noise is
    reproducible regardless of how the domain is decomposed across ranks.

    This is intended **for testing only** — at high resolutions the redundant
    global allocation is wasteful.  In production a per-rank seed is fine.

    Args:
        img_shape: The global ``(H, W)`` spatial shape.
    """
    global _GLOBAL_IMG_SHAPE
    old = _GLOBAL_IMG_SHAPE
    _GLOBAL_IMG_SHAPE = img_shape
    try:
        yield
    finally:
        _GLOBAL_IMG_SHAPE = old


def alternate_seed(seed: int) -> int:
    """
    Get the alternate seed given a seed.

    Used when a new deterministic random shuffle is desired.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return int(torch.randint(0, 2**31, (1,), generator=g).item())
