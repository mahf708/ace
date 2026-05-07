"""DCP-based checkpoint helpers with backward-compatible loading of legacy files.

The current trainer historically saved a single pickled file via ``torch.save``
on the root rank only. That format is incompatible with FSDP-sharded weights
and creates an I/O bottleneck at multi-node scale.

This module saves checkpoints as ``torch.distributed.checkpoint`` (DCP)
directories and transparently loads both DCP directories and legacy
single-file checkpoints.

Layout
------

A DCP checkpoint at ``path`` is a directory containing:

- ``path/.metadata`` — DCP-managed index of all (sharded and non-tensor) leaves
- ``path/__<rank>_<thread>.distcp`` — per-rank shards of tensor data

Non-tensor leaves in the saved state dict (ints, floats, strings, dicts of
strings, etc.) are stored in ``.metadata`` via DCP's default bytes-writer.

Save / load mode
----------------

Both helpers support a ``collective`` flag.

- ``collective=False`` (default): a single rank reads/writes the whole
  state dict in isolation. Matches the existing root-only save and
  per-rank load pattern. Today's models replicate parameters across data
  ranks, so a single-rank capture of the full state dict is faithful.
- ``collective=True``: every rank participates and DCP shards tensor data
  across ranks. Required once FSDP-sharded parameters are introduced.

Atomicity
---------

Save writes to ``<path>.tmp/`` and renames to ``<path>`` after the write
returns. Partial writes are removed on failure.
"""

import logging
import os
import shutil
import warnings
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

logger = logging.getLogger(__name__)


def is_dcp_checkpoint(path: str) -> bool:
    """Return True iff ``path`` is a DCP checkpoint directory.

    A DCP checkpoint is a directory containing a ``.metadata`` file.
    A legacy checkpoint is a single regular file.
    """
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, ".metadata"))


def save_state_dict(
    state_dict: dict[str, Any],
    path: str,
    *,
    collective: bool = False,
) -> None:
    """Save ``state_dict`` as a DCP checkpoint at ``path``.

    Args:
        state_dict: Nested dict whose tensor leaves will be sharded and whose
            non-tensor leaves will be stored in DCP metadata.
        path: Destination directory. Created if missing; replaced atomically
            if already present.
        collective: When False (default), only the root rank writes; other
            ranks return immediately. When True, all ranks must call this
            collectively and DCP shards across them. Use ``True`` once any
            saved tensor is FSDP-sharded.
    """
    if collective:
        _save_collective(state_dict, path)
    else:
        _save_root_only(state_dict, path)


def load_state_dict(
    template: dict[str, Any],
    path: str,
    *,
    collective: bool = False,
) -> None:
    """Load a DCP checkpoint at ``path`` into ``template`` in-place.

    ``template`` must have the same nested-dict shape as the saved state
    dict, with tensor leaves of matching shape/dtype. Non-tensor leaves can
    have any value — they are overwritten by the values read from disk.

    Args:
        template: State dict shaped exactly like the saved one, used as
            the destination for the load.
        path: DCP checkpoint directory.
        collective: When False (default), each rank reads independently
            (the saved tensors are replicated across ranks). When True,
            DCP scatters sharded tensors across the ranks calling load.
    """
    if not is_dcp_checkpoint(path):
        raise ValueError(
            f"{path!r} is not a DCP checkpoint directory (no .metadata file found)."
        )
    dcp.load(
        template,
        storage_reader=dcp.FileSystemReader(path),
        no_dist=not collective,
    )


def load_legacy_checkpoint(path: str, map_location: str = "cpu") -> dict[str, Any]:
    """Load a legacy single-file ``torch.save``-style checkpoint.

    Emits a ``DeprecationWarning`` and returns the unpickled dict. New
    checkpoints are always saved as DCP directories; legacy files remain
    loadable for resume of older runs.
    """
    warnings.warn(
        f"Loading legacy single-file checkpoint at {path!r}; new "
        "checkpoints will be saved in DCP (sharded) format.",
        DeprecationWarning,
        stacklevel=2,
    )
    return torch.load(path, map_location=map_location, weights_only=False)


def _save_root_only(state_dict: dict[str, Any], path: str) -> None:
    if not _is_root_rank():
        return
    tmp_path = f"{path}.tmp"
    _replace_with_empty_dir(tmp_path)
    try:
        dcp.save(
            state_dict,
            storage_writer=dcp.FileSystemWriter(tmp_path),
            no_dist=True,
        )
    except BaseException:
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    _replace_path(tmp_path, path)


def _save_collective(state_dict: dict[str, Any], path: str) -> None:
    is_root = _is_root_rank()
    tmp_path = f"{path}.tmp"
    if is_root:
        _replace_with_empty_dir(tmp_path)
    _maybe_barrier()
    try:
        dcp.save(state_dict, storage_writer=dcp.FileSystemWriter(tmp_path))
    except BaseException:
        if is_root and os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    _maybe_barrier()
    if is_root:
        _replace_path(tmp_path, path)
    _maybe_barrier()


def _replace_with_empty_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    os.makedirs(path, exist_ok=True)


def _replace_path(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    elif os.path.isfile(dst):
        os.remove(dst)
    os.rename(src, dst)


def _is_root_rank() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
