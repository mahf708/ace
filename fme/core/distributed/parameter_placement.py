"""What spatial co-ranks own of each parameter, and keeping them agreed.

Spatial parallelism splits *activations* across ranks; it does not, today,
split parameters. Every rank builds the model at the global extents and slices
in the forward pass, so each rank's gradient is a partial sum over its own
tile, and summing those partials across the spatial group reconstructs the
gradient a single-rank run would have computed.

That reasoning rests on two things nothing else enforces.

The parameters must really be replicated. If one is localized -- storing only
this rank's spectral modes, say -- the shapes still line up, the spatial sum
still runs, and each shard is updated with the sum of gradients belonging to
*different* modes. The result is numerically stable and scientifically wrong,
and the root-only checkpoint has the same blind spot: it would save rank 0's
slice and call it the model. There is no way to fix that here, so a
differently-sized parameter is refused.

They must also *start* identical. DDP broadcasts initial parameters across the
group it is given, which here is the data group; nothing does the same for the
spatial group, which is left relying on every rank having been seeded
identically and having executed exactly the same construction. When that holds
the broadcast below is a no-op; when it does not, spatial co-ranks would
otherwise apply the same summed gradient to different weights forever, and no
rank would hold the model anyone believes was trained. So rather than assert
the assumption, this makes it true: the spatial-group root's parameters win.

Buffers are deliberately left alone -- the SHT layers' precomputed Legendre
polynomials are per-rank by design, and broadcasting them would destroy the
decomposition.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed

logger = logging.getLogger(__name__)


class SpatiallyShardedParameter(NotImplementedError):
    """A parameter is sized differently across spatial ranks."""


def _trainable(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Trainable parameters in a rank-independent order."""
    return [
        (name, param)
        for name, param in sorted(module.named_parameters())
        if param.requires_grad
    ]


def _disagreeing(
    local: torch.Tensor, group: torch.distributed.ProcessGroup
) -> list[bool]:
    """Which entries are not identical across every rank in the group."""
    if local.numel() == 0:
        return []
    lo, hi = local.clone(), local.clone()
    torch.distributed.all_reduce(lo, op=torch.distributed.ReduceOp.MIN, group=group)
    torch.distributed.all_reduce(hi, op=torch.distributed.ReduceOp.MAX, group=group)
    return (lo != hi).tolist()


def synchronize_replicated_parameters(
    module: torch.nn.Module,
    spatial_group: torch.distributed.ProcessGroup,
) -> list[str]:
    """Require replicated shapes, then make spatial co-ranks agree on values.

    Runs once per wrapped module, so its handful of collectives costs nothing
    next to training.

    Args:
        module: The module about to be wrapped for distributed training.
        spatial_group: The process group whose ranks hold tiles of the same
            sample.

    Returns:
        Names of the parameters whose values had to be corrected, empty when
        the ranks already agreed. A non-empty result means the model was built
        rank-dependently, which is worth knowing about even though it has just
        been repaired.

    Raises:
        SpatiallyShardedParameter: If a parameter's size differs between
            spatial co-ranks.
    """
    trainable = _trainable(module)
    if not trainable:
        return []
    device = trainable[0][1].device
    names = [name for name, _ in trainable]

    numels = torch.tensor(
        [float(param.numel()) for _, param in trainable],
        dtype=torch.float64,
        device=device,
    )
    mismatched_size = _disagreeing(numels, spatial_group)
    if any(mismatched_size):
        offenders = [n for n, bad in zip(names, mismatched_size) if bad]
        raise SpatiallyShardedParameter(
            "spatial co-ranks hold different numbers of elements for "
            f"{offenders}, so these parameters are spatially sharded. That is "
            "not supported: the spatial gradient reduction sums each rank's "
            "gradient, which for a sharded parameter mixes gradients belonging "
            "to different shards, and the root-only checkpoint would save only "
            "rank 0's slice. Supporting sharded parameters needs "
            "placement-aware gradient reduction and a distributed checkpoint "
            "format; until both exist, store the parameter at its global "
            "extent and slice it in the forward pass."
        )

    # Bit-identical parameters give a bit-identical checksum, since the
    # reduction order is fixed by the tensor layout, so this has no false
    # positives.
    checksums = torch.stack(
        [param.detach().double().sum() for _, param in trainable]
    ).to(device)
    corrected = [
        name for name, bad in zip(names, _disagreeing(checksums, spatial_group)) if bad
    ]

    root = torch.distributed.get_global_rank(spatial_group, 0)
    with torch.no_grad():
        for _, param in trainable:
            torch.distributed.broadcast(param.data, root, group=spatial_group)

    if corrected:
        logger.warning(
            "spatial co-ranks disagreed on the initial value of %s; they have "
            "been overwritten with the spatial-group root's. Replicated "
            "parameters are expected to be constructed identically on every "
            "rank, so this usually means a rank-dependent draw or that ranks "
            "disagree on the seed.",
            corrected,
        )
    return corrected
