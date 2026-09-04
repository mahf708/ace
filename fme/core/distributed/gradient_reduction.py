"""The DDP communication hook that closes spatial parallelism's backward pass.

Note this module deliberately does not use ``from __future__ import
annotations``: DDP validates a communication hook by comparing the ``bucket``
parameter's annotation against ``dist.GradBucket`` with ``inspect.signature``,
and postponed evaluation would make that annotation a string and fail the
check.
"""

import torch
import torch.distributed


def spatial_then_data_allreduce_hook(
    state: tuple[torch.distributed.ProcessGroup, torch.distributed.ProcessGroup],
    bucket: torch.distributed.GradBucket,
) -> torch.futures.Future[torch.Tensor]:
    """Sum a gradient bucket over the spatial group, then average over data.

    Each spatial rank sees only its tile of the input, so its gradient is a
    partial sum over that tile; summing the partials reconstructs the gradient
    a single-rank run would have computed. Averaging that over the data group
    is what DDP would otherwise have done on its own. The two groups are
    orthogonal, so the order does not matter -- what matters is that each
    happens exactly once per parameter.

    Doing this here rather than from a per-parameter ``Tensor.register_hook``
    means one collective per DDP bucket instead of one per parameter -- for an
    SFNO, hundreds of small blocking collectives per step -- and lets the
    spatial reduction ride the buckets DDP already overlaps with the rest of
    the backward pass.

    A bucket is a flat concatenation of gradients and cannot treat its members
    differently, so this is correct only while every parameter is replicated
    across the spatial group. `parameter_placement.assert_parameters_replicated`
    checks that before the hook is installed.

    Args:
        state: The ``(spatial_group, data_group)`` process groups.
        bucket: The gradient bucket DDP is ready to reduce.

    Returns:
        A future completing when the bucket holds the reduced gradient.
    """
    spatial_group, data_group = state
    buffer = bucket.buffer()
    torch.distributed.all_reduce(buffer, group=spatial_group)
    buffer.div_(torch.distributed.get_world_size(group=data_group))
    return (
        torch.distributed.all_reduce(buffer, group=data_group, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )
