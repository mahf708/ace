import torch
import torch.distributed as dist
from torch import nn

try:
    from fme.core.distributed import model_torch_distributed_comm as comm
except ImportError:
    comm = None


def gather_tensor(tensor: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """
    Gathers a tensor along a specific dimension using a distributed process group.

    Args:
        tensor: The local tensor shard.
        dim: The dimension to gather along.
        group_name: The name of the process group (e.g., "h", "w").

    Returns:
        The gathered full tensor.
    """
    if comm is None:
        return tensor

    group = comm.get_group(group_name)
    if group is None or comm.get_size(group_name) <= 1:
        return tensor

    world_size = comm.get_size(group_name)

    # Gather shapes first to handle potentially uneven splits
    local_shape = torch.tensor(tensor.shape, device=tensor.device)
    all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
    dist.all_gather(all_shapes, local_shape, group=group)

    # Allocate output tensors
    output_tensors = []
    for i in range(world_size):
        shape = tuple(all_shapes[i].tolist())
        output_tensors.append(torch.empty(shape, device=tensor.device, dtype=tensor.dtype))

    # Gather data
    dist.all_gather(output_tensors, tensor, group=group)

    # Concatenate along dim
    return torch.cat(output_tensors, dim=dim)


def scatter_tensor(tensor: torch.Tensor, dim: int, group_name: str) -> torch.Tensor:
    """
    Scatters/slices a full tensor into a local shard along a dimension.

    Args:
        tensor: The full tensor (must be same on all ranks in the group ideally,
                or at least this rank has the full tensor).
        dim: The dimension to scatter along.
        group_name: The name of the process group.

    Returns:
        The local shard.
    """
    if comm is None:
        return tensor

    group = comm.get_group(group_name)
    if group is None or comm.get_size(group_name) <= 1:
        return tensor

    # We need to know the split sizes.
    # We assume physicsnemo's split logic or we need to recompute it.
    # physicsnemo.distributed.utils.compute_split_shapes(total_size, num_chunks)

    try:
        from physicsnemo.distributed import utils as pnd_utils
    except ImportError:
        # Fallback to even split if pnd is missing (should not happen if comm is present)
        # Or raise error
        total_size = tensor.shape[dim]
        rank = comm.get_rank(group_name)
        size = comm.get_size(group_name)
        chunk_size = total_size // size
        start = rank * chunk_size
        end = start + chunk_size
        # Handle remainder if any? physicsnemo handles it.
        # If we can't import pnd, we are in trouble.
        return torch.split(tensor, chunk_size, dim=dim)[rank]

    total_size = tensor.shape[dim]
    world_size = comm.get_size(group_name)
    rank = comm.get_rank(group_name)

    shapes = pnd_utils.compute_split_shapes(total_size, world_size)
    local_size = shapes[rank]

    # Calculate start offset
    start = sum(shapes[:rank])
    end = start + local_size

    # Slice
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(start, end)

    return tensor[tuple(slices)].contiguous()


def gather_sharded_state_dict(model: nn.Module, state_dict: dict = None) -> dict:
    """
    Gathers sharded parameters in state_dict based on annotations in model.
    Must be called on all ranks in the relevant process groups.

    Args:
        model: The model containing parameter annotations.
        state_dict: The state dict containing local shards. If None, uses model.state_dict().

    Returns:
        New state dict with gathered parameters.
    """
    if state_dict is None:
        state_dict = model.state_dict()

    new_state_dict = {}

    # Build map from name to param/buffer
    name_to_obj = {}
    for name, param in model.named_parameters():
        name_to_obj[name] = param
    for name, buf in model.named_buffers():
        name_to_obj[name] = buf

    for key, value in state_dict.items():
        # Handle potential prefix mismatch if state_dict keys are different?
        # Assuming keys match named_parameters/buffers.
        obj = name_to_obj.get(key)

        if obj is not None and hasattr(obj, "sharded_dims_mp"):
            sharded_dims = obj.sharded_dims_mp
            gathered_value = value

            # Gather along dimensions
            # We iterate and gather. Note that subsequent gathers act on the grown tensor.
            for dim, group_name in enumerate(sharded_dims):
                if group_name is not None:
                    # Check if shared (replicated) on this group
                    if hasattr(obj, "is_shared_mp") and group_name in obj.is_shared_mp:
                        continue

                    gathered_value = gather_tensor(gathered_value, dim, group_name)

            new_state_dict[key] = gathered_value
        else:
            new_state_dict[key] = value

    return new_state_dict


def scatter_sharded_state_dict(full_state_dict: dict, model: nn.Module) -> dict:
    """
    Scatters/slices a full state dict into local shards.

    Args:
        full_state_dict: The full state dict.
        model: The model with annotations.

    Returns:
        Local state dict.
    """
    local_state_dict = {}

    # Build map
    name_to_obj = {}
    for name, param in model.named_parameters():
        name_to_obj[name] = param
    for name, buf in model.named_buffers():
        name_to_obj[name] = buf

    for key, value in full_state_dict.items():
        obj = name_to_obj.get(key)

        if obj is not None and hasattr(obj, "sharded_dims_mp"):
            sharded_dims = obj.sharded_dims_mp
            scattered_value = value

            # Scatter along dimensions
            for dim, group_name in enumerate(sharded_dims):
                if group_name is not None:
                    # Check if shared
                    if hasattr(obj, "is_shared_mp") and group_name in obj.is_shared_mp:
                        continue

                    scattered_value = scatter_tensor(scattered_value, dim, group_name)

            local_state_dict[key] = scattered_value
        else:
            local_state_dict[key] = value

    return local_state_dict


def gather_optimizer_state_dict(optimizer: torch.optim.Optimizer) -> dict:
    """
    Gathers sharded optimizer state (momentum, variance, etc.).
    """
    state_dict = optimizer.state_dict()

    # Create map from param_id to model parameter
    # optimizer.param_groups corresponds to state_dict['param_groups']
    id_to_param = {}

    # The order of params in optimizer groups matches init order.
    # Assuming optimizer groups structure matches saved state dict groups structure.
    for group_idx, group in enumerate(optimizer.param_groups):
        saved_group = state_dict['param_groups'][group_idx]
        for i, param in enumerate(group['params']):
            saved_id = saved_group['params'][i]
            id_to_param[saved_id] = param

    # Iterate state and gather
    new_state = {}
    for param_id, param_state in state_dict['state'].items():
        param = id_to_param.get(param_id)
        new_param_state = {}

        if param is not None and hasattr(param, "sharded_dims_mp"):
            # Gather each tensor in state (e.g. exp_avg, exp_avg_sq)
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor) and v.ndim == param.ndim:
                    # Assume optimizer state tensors have same shape/sharding as param
                    gathered_v = v
                    for dim, group_name in enumerate(param.sharded_dims_mp):
                        if group_name is not None:
                            if hasattr(param, "is_shared_mp") and group_name in param.is_shared_mp:
                                continue
                            gathered_v = gather_tensor(gathered_v, dim, group_name)
                    new_param_state[k] = gathered_v
                else:
                    # Scalar state (e.g. step) or non-matching shape
                    new_param_state[k] = v
        else:
            new_param_state = param_state

        new_state[param_id] = new_param_state

    state_dict['state'] = new_state
    return state_dict


def scatter_optimizer_state_dict(full_state_dict: dict, optimizer: torch.optim.Optimizer) -> dict:
    """
    Scatters optimizer state dict to local shards.
    """
    # Clone to avoid modifying input if necessary, but we are replacing 'state'
    state_dict = full_state_dict.copy()

    id_to_param = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        saved_group = state_dict['param_groups'][group_idx]
        for i, param in enumerate(group['params']):
            saved_id = saved_group['params'][i]
            id_to_param[saved_id] = param

    new_state = {}
    for param_id, param_state in state_dict['state'].items():
        param = id_to_param.get(param_id)
        new_param_state = {}

        if param is not None and hasattr(param, "sharded_dims_mp"):
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor) and v.ndim == param.ndim:
                    scattered_v = v
                    for dim, group_name in enumerate(param.sharded_dims_mp):
                        if group_name is not None:
                            if hasattr(param, "is_shared_mp") and group_name in param.is_shared_mp:
                                continue
                            scattered_v = scatter_tensor(scattered_v, dim, group_name)
                    new_param_state[k] = scattered_v
                else:
                    new_param_state[k] = v
        else:
            new_param_state = param_state

        new_state[param_id] = new_param_state

    state_dict['state'] = new_state
    return state_dict
