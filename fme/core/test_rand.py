import numpy as np
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.rand import (
    randn_like,
    randn_spatial,
    set_seed,
    use_global_random_state,
)


def test_set_seed_np_rand():
    set_seed(0)
    a = np.random.randn(10)
    set_seed(0)
    b = np.random.randn(10)
    assert np.allclose(a, b)


def test_set_seed_torch_rand():
    device = get_device()
    set_seed(0)
    a = torch.randn(10, device=device)
    set_seed(0)
    b = torch.randn(10, device=device)
    assert torch.allclose(a, b)


def test_set_distributed_shuffler():
    dist = Distributed.get_instance()
    set_seed(0)
    dataset = torch.utils.data.TensorDataset(torch.randn(10))
    sampler = dist.get_sampler(dataset, shuffle=True)
    first_results = list(sampler)
    set_seed(0)
    dist = Distributed.get_instance()
    sampler = dist.get_sampler(dataset, shuffle=True)
    second_results = list(sampler)
    assert torch.allclose(
        torch.as_tensor(first_results), torch.as_tensor(second_results)
    )


def test_set_random_sampler():
    dataset = torch.utils.data.TensorDataset(torch.randn(10))
    set_seed(0)
    sampler = torch.utils.data.RandomSampler(dataset)
    first_results = list(sampler)
    set_seed(0)
    sampler = torch.utils.data.RandomSampler(dataset)
    second_results = list(sampler)
    assert torch.allclose(
        torch.as_tensor(first_results), torch.as_tensor(second_results)
    )


def test_use_global_random_state_randn_like():
    """randn_like under use_global_random_state produces results
    that match the corresponding slice of the full global tensor."""
    img_shape = (8, 16)
    set_seed(42)
    # Generate the full global reference
    ref = torch.randn(2, 3, img_shape[0], img_shape[1])

    set_seed(42)
    # Non-distributed: local slices are slice(None), so result should match exactly
    with use_global_random_state(img_shape):
        local_input = torch.zeros(2, 3, img_shape[0], img_shape[1])
        result = randn_like(local_input)

    assert torch.allclose(ref, result)


def test_use_global_random_state_randn_spatial():
    """randn_spatial under use_global_random_state produces results
    that match the corresponding slice of the full global tensor."""
    img_shape = (8, 16)
    set_seed(42)
    ref = torch.randn(2, 3, img_shape[0], img_shape[1])

    set_seed(42)
    with use_global_random_state(img_shape):
        result = randn_spatial([2, 3, img_shape[0], img_shape[1]])

    assert torch.allclose(ref, result)


def test_use_global_random_state_is_noop_when_not_active():
    """Without the context manager, randn_like behaves normally."""
    set_seed(42)
    x = torch.zeros(2, 3, 4, 4)
    a = randn_like(x)
    set_seed(42)
    b = randn_like(x)
    assert torch.allclose(a, b)


def test_use_global_random_state_context_restores():
    """The context manager properly restores the previous state."""
    from fme.core import rand

    assert rand._GLOBAL_IMG_SHAPE is None
    with use_global_random_state((8, 16)):
        assert rand._GLOBAL_IMG_SHAPE == (8, 16)
    assert rand._GLOBAL_IMG_SHAPE is None
