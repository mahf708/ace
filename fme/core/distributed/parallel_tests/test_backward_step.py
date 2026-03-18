"""System-level tests for the forward+backward training loop with distributed
spatial model parallelism.

These tests build on the existing ``test_step.py`` infrastructure and exercise
the full forward → loss → backward path using the conditional SFNO model
under spatial decomposition.

Run example::

    FME_FORCE_CPU=1 FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 \\
        FME_DISTRIBUTED_W=1 torchrun --nproc-per-node 2 -m pytest -m parallel \\
        fme/core/distributed/parallel_tests/test_backward_step.py -v
"""

import dataclasses
import datetime

import pytest
import torch

import fme
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed.distributed import Distributed
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry import ModuleSelector
from fme.core.step.args import StepArgs
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.typing_ import TensorDict

IMG_SHAPE = (20, 40)
TIMESTEP = datetime.timedelta(hours=6)
N_SAMPLES = 2


def _get_selector() -> StepSelector:
    """Small NoiseConditionedSFNO config for backward pass testing."""
    names = ["forcing_a", "forcing_b", "diagnostic_a"]
    normalization = NetworkAndLossNormalizationConfig(
        network=NormalizationConfig(
            means={n: 0.0 for n in names},
            stds={n: 1.0 for n in names},
        ),
    )
    return StepSelector(
        type="single_module",
        config=dataclasses.asdict(
            SingleModuleStepConfig(
                builder=ModuleSelector(
                    type="NoiseConditionedSFNO",
                    config=dataclasses.asdict(
                        NoiseConditionedSFNOBuilder(
                            embed_dim=4,
                            noise_embed_dim=4,
                            noise_type="gaussian",
                            filter_type="linear",
                            filter_num_groups=2,
                            num_layers=2,
                            local_blocks=[0],
                        )
                    ),
                ),
                in_names=["forcing_a", "forcing_b"],
                out_names=["diagnostic_a"],
                normalization=normalization,
            ),
        ),
    )


def _build_step(img_shape=IMG_SHAPE):
    device = fme.get_device()
    horizontal = LatLonCoordinates(
        lat=torch.zeros(img_shape[0], device=device),
        lon=torch.zeros(img_shape[1], device=device),
    )
    vertical = HybridSigmaPressureCoordinate(
        ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
    )
    dataset_info = DatasetInfo(
        horizontal_coordinates=horizontal,
        vertical_coordinate=vertical,
        timestep=TIMESTEP,
    )
    return _get_selector().get_step(dataset_info)


def _get_tensor_dict(names: list[str], img_shape, n_samples) -> TensorDict:
    device = fme.get_device()
    return {name: torch.rand(n_samples, *img_shape, device=device) for name in names}


def _run_forward_backward(step, dist):
    """Run one forward+backward pass, return (output, loss)."""
    torch.manual_seed(0)
    input_data = _get_tensor_dict(step.input_names, IMG_SHAPE, N_SAMPLES)
    next_input = _get_tensor_dict(step.next_step_input_names, IMG_SHAPE, N_SAMPLES)

    input_data = dist.scatter_spatial(input_data, IMG_SHAPE)
    next_input = dist.scatter_spatial(next_input, IMG_SHAPE)

    output = step.step(
        args=StepArgs(input=input_data, next_step_input_data=next_input, labels=None),
        wrapper=lambda x: x,
    )

    loss = sum(v.pow(2).mean() for v in output.values())
    loss.backward()
    return output, loss


@pytest.mark.parallel
def test_forward_backward_all_params_have_grad():
    """Every trainable parameter should receive a gradient after backward."""
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    step = _build_step()

    _run_forward_backward(step, dist)

    for module in step.modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Parameter {name} has requires_grad=True but grad is None"
                )
                assert not torch.all(param.grad == 0), (
                    f"Parameter {name} has all-zero gradient"
                )


@pytest.mark.parallel
def test_optimizer_step_changes_params():
    """Forward → backward → optimizer.step() should update parameters."""
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    step = _build_step()

    initial_params = {}
    for module in step.modules:
        for name, param in module.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()

    _run_forward_backward(step, dist)

    params = [
        p
        for m in step.modules
        for p in m.parameters()
        if p.requires_grad and p.grad is not None
    ]
    optimizer = torch.optim.SGD(params, lr=0.01)
    optimizer.step()

    changed = sum(
        1
        for m in step.modules
        for name, p in m.named_parameters()
        if name in initial_params and not torch.allclose(p.data, initial_params[name])
    )
    assert changed > 0, "No parameters changed after optimizer step"


@pytest.mark.parallel
def test_loss_consistent_across_spatial_ranks():
    """Loss should be identical on all spatial ranks after forward."""
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    step = _build_step()

    _, loss = _run_forward_backward(step, dist)

    loss_tensor = loss.detach().unsqueeze(0)
    gathered = dist.gather(loss_tensor)
    if dist.is_root():
        assert gathered is not None
        values = torch.stack([g.to(fme.get_device()) for g in gathered])
        for i in range(1, len(values)):
            torch.testing.assert_close(
                values[0],
                values[i],
                rtol=1e-5,
                atol=1e-6,
                msg=f"Loss mismatch between rank 0 and rank {i}",
            )


@pytest.mark.parallel
def test_gradient_accumulation():
    """Two forward+backward passes should accumulate gradients correctly."""
    dist = Distributed.get_instance()
    torch.manual_seed(0)
    step = _build_step()

    _run_forward_backward(step, dist)

    grads_one = {}
    for module in step.modules:
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads_one[name] = param.grad.clone()

    # Second pass (gradients accumulate)
    _run_forward_backward(step, dist)

    for module in step.modules:
        for name, param in module.named_parameters():
            if name in grads_one and param.grad is not None:
                torch.testing.assert_close(
                    param.grad,
                    2 * grads_one[name],
                    rtol=1e-4,
                    atol=1e-5,
                    msg=f"Gradient accumulation failed for {name}",
                )


@pytest.mark.parallel
def test_gradients_match_non_distributed_reference():
    """Distributed gradients should match single-process non-distributed reference.

    Run the same model non-distributed (full spatial input), compare gradients.
    This is the gold-standard correctness test.
    """
    dist = Distributed.get_instance()
    sp = dist.world_size // dist.total_data_parallel_ranks
    if sp == 1:
        pytest.skip("Need >1 spatial rank for distributed vs reference comparison")

    torch.manual_seed(0)
    step = _build_step()
    state = step.get_state()

    # --- Distributed forward+backward ---
    _run_forward_backward(step, dist)

    dist_grads = {}
    for module in step.modules:
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                dist_grads[name] = param.grad.clone()

    # --- Non-distributed reference ---
    step_ref = _build_step()
    step_ref.load_state(state)

    torch.manual_seed(0)
    input_data = _get_tensor_dict(step_ref.input_names, IMG_SHAPE, N_SAMPLES)
    next_input = _get_tensor_dict(step_ref.next_step_input_names, IMG_SHAPE, N_SAMPLES)

    output_ref = step_ref.step(
        args=StepArgs(input=input_data, next_step_input_data=next_input, labels=None),
        wrapper=lambda x: x,
    )
    loss_ref = sum(v.pow(2).mean() for v in output_ref.values())
    loss_ref.backward()

    ref_grads = {}
    for module in step_ref.modules:
        for name, param in module.named_parameters():
            if param.requires_grad and param.grad is not None:
                ref_grads[name] = param.grad.clone()

    for name in ref_grads:
        if name in dist_grads:
            torch.testing.assert_close(
                dist_grads[name],
                ref_grads[name],
                rtol=1e-3,
                atol=1e-4,
                msg=f"Gradient mismatch for {name}",
            )
