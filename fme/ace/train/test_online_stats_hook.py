"""Unit tests for the online-stats wiring in train.build_trainer."""

import dataclasses

from fme.ace.train.train import _find_online_normalization_configs
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig


@dataclasses.dataclass
class _FakeStepConfig:
    """Mimics SingleModuleStepConfig's normalization field for the walker."""

    normalization: NetworkAndLossNormalizationConfig


class _FakeSelector:
    """Mimics StepSelector's lazily-instantiated step config attribute."""

    def __init__(self, step_config: _FakeStepConfig):
        self._step_config_instance = step_config


@dataclasses.dataclass
class _FakeStepperConfig:
    step: _FakeSelector


def _make_normalization(network_online: bool, residual_online: bool):
    network = (
        NormalizationConfig(compute_online=True)
        if network_online
        else NormalizationConfig(means={"a": 0.0}, stds={"a": 1.0})
    )
    residual = (
        NormalizationConfig(compute_online=True)
        if residual_online
        else NormalizationConfig(means={"a": 0.0}, stds={"a": 1.0})
    )
    return NetworkAndLossNormalizationConfig(network=network, residual=residual)


def test_walker_finds_nested_online_configs():
    normalization = _make_normalization(network_online=True, residual_online=True)
    root = _FakeStepperConfig(
        step=_FakeSelector(_FakeStepConfig(normalization=normalization))
    )
    found = _find_online_normalization_configs(root)
    assert len(found) == 2
    assert all(c.compute_online for c in found)


def test_walker_skips_when_no_online_flag():
    normalization = _make_normalization(network_online=False, residual_online=False)
    root = _FakeStepperConfig(
        step=_FakeSelector(_FakeStepConfig(normalization=normalization))
    )
    assert _find_online_normalization_configs(root) == []


def test_walker_finds_partial_online_configs():
    normalization = _make_normalization(network_online=True, residual_online=False)
    root = _FakeStepperConfig(
        step=_FakeSelector(_FakeStepConfig(normalization=normalization))
    )
    found = _find_online_normalization_configs(root)
    assert len(found) == 1
    assert found[0].compute_online
    assert found[0] is normalization.network


def test_walker_handles_apply_stats_idempotently():
    normalization = _make_normalization(network_online=True, residual_online=True)
    root = _FakeStepperConfig(
        step=_FakeSelector(_FakeStepConfig(normalization=normalization))
    )
    pending = _find_online_normalization_configs(root)
    for cfg in pending:
        cfg.apply_stats(means={"a": 0.0}, stds={"a": 1.0})
    assert _find_online_normalization_configs(root) == []
