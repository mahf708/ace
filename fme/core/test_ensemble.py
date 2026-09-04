import dataclasses

import pytest
import torch

from fme.core.ensemble import get_crps, get_energy_score


@dataclasses.dataclass
class CRPSExperiment:
    name: str
    truth_amount: float
    random_amount: float


@pytest.mark.parametrize("n_ensemble", [2, 5])
@pytest.mark.parametrize("alpha", [1.0, 0.95])
def test_crps(n_ensemble: int, alpha: float):
    """
    Test that get_crps is a proper scoring rule.

    Scoring rules that are proper are proven to have the lowest
    expected score if the predicted distribution equals the
    underlying distribution of the target variable. Note that
    the assumptions in this test are only valid for values of
    alpha near 1.
    """
    torch.manual_seed(0)
    nx = 1
    ny = 1
    n_batch = 10000
    n_sample = n_ensemble
    truth_amount = 0.8
    random_amount = 0.5
    experiments = [
        CRPSExperiment("perfect", truth_amount, random_amount),
        CRPSExperiment("extra_variance", truth_amount, random_amount * 1.1),
        CRPSExperiment("less_variance", truth_amount, random_amount * 0.9),
        CRPSExperiment("deterministic", truth_amount, random_amount * 1e-5),
    ]
    x_predictable = torch.rand(n_batch, 1, nx, ny)
    x = truth_amount * x_predictable + random_amount * torch.rand(n_batch, 1, nx, ny)
    crps_values = {}
    for experiment in experiments:
        x_sample = (
            experiment.truth_amount * x_predictable
            + experiment.random_amount * torch.rand(n_batch, n_sample, nx, ny)
        )
        crps_values[experiment.name] = get_crps(
            gen=x_sample, target=x, alpha=alpha
        ).mean()
    assert crps_values["perfect"] < crps_values["extra_variance"]
    assert crps_values["perfect"] < crps_values["less_variance"]
    assert crps_values["extra_variance"] < crps_values["deterministic"]
    assert crps_values["less_variance"] < crps_values["deterministic"]


def _two_member_energy_score(gen: torch.Tensor, target: torch.Tensor):
    """The original two-member special case, kept as a regression anchor.

    It pulls the 0.5 out of the pairwise term because a mean over a single
    pair leaves it alone. Any generalisation must reproduce this exactly at
    two members, or every trained two-member model silently changes meaning.
    """
    target_term = torch.abs(gen - target).mean(axis=1)
    internal_term = -0.5 * torch.abs(gen[:, 0, ...] - gen[:, 1, ...])
    return target_term + internal_term


def test_energy_score_is_unchanged_at_two_members():
    """Pin the two-member value: it is what every trained model was fit to."""
    torch.manual_seed(0)
    for shape in ((16, 2, 3, 5), (4, 2, 2, 6, 7)):
        gen = torch.randn(*shape) + 1j * torch.randn(*shape)
        target_shape = (shape[0], 1, *shape[2:])
        target = torch.randn(*target_shape) + 1j * torch.randn(*target_shape)
        torch.testing.assert_close(
            get_energy_score(gen, target),
            _two_member_energy_score(gen, target),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("n_ensemble", [1, 2, 3, 5])
def test_energy_score_matches_the_unbiased_estimator(n_ensemble: int):
    """ES = E||X - y|| - 1/2 E||X - X'||, with the pairwise term unbiased.

    The unbiased (fair) estimator averages over ordered pairs i != j, which
    is the same normalisation `get_crps` uses.
    """
    torch.manual_seed(0)
    gen = torch.randn(32, n_ensemble, 4) + 1j * torch.randn(32, n_ensemble, 4)
    target = torch.randn(32, 1, 4) + 1j * torch.randn(32, 1, 4)

    target_term = (gen - target).abs().mean(dim=1)
    if n_ensemble == 1:
        expected = target_term
    else:
        ordered_pair_sum = sum(
            (gen[:, i] - gen[:, j]).abs()
            for i in range(n_ensemble)
            for j in range(n_ensemble)
            if i != j
        )
        expected = target_term - ordered_pair_sum / (2 * n_ensemble * (n_ensemble - 1))
    torch.testing.assert_close(get_energy_score(gen, target), expected)
