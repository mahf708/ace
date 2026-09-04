import dataclasses

import pytest
import torch

from fme.core.ensemble import get_crps


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


@pytest.mark.parametrize("n_ensemble", [2, 3, 5])
def test_almost_fair_crps_matches_its_definition(n_ensemble: int):
    """Almost-fair CRPS interpolates between fair and ordinary CRPS.

    From AIFS-CRPS (https://arxiv.org/abs/2412.15832), for ensemble size M:

        aCRPS(alpha) = alpha * fairCRPS + (1 - alpha) * CRPS

    Expanding that gives a pairwise coefficient of ``1 - (1 - alpha) / M``,
    i.e. ``epsilon = (1 - alpha) / M``. Writing epsilon as ``(1 - alpha) / 2``
    happens to be right at two members and is wrong everywhere else.
    """
    torch.manual_seed(0)
    alpha = 0.95
    gen = torch.randn(64, n_ensemble, 8)
    target = torch.randn(64, 1, 8)

    target_term = (gen - target).abs().mean(dim=1)
    ordered_pair_sum = sum(
        (gen[:, i] - gen[:, j]).abs()
        for i in range(n_ensemble)
        for j in range(n_ensemble)
        if i != j
    )
    fair = target_term - ordered_pair_sum / (2 * n_ensemble * (n_ensemble - 1))
    ordinary = target_term - ordered_pair_sum / (2 * n_ensemble * n_ensemble)
    expected = alpha * fair + (1.0 - alpha) * ordinary

    torch.testing.assert_close(get_crps(gen, target, alpha=alpha), expected)
