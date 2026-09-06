import math

import pytest
import torch

from fme.ace.aggregator.one_step.ensemble import (
    CRPSMetric,
    EnsembleMeanRMSEMetric,
    RankBiasMetric,
    RankDispersionMetric,
    SSRBiasMetric,
    _EnsembleAggregator,
)
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import LatLonOperations
from fme.core.typing_ import EnsembleTensorDict


def get_tensor(shape):
    return torch.randn(*shape)


def _make_ensemble_batch(names, shape=(2, 3, 1, 4, 4)):
    """Make an EnsembleTensorDict of given shape for each named variable."""
    return EnsembleTensorDict(
        {name: torch.randn(*shape, device=get_device()) for name in names}
    )


def test_crps_metric_gives_correct_shape():
    metric = CRPSMetric()
    n_batch = 10
    n_sample = 2
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)


def test_ssr_metric_gives_correct_shape():
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 10
    n_sample = 2
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_metric_unbiased(n_sample):
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    # identical distribution for target and gen
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(0.0), atol=1e-2, rtol=0.0)


@pytest.mark.parametrize("n_sample", [2, 5, 10])
def test_ssr_bias_metric_doubled_spread(n_sample):
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x)) * 2
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(1.0), atol=1e-2, rtol=0.0)


def test_ssr_finite_with_small_ensemble():
    """SSR must be finite for all grid cells, even with n_ensemble=2 and
    few batches, which previously produced NaN via sqrt of negative
    unbiased_mse."""
    torch.manual_seed(42)
    metric = SSRBiasMetric()
    n_batch = 3
    n_sample = 2
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = target + 0.01 * get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert torch.isfinite(got).all(), f"Non-finite SSR values: {got}"


def test_ssr_nontrivial_with_differing_members():
    """When ensemble members actually differ, SSR should not be -1."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 500
    n_sample = 5
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = get_tensor((n_batch, n_sample, n_time, n_y, n_x))
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert torch.isfinite(got).all()
    assert (got > -0.9).all(), f"SSR unexpectedly near -1: mean={got.mean():.3f}"


def test_ssr_identical_members_gives_negative_one():
    """When all ensemble members are identical, SSR should be -1
    (zero spread / nonzero skill)."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch = 10
    n_sample = 3
    n_time = 1
    n_y = 4
    n_x = 5
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    single_pred = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = single_pred.expand(n_batch, n_sample, n_time, n_y, n_x).clone()
    metric.record(target=target, gen=gen)
    got = metric.get()
    torch.testing.assert_close(got, torch.full_like(got, -1.0), atol=1e-6, rtol=0.0)


def test_ssr_prescribed_cell_is_zero_but_zero_spread_with_error_is_negative_one():
    """A prescribed cell (members identical and equal to the target) is a 0/0
    and reports 0, while a zero-spread cell with nonzero error still gives -1."""
    torch.manual_seed(0)
    metric = SSRBiasMetric()
    n_batch, n_sample, n_time, n_y, n_x = 10, 3, 1, 2, 2
    target = get_tensor((n_batch, 1, n_time, n_y, n_x))
    # zero spread everywhere: every ensemble member identical
    single_pred = get_tensor((n_batch, 1, n_time, n_y, n_x))
    gen = single_pred.expand(n_batch, n_sample, n_time, n_y, n_x).clone()
    # left column: members also equal the target -> zero skill too (prescribed)
    gen[..., 0] = target[..., 0]
    metric.record(target=target, gen=gen)
    got = metric.get()
    torch.testing.assert_close(
        got[..., 0], torch.zeros_like(got[..., 0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        got[..., 1], torch.full_like(got[..., 1], -1.0), atol=1e-6, rtol=0.0
    )


def test_aggregator_ssr_bias_prescribed_cells_do_not_pull_scalar_to_negative_one():
    """Prescribed cells contribute 0, not the -1 floor, so a mostly-prescribed
    field is not dragged toward -1. With uniform weights the scalar equals the
    plain mean of the per-cell field."""
    torch.manual_seed(0)
    n_batch, n_sample, n_time, n_y, n_x = 50, 4, 1, 2, 4
    area_weights = torch.ones([n_y, n_x], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="denorm",
    )
    target = torch.randn(n_batch, 1, n_time, n_y, n_x, device=get_device())
    gen = torch.randn(n_batch, n_sample, n_time, n_y, n_x, device=get_device())
    # left half of the grid prescribed: every member equals the target
    gen[..., :2] = target[..., :2]
    agg.record_batch(
        target_data=EnsembleTensorDict({"a": target}),
        gen_data=EnsembleTensorDict({"a": gen}),
    )
    scalar = float(agg.get_logs(label="metrics")["metrics/ssr_bias/a"])
    assert math.isfinite(scalar)
    # not pinned to the -1 floor by the prescribed half
    assert scalar > -0.5, scalar

    # with uniform weights the scalar is the plain mean of the per-cell field,
    # where the prescribed cells contribute 0 (not -1)
    field = SSRBiasMetric()
    field.record(target=target, gen=gen)
    expected = float(field.get().mean())
    assert math.isclose(scalar, expected, rel_tol=1e-5, abs_tol=1e-5), (
        scalar,
        expected,
    )


@pytest.mark.parametrize("n_sample", [2, 10])
def test_ensemble_mean_metric(n_sample):
    # this simple check only works for even n_sample
    torch.manual_seed(0)
    metric = EnsembleMeanRMSEMetric()
    n_batch = 5000
    n_time = 3
    n_y = 4
    n_x = 5
    target = torch.ones((n_batch, 1, n_time, n_y, n_x))
    gen = torch.ones((n_batch, n_sample, n_time, n_y, n_x)) * 2
    # half the samples are 0
    gen[:, : n_sample // 2, ...] = 0
    metric.record(target=target, gen=gen)
    got = metric.get()
    assert isinstance(got, torch.Tensor)
    assert got.shape == (n_y, n_x)
    torch.testing.assert_close(got.mean(), torch.tensor(0.0), atol=1e-2, rtol=0.0)


def test_aggregator_denorm_does_not_log_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="denorm",
    )
    names = ["a", "b"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(target_data=target, gen_data=gen)
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/a" in logs
        assert f"metrics/{metric}/b" in logs
        assert f"metrics/{metric}/channel_mean" not in logs


def test_aggregator_norm_requires_norm_data():
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
    )
    target = _make_ensemble_batch(["a"])
    gen = _make_ensemble_batch(["a"])
    with pytest.raises(ValueError, match="target_data_norm and gen_data_norm"):
        agg.record_batch(target_data=target, gen_data=gen)


def test_aggregator_norm_logs_channel_mean_all_variables():
    """When channel_mean_names is None and target='norm', channel mean is
    computed over all variables."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
    )
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/channel_mean" in logs
        expected = sum(logs[f"metrics/{metric}/{n}"] for n in names) / len(names)
        torch.testing.assert_close(
            torch.tensor(float(logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(expected)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_norm_channel_mean_excludes_all_nan_target():
    """A variable whose target is entirely NaN (e.g. filled by
    allow_missing_variables) is excluded from the channel mean."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
    )
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    # "c" is missing from the data: entirely-NaN target.
    target = EnsembleTensorDict(
        {**target, "c": torch.full_like(target["c"], torch.nan)}
    )
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ensemble_mean_rmse"):
        channel_mean = float(logs[f"metrics/{metric}/channel_mean"])
        assert not math.isnan(channel_mean)
        # equals the mean of the two valid-target channels (a, b); "c" excluded.
        expected = (logs[f"metrics/{metric}/a"] + logs[f"metrics/{metric}/b"]) / 2
        torch.testing.assert_close(
            torch.tensor(channel_mean),
            torch.tensor(float(expected)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_norm_logs_channel_mean_subset():
    """channel_mean_names restricts the channel mean to the listed variables."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=["a", "c"],
    )
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    logs = agg.get_logs(label="metrics")
    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        expected = (logs[f"metrics/{metric}/a"] + logs[f"metrics/{metric}/c"]) / 2.0
        torch.testing.assert_close(
            torch.tensor(float(logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(expected)),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_report_variables_filters_per_variable_but_keeps_channel_mean():
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    names = ["a", "b", "c"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)

    agg_full = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=names,
    )
    agg_full.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    full_logs = agg_full.get_logs(label="metrics")

    agg_filtered = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=names,
        report_variables=["a"],
    )
    agg_filtered.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    filtered_logs = agg_filtered.get_logs(label="metrics")

    for metric in ("crps", "ssr_bias", "ensemble_mean_rmse"):
        assert f"metrics/{metric}/a" in filtered_logs
        assert f"metrics/{metric}/b" not in filtered_logs
        assert f"metrics/{metric}/c" not in filtered_logs
        assert f"metrics/{metric}/channel_mean" in filtered_logs
        torch.testing.assert_close(
            torch.tensor(float(filtered_logs[f"metrics/{metric}/channel_mean"])),
            torch.tensor(float(full_logs[f"metrics/{metric}/channel_mean"])),
            atol=1e-6,
            rtol=1e-6,
        )


def test_aggregator_norm_raises_on_unknown_channel_mean_names():
    """Names in channel_mean_names that aren't in the data raise KeyError."""
    torch.manual_seed(0)
    area_weights = torch.ones([4, 4], device=get_device())
    agg = _EnsembleAggregator(
        gridded_operations=LatLonOperations(area_weights),
        log_mean_maps=False,
        target="norm",
        channel_mean_names=["a", "not_present"],
    )
    names = ["a", "b"]
    target = _make_ensemble_batch(names)
    gen = _make_ensemble_batch(names)
    agg.record_batch(
        target_data=target,
        gen_data=gen,
        target_data_norm=target,
        gen_data_norm=gen,
    )
    with pytest.raises(KeyError, match="not_present"):
        agg.get_logs(label="metrics")


def _record_calibrated(metric, n_sample, n_batch=400, shape=(1, 3, 3), seed=0):
    """Draw target and members from the same distribution, which is what
    calibration means: the target is exchangeable with the members."""
    torch.manual_seed(seed)
    n_time, n_y, n_x = shape
    both = torch.randn(n_batch, n_sample + 1, n_time, n_y, n_x)
    metric.record(target=both[:, :1], gen=both[:, 1:])
    return metric.get()


@pytest.mark.parametrize("n_sample", [2, 4, 10])
def test_rank_metrics_near_zero_on_a_calibrated_ensemble(n_sample):
    """An ensemble drawn from the truth's own distribution is calibrated by
    construction, so both rank statistics sit at zero up to sampling noise."""
    bias = _record_calibrated(RankBiasMetric(), n_sample)
    dispersion = _record_calibrated(RankDispersionMetric(), n_sample)
    assert bias.shape == (3, 3)
    assert bias.abs().mean() < 0.05
    assert dispersion.abs().mean() < 0.2


def test_rank_dispersion_is_not_pinned_by_one_sample_per_rank():
    """The scores pass gives each rank one initial condition at one lead, so a
    variance formed per rank is a variance over a single sample: identically
    zero, pinning the statistic at its -1 floor. The moments are pooled before
    the variance is formed, so a calibrated ensemble spread over eight such
    single-sample records still reads near zero."""
    torch.manual_seed(0)
    metric = RankDispersionMetric()
    for _ in range(8):
        both = torch.randn(1, 5, 1, 8, 8)
        metric.record(target=both[:, :1], gen=both[:, 1:])
    got = metric.get()
    assert torch.isfinite(got).all()
    assert got.mean().abs() < 0.3, got.mean()


def test_rank_dispersion_is_undefined_below_two_samples():
    """One sample carries no variance at all, and reporting the floor there
    would read as a fully collapsed ensemble."""
    metric = RankDispersionMetric()
    both = torch.randn(1, 5, 1, 2, 2)
    metric.record(target=both[:, :1], gen=both[:, 1:])
    assert torch.isnan(metric.get()).all()


@pytest.mark.parametrize("n_sample", [2, 4, 10])
def test_rank_dispersion_reference_is_the_discrete_uniform(n_sample):
    """The calibrated rank variance is the discrete uniform's,
    (1 - (M+1)^-2)/12, not the continuous 1/12; at four members they differ by
    4%, which is the size of the effects being looked for.

    Pinned on an exactly flat histogram of M+1 samples, one per rank. Its
    *population* variance is the reference exactly, so the unbiased estimator
    -- which targets the population value from a finite draw -- returns the
    reference times (M+1)/M, and the statistic is 1/M. Getting the reference
    wrong moves that: at M=4 the continuous 1/12 would give 0.200, not 0.250.
    """
    metric = RankDispersionMetric()
    n_y, n_x = 2, 2
    members = torch.arange(1.0, n_sample + 1).reshape(1, n_sample, 1, 1, 1)
    gen = members.expand(n_sample + 1, n_sample, 1, n_y, n_x).contiguous()
    # target below all members, between each pair, then above all
    targets = torch.arange(0.5, n_sample + 1).reshape(n_sample + 1, 1, 1, 1, 1)
    metric.record(target=targets.expand(-1, 1, 1, n_y, n_x), gen=gen)
    expected = torch.full((n_y, n_x), 1.0 / n_sample)
    assert torch.allclose(metric.get(), expected, atol=1e-6), metric.get()


def test_rank_dispersion_positive_when_underdispersed():
    """Members tightly clustered around a centre that is itself far from the
    truth: the ensemble is narrow relative to its own error, so the target
    lands outside it and fills the ends of the histogram. This is the U shape
    a collapsing stochastic model produces."""
    torch.manual_seed(0)
    n_batch, n_sample = 500, 4
    target = torch.randn(n_batch, 1, 1, 2, 2)
    centre = target + torch.randn(n_batch, 1, 1, 2, 2)
    gen = centre + 0.1 * torch.randn(n_batch, n_sample, 1, 2, 2)
    metric = RankDispersionMetric()
    metric.record(target=target, gen=gen)
    assert metric.get().mean() > 0.5


def test_rank_dispersion_negative_when_overdispersed():
    """Members scattered around the truth itself: the spread is wider than
    the error of the mean they form, so the target sits inside the ensemble
    and the histogram is a dome."""
    torch.manual_seed(0)
    n_batch, n_sample = 500, 4
    target = torch.randn(n_batch, 1, 1, 2, 2)
    gen = target + torch.randn(n_batch, n_sample, 1, 2, 2)
    metric = RankDispersionMetric()
    metric.record(target=target, gen=gen)
    assert metric.get().mean() < -0.3


def test_rank_dispersion_reaches_minus_one_when_the_target_is_always_centred():
    """Deterministic members straddling the target put its rank at the centre
    every time, so the rank variance is zero: the floor of the statistic."""
    n_batch, n_y, n_x = 6, 2, 2
    target = torch.randn(n_batch, 1, 1, n_y, n_x)
    offsets = torch.tensor([-2.0, -1.0, 1.0, 2.0]).reshape(1, 4, 1, 1, 1)
    metric = RankDispersionMetric()
    metric.record(target=target, gen=target + offsets)
    assert torch.allclose(metric.get(), -torch.ones(n_y, n_x), atol=1e-6)


def test_rank_bias_sees_an_offset_ensemble_that_ssr_bias_can_miss():
    """Shift every member up and the ensemble is systematically above the
    truth. The rank statistic reports the sign; the spread-skill ratio, being
    a ratio of magnitudes, does not distinguish the direction."""
    torch.manual_seed(0)
    n_batch, n_sample = 400, 4
    target = torch.randn(n_batch, 1, 1, 2, 2)
    gen = target + 1.0 + torch.randn(n_batch, n_sample, 1, 2, 2)
    metric = RankBiasMetric()
    metric.record(target=target, gen=gen)
    # members above the target push its rank down, so the statistic is negative
    assert metric.get().mean() < -0.1


def test_rank_metrics_report_zero_on_a_prescribed_cell():
    """Every member equal to the target is a genuine 0/0, not a collapse: it
    must not drag the field mean to the under-dispersed extreme, the same
    convention ssr_bias already follows."""
    n_batch, n_sample = 8, 4
    target = torch.randn(n_batch, 1, 1, 2, 2)
    gen = target.expand(n_batch, n_sample, 1, 2, 2).contiguous()
    for metric in (RankBiasMetric(), RankDispersionMetric()):
        metric.record(target=target, gen=gen)
        assert torch.allclose(metric.get(), torch.zeros(2, 2))


def test_rank_metrics_accumulate_across_batches():
    """The moments are running sums, so one call with 2N samples and two with
    N must agree."""
    torch.manual_seed(0)
    target = torch.randn(20, 1, 1, 2, 2)
    gen = torch.randn(20, 4, 1, 2, 2)
    whole = RankDispersionMetric()
    whole.record(target=target, gen=gen)
    split = RankDispersionMetric()
    split.record(target=target[:10], gen=gen[:10])
    split.record(target=target[10:], gen=gen[10:])
    assert torch.allclose(whole.get(), split.get(), atol=1e-5)


def test_aggregator_reports_the_rank_metrics():
    """They reach the logs and the dataset alongside crps and ssr_bias, which
    is what makes them readable offline."""
    aggregator = _EnsembleAggregator(
        gridded_operations=LatLonOperations(torch.ones(4, 4).to(get_device()))
    )
    aggregator.record_batch(
        target_data=_make_ensemble_batch(["a"], shape=(2, 1, 1, 4, 4)),
        gen_data=_make_ensemble_batch(["a"], shape=(2, 3, 1, 4, 4)),
    )
    logs = aggregator.get_logs(label="test")
    assert "test/rank_bias/a" in logs
    assert "test/rank_dispersion/a" in logs
    assert "rank_dispersion-a" in aggregator.get_dataset().data_vars


def test_rank_metrics_are_nan_where_the_target_is_missing():
    """Comparisons against NaN are all False, which would put the rank at the
    bottom of the histogram and read as a strong calibration signal. A missing
    target has no rank, and the metric says so rather than inventing one."""
    target = torch.randn(4, 1, 1, 2, 2)
    target[:, :, :, 0, 0] = float("nan")
    gen = torch.randn(4, 3, 1, 2, 2)
    for metric in (RankBiasMetric(), RankDispersionMetric()):
        metric.record(target=target, gen=gen)
        got = metric.get()
        assert torch.isnan(got[0, 0])
        assert torch.isfinite(got[1, 1])


@pytest.mark.parallel
def test_rank_dispersion_pools_across_data_parallel_ranks():
    """Each rank holds one initial condition, which is the shape the scores
    pass runs in, and the pooled variance must be the variance of the union of
    what the ranks saw rather than a mean of per-rank zeros.

    Ranks record disjoint halves of a fixed set of ranks-in-the-histogram, so a
    per-rank variance is far from the pooled one and averaging them cannot
    recover it. Run under torchrun; also pins that the collective inside get()
    is reached by every rank.
    """
    dist = Distributed.get_instance()
    rank = dist.data_parallel_rank
    world_size = dist.total_data_parallel_ranks
    n_sample, n_y, n_x = 3, 2, 2
    members = torch.arange(1.0, n_sample + 1, device=get_device()).reshape(
        1, n_sample, 1, 1, 1
    )
    # rank r takes the (r mod n_sample+1)-th position: below all members,
    # between each pair, or above all
    position = 0.5 + (rank % (n_sample + 1))
    target = torch.full(
        (1, 1, 1, n_y, n_x), float(position), device=get_device()
    )
    metric = RankDispersionMetric()
    metric.record(target=target, gen=members.expand(1, n_sample, 1, n_y, n_x))
    got = metric.get()
    if world_size == 1:
        assert torch.isnan(got).all()
        return
    # a per-rank variance is exactly zero here; the pooled one is not
    assert torch.isfinite(got).all()
    assert (got > -1.0 + 1e-6).all(), got
