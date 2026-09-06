import abc
import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
import xarray as xr

from fme.ace.aggregator.plotting import plot_paneled_data
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.ensemble import get_crps
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import EnsembleTensorDict, TensorMapping

from ..inference.build_context import MetricBuildContext, MetricNotSupportedError
from ..inference.data import MetricBuildResult
from .build_context import OneStepBuildContext, OneStepMetricBuildResult

# A zero-spread cell with unbiased MSE below this fraction of the field's
# largest is treated as prescribed (a 0/0): the rounding residue from
# averaging identical members sits far below this, genuine skill far above.
_PRESCRIBED_MSE_RTOL = 1e-6


def get_gen_shape(gen_data: TensorMapping):
    for name in gen_data:
        return gen_data[name].shape


def get_one_step_ensemble_aggregator(
    gridded_operations: GriddedOperations,
    target_time: int = 1,
    log_mean_maps: bool = True,
    metadata: Mapping[str, VariableMetadata] | None = None,
    target: Literal["norm", "denorm"] = "denorm",
    channel_mean_names: Sequence[str] | None = None,
    report_variables: Sequence[str] | None = None,
) -> "SelectStepEnsembleAggregator":
    return SelectStepEnsembleAggregator(
        aggregator=_EnsembleAggregator(
            gridded_operations=gridded_operations,
            log_mean_maps=log_mean_maps,
            metadata=metadata,
            target=target,
            channel_mean_names=channel_mean_names,
            report_variables=report_variables,
        ),
        i_target_time=target_time,
    )


class ReducedMetric(abc.ABC):
    """
    Used to record a metric value on batches of data (potentially out-of-memory)
    and then get the final metric at the end.
    """

    @abc.abstractmethod
    def record(self, target: torch.Tensor, gen: torch.Tensor):
        """
        Update metric for a batch of data.
        """
        ...

    @abc.abstractmethod
    def get(self) -> torch.Tensor:
        """
        Get the final metric value.
        """
        ...


class CRPSMetric(ReducedMetric):
    def __init__(self):
        self._total = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        crps = get_crps(gen=gen, target=target, alpha=0.95).mean(dim=(0, 1))
        if self._total is None:
            self._total = crps
        else:
            self._total += crps
        self._n_batches += 1

    def get(self) -> torch.Tensor:
        if self._total is None:
            raise ValueError("No batches have been recorded.")
        return self._total / self._n_batches


class EnsembleMeanRMSEMetric(ReducedMetric):
    """
    Computes the ensemble mean RMSE.
    """

    def __init__(self):
        self._total_rmse = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        ensemble_mean = gen.mean(dim=1, keepdim=True)  # mean over ensemble dimension
        rmse = ((ensemble_mean - target) ** 2).mean(dim=(0, 1, 2)).sqrt()
        if self._total_rmse is None:
            self._total_rmse = rmse
        else:
            self._total_rmse += rmse
        self._n_batches += 1

    def get(self) -> torch.Tensor:
        if self._total_rmse is None:
            raise ValueError("No batches have been recorded.")
        return self._total_rmse / self._n_batches


class SSRBiasMetric(ReducedMetric):
    """
    Computes the spread-skill ratio bias (equal to (stdev / rmse) - 1).
    """

    def __init__(self):
        self._total_unbiased_mse = None
        self._total_variance = None
        self._n_batches = 0

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        num_ensemble = gen.shape[1]
        ensemble_mean = gen.mean(dim=1, keepdim=True)  # batch, 1, time
        mse = ((ensemble_mean - target) ** 2).mean(dim=(0, 1, 2))  # batch, 1, time
        variance = gen.var(dim=1, unbiased=True).mean(dim=(0, 1))
        self._add_unbiased_mse(mse, variance, num_ensemble)
        self._add_variance(variance)
        self._n_batches += 1

    def _add_unbiased_mse(
        self, mse: torch.Tensor, variance: torch.Tensor, num_ensemble: int
    ):
        if self._total_unbiased_mse is None:
            self._total_unbiased_mse = torch.zeros_like(mse)
        # must remove the component of the MSE that is due to the
        # variance of the generated values
        self._total_unbiased_mse += mse - variance / num_ensemble

    def _add_variance(self, variance: torch.Tensor):
        if self._total_variance is None:
            self._total_variance = torch.zeros_like(variance)
        self._total_variance += variance

    def get(self) -> torch.Tensor:
        if self._total_unbiased_mse is None or self._total_variance is None:
            raise ValueError("No batches have been recorded.")
        spread = self._total_variance.sqrt()
        # Clamp before sqrt: the unbiased-MSE correction (mse - variance/n)
        # can go slightly negative with small ensembles without meaning spread
        # truly exceeds skill.
        skill = torch.clamp(self._total_unbiased_mse, min=0.0).sqrt()
        # Zero skill with nonzero spread is undefined; use -1 by convention
        # (also the limit of spread / skill - 1 as spread -> 0 at nonzero skill).
        ssr_bias = torch.where(
            skill > 0, spread / skill - 1, torch.full_like(spread, -1.0)
        )
        # Prescribed cells (every member equals the target, e.g. SST over ocean)
        # are a genuine 0/0: zero spread and zero error. Report 0 (perfectly
        # calibrated) rather than the -1 floor, which would otherwise dominate
        # the global mean of a mostly-prescribed field. Variance is exactly 0
        # there, but the unbiased MSE carries a tiny rounding residue, so test
        # it against a small fraction of the field's largest MSE, not against 0.
        mse_floor = _PRESCRIBED_MSE_RTOL * skill.square().max()
        prescribed = (self._total_variance == 0) & (
            self._total_unbiased_mse <= mse_floor
        )
        return torch.where(prescribed, torch.zeros_like(spread), ssr_bias)


class _RankHistogramMetric(ReducedMetric):
    """
    Accumulates the first two moments of the ensemble's rank histogram.

    If an ensemble is calibrated, the target is exchangeable with its members,
    so the target's rank among ``M`` of them is uniform on ``{0..M}``. Three
    departures from that uniformity are worth naming -- the histogram's slope,
    its width, and the weight in its two end bins -- and the subclasses below
    report one each.

    The end bins deserve their own statistic rather than being left to the
    variance: they are the only part of the histogram that says the truth was
    *outside* the ensemble at all, which is what an extremes claim needs. They
    are not better estimated than the variance -- measured, they are noisier --
    only more directly interpretable.

    This is a distribution-free statement, which is why it is worth having
    next to ``ssr_bias``. The spread-skill ratio compares RMS spread to RMS
    error: a second-moment claim, dominated by the largest-error cells, and
    satisfiable by an ensemble of the wrong shape as long as its width is
    right. The rank histogram is not -- an ensemble that is too narrow in the
    core and too wide in the tails is flagged here and can pass there.

    Ties take the mid-rank, so a cell whose members all equal the target lands
    at the centre of the histogram rather than at an edge.
    """

    def __init__(self):
        self._sum_u: torch.Tensor | None = None
        self._sum_u_squared: torch.Tensor | None = None
        self._n_outside: torch.Tensor | None = None
        self._n_degenerate: torch.Tensor | None = None
        self._n_samples = 0
        self._n_ensemble: int | None = None

    def record(self, target: torch.Tensor, gen: torch.Tensor):
        n_ensemble = gen.shape[1]
        if self._n_ensemble is None:
            self._n_ensemble = n_ensemble
        elif self._n_ensemble != n_ensemble:
            raise ValueError(
                "rank histogram needs a fixed ensemble size, got "
                f"{n_ensemble} after {self._n_ensemble}."
            )
        below = (gen < target).sum(dim=1)
        tied = (gen == target).sum(dim=1)
        # Normalised mid-rank in (0, 1): under calibration its mean is 1/2.
        u = (below + 0.5 * tied + 0.5) / (n_ensemble + 1)
        # Comparisons against NaN are all False, which would put the rank at
        # the bottom of the histogram and read as a strong calibration signal.
        # A variable whose target is missing has no rank; say so, the way the
        # other metrics do by propagating the NaN through their means.
        missing = torch.isnan(target).any(dim=1) | torch.isnan(gen).any(dim=1)
        u = torch.where(missing, torch.full_like(u, float("nan")), u)
        highest = gen.amax(dim=1)
        # A prescribed cell (every member equal to the target, e.g. SST over
        # ocean) is a genuine 0/0 rather than a collapsed ensemble.
        degenerate = (highest == gen.amin(dim=1)) & (highest == target.amax(dim=1))
        # The end bins: the target ranked below every member or above every
        # one. Ties take the mid-rank as everywhere else, so a prescribed cell
        # sits at the centre and is not counted an outlier.
        rank = below + 0.5 * tied
        outside = ((rank == 0) | (rank == n_ensemble)) & ~missing
        if self._sum_u is None:
            self._sum_u = torch.zeros_like(u[0, 0])
            self._sum_u_squared = torch.zeros_like(u[0, 0])
            self._n_outside = torch.zeros_like(u[0, 0])
            self._n_degenerate = torch.zeros_like(u[0, 0])
        self._sum_u += u.sum(dim=(0, 1))
        self._sum_u_squared += (u * u).sum(dim=(0, 1))
        self._n_outside += outside.sum(dim=(0, 1))
        self._n_degenerate += degenerate.sum(dim=(0, 1))
        self._n_samples += u.shape[0] * u.shape[1]

    def _moments(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pooled mean and unbiased variance of the normalised rank.

        The reduction happens here rather than in the caller because the
        variance has to be formed *after* pooling. This aggregator scores one
        lead time, and the divisibility guard gives each rank an equal share of
        the initial conditions -- often exactly one. A variance taken per rank
        is then a variance over a single sample: identically zero, so the
        statistic pins at its -1 floor and says nothing. Pooling the two
        moments first and forming the variance from them is the whole point.

        ``reduce_mean`` is an all-reduce, so every rank ends with the same
        value and the caller's own reduction is idempotent. Equal counts per
        rank are what make a mean of per-rank means the pooled mean, and
        ``InferenceEvaluatorConfig`` refuses a shape that would break that.
        """
        if self._sum_u is None or self._n_samples == 0:
            raise ValueError("No batches have been recorded.")
        assert self._sum_u_squared is not None and self._n_degenerate is not None
        dist = Distributed.get_instance()
        n_local = self._n_samples
        mean = dist.reduce_mean(self._sum_u / n_local)
        second = dist.reduce_mean(self._sum_u_squared / n_local)
        degenerate_fraction = dist.reduce_mean(self._n_degenerate / n_local)
        n_total = n_local * dist.total_data_parallel_ranks
        if n_total < 2:
            # No spread to estimate from; say so rather than report the floor.
            variance = torch.full_like(mean, float("nan"))
        else:
            # Unbiased: with eight initial conditions the n/(n-1) factor is
            # 14%, far larger than the departures from calibration being
            # looked for, so the biased estimator would read as
            # under-dispersion everywhere.
            variance = (second - mean * mean) * (n_total / (n_total - 1))
        return mean, variance, degenerate_fraction == 1.0

    def _outlier_rate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pooled observed and calibrated fraction of samples outside the
        ensemble. Pooled here for the reason given in ``_moments``.
        """
        if self._sum_u is None or self._n_samples == 0:
            raise ValueError("No batches have been recorded.")
        assert self._n_outside is not None and self._n_degenerate is not None
        dist = Distributed.get_instance()
        observed = dist.reduce_mean(self._n_outside / self._n_samples)
        degenerate_fraction = dist.reduce_mean(self._n_degenerate / self._n_samples)
        assert self._n_ensemble is not None
        # two end bins out of the M+1 a calibrated rank is uniform over
        expected = 2.0 / (self._n_ensemble + 1)
        return (
            torch.where(
                degenerate_fraction == 1.0,
                torch.full_like(observed, expected),
                observed,
            ),
            torch.full_like(observed, expected),
        )


class RankOutlierRateMetric(_RankHistogramMetric):
    """
    How often the truth falls outside the ensemble entirely, against how often
    it should, as a fraction.

    Positive means the ensemble misses the truth more often than a calibrated
    one would -- too narrow in the tails, which is the failure a small ensemble
    shows first and the one that matters for extremes. Negative means the truth
    is bracketed more often than it should be.

    This is the two end bins of the rank histogram. It is not a more precise
    statistic than the variance -- MEASURED over repeated eight-sample draws at
    four members, its spread across trials is 0.105 against the variance's
    0.070, so it is the noisier of the two. It is a more *direct* one: the
    variance describes the shape of a histogram, while this counts how often
    the ensemble simply did not contain the truth, which is the quantity an
    extremes claim rests on. At ``M`` members a calibrated ensemble misses
    ``2/(M+1)`` of the time, 40% at the four this campaign runs.
    """

    def get(self) -> torch.Tensor:
        observed, expected = self._outlier_rate()
        return observed / expected - 1.0


class RankBiasMetric(_RankHistogramMetric):
    """
    Mean normalised rank minus 1/2: where the ensemble sits against the truth.

    Positive means the target falls above the members more often than it
    should, so the ensemble is biased low; negative is the reverse. It is the
    sloped rank histogram, and it is a bias in the ensemble's location, which
    no amount of rescaling the spread will fix.
    """

    def get(self) -> torch.Tensor:
        mean, _, always_degenerate = self._moments()
        return torch.where(always_degenerate, torch.zeros_like(mean), mean - 0.5)


class RankDispersionMetric(_RankHistogramMetric):
    """
    Rank variance against its calibrated value, as a fraction: the U or the
    dome in the rank histogram.

    Positive means the target lands near the edges of the ensemble too often
    -- the ensemble is under-dispersed, the failure a collapsing stochastic
    model shows. Negative means it lands in the middle too often, so the
    ensemble is over-dispersed.

    The reference is the variance of a discrete uniform rank, ``(1 - (M+1)^-2)
    / 12``, not the continuous 1/12. At the four members this campaign runs
    those differ by 4%, which is the size of the effects being looked for.
    """

    def get(self) -> torch.Tensor:
        _, variance, always_degenerate = self._moments()
        assert self._n_ensemble is not None
        reference = (1.0 - (self._n_ensemble + 1) ** -2.0) / 12.0
        return torch.where(
            always_degenerate,
            torch.zeros_like(variance),
            variance / reference - 1.0,
        )


class _EnsembleAggregator:
    """
    Aggregator for ensemble-based metrics.
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        log_mean_maps: bool = True,
        metadata: Mapping[str, VariableMetadata] | None = None,
        target: Literal["norm", "denorm"] = "denorm",
        channel_mean_names: Sequence[str] | None = None,
        report_variables: Sequence[str] | None = None,
    ):
        """
        Args:
            gridded_operations: Gridded operations to use.
            log_mean_maps: Whether to log mean maps.
            metadata: Mapping of variable names their metadata that will
                used in generating logged image captions.
            target: Whether to compute metrics on normalized ("norm") or
                denormalized ("denorm") data. Channel-mean metrics are only
                logged when target is "norm", since averaging metrics across
                variables with different physical units is not meaningful.
            channel_mean_names: Names of variables to include in channel-mean
                metrics. If None and target is "norm", the channel mean is
                computed over all variables present in the data. Names that
                are not present in the data raise KeyError. Ignored when
                target is "denorm".
            report_variables: If set, only per-variable entries for these
                variables will appear in logs and datasets. Aggregate entries
                like ``channel_mean`` are always included. All variables are
                still used for channel-mean computation.
        """
        self._gridded_operations = gridded_operations
        self._n_batches = 0
        self._variable_metrics: dict[str, dict[str, ReducedMetric]] | None = None
        self._dist = Distributed.get_instance()
        self._log_mean_maps = log_mean_maps
        self._metadata = metadata
        self._diverging_metrics = {
            "ssr_bias",
            "rank_bias",
            "rank_dispersion",
            "rank_outlier_rate",
        }
        self._target = target
        self._channel_mean_names = channel_mean_names
        self._report_variables = (
            frozenset(report_variables) if report_variables is not None else None
        )
        # Variables whose target is entirely NaN (e.g. filled by
        # allow_missing_variables); detected once on the first batch since
        # missingness is constant, then excluded from the channel mean.
        self._all_nan_target_names: set[str] | None = None

    def _get_variable_metrics(self, gen_data: TensorMapping):
        if self._variable_metrics is None:
            self._variable_metrics = {
                "crps": {},
                "ssr_bias": {},
                "ensemble_mean_rmse": {},
                "rank_bias": {},
                "rank_dispersion": {},
                "rank_outlier_rate": {},
            }
            for key in gen_data:
                self._variable_metrics["crps"][key] = CRPSMetric()
                self._variable_metrics["ssr_bias"][key] = SSRBiasMetric()
                self._variable_metrics["ensemble_mean_rmse"][key] = (
                    EnsembleMeanRMSEMetric()
                )
                self._variable_metrics["rank_bias"][key] = RankBiasMetric()
                self._variable_metrics["rank_dispersion"][key] = RankDispersionMetric()
                self._variable_metrics["rank_outlier_rate"][key] = (
                    RankOutlierRateMetric()
                )
        return self._variable_metrics

    @torch.no_grad()
    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
        target_data_norm: EnsembleTensorDict | None = None,
        gen_data_norm: EnsembleTensorDict | None = None,
    ):
        """
        Record a batch of data.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
            target_data_norm: Normalized target data. Required when target is
                "norm".
            gen_data_norm: Normalized generated data. Required when target is
                "norm".
        """
        if self._target == "norm":
            if target_data_norm is None or gen_data_norm is None:
                raise ValueError(
                    "target_data_norm and gen_data_norm must be provided "
                    "when target is 'norm'."
                )
            target_data = target_data_norm
            gen_data = gen_data_norm
        variable_metrics = self._get_variable_metrics(gen_data)
        for metric in variable_metrics:
            for name in gen_data:
                variable_metrics[metric][name].record(
                    target=target_data[name],
                    gen=gen_data[name],
                )
        if self._all_nan_target_names is None:
            self._all_nan_target_names = {
                name for name in gen_data if torch.isnan(target_data[name]).all()
            }
        self._n_batches += 1

    def _get_caption(self, name: str) -> str:
        if self._metadata is not None and name in self._metadata:
            caption_name = self._metadata[name].display_long_name(name)
            units = self._metadata[name].display_units()
        else:
            caption_name, units = name, "unknown_units"
        caption = f"{caption_name} ({units})"
        return caption

    def _get_data(self):
        if self._variable_metrics is None or self._n_batches == 0:
            raise ValueError("No batches have been recorded.")
        data: dict[str, torch.Tensor] = {}
        all_variable_names: set[str] = set()
        for metric in sorted(self._variable_metrics):
            for key in sorted(self._variable_metrics[metric]):
                all_variable_names.add(key)
                metric_value = self._dist.reduce_mean(
                    self._variable_metrics[metric][key].get()
                )
                data[f"{metric}/{key}"] = (
                    self._gridded_operations.area_weighted_mean(metric_value, name=key)
                    .cpu()
                    .numpy()
                )
                if self._log_mean_maps:
                    data[f"{metric}/mean_map/{key}"] = plot_paneled_data(
                        [[metric_value.cpu().numpy()]],
                        diverging=metric in self._diverging_metrics,
                        caption=self._get_caption(key),
                    )
            if self._target == "norm":
                all_keys = list(self._variable_metrics[metric].keys())
                if self._channel_mean_names is None:
                    names = all_keys
                else:
                    missing = [n for n in self._channel_mean_names if n not in all_keys]
                    if missing:
                        raise KeyError(
                            f"channel_mean_names contains entries not present "
                            f"in the recorded data: {missing}. Available: "
                            f"{sorted(all_keys)}."
                        )
                    names = list(self._channel_mean_names)
                # Exclude variables whose target was entirely NaN (e.g. filled
                # by allow_missing_variables) from the channel mean.
                nan_targets = self._all_nan_target_names or set()
                names = [name for name in names if name not in nan_targets]
                if names:
                    scalars = [data[f"{metric}/{key}"] for key in names]
                    data[f"{metric}/channel_mean"] = sum(scalars) / len(scalars)
        if self._report_variables is not None:
            nan_targets = all_variable_names - self._report_variables
            data = {
                k: v
                for k, v in data.items()
                if not any(seg in nan_targets for seg in k.split("/"))
            }
        return data

    @torch.no_grad()
    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return {
            f"{label}/{key}": data for key, data in sorted(self._get_data().items())
        }

    @torch.no_grad()
    def get_dataset(self) -> xr.Dataset:
        data = self._get_data()
        data = {key.replace("/", "-"): data[key] for key in data}
        data_vars = {}
        for key, value in data.items():
            data_vars[key] = xr.DataArray(value)
        return xr.Dataset(data_vars=data_vars)


class SelectStepEnsembleAggregator:
    """
    Wraps an aggregator that takes a time dimension, and records
    metrics for a specific time step.
    """

    def __init__(
        self,
        aggregator: _EnsembleAggregator,
        i_target_time: int,
    ):
        """
        Args:
            aggregator: Aggregator to wrap.
            i_target_time: Global time index of the target time step.
        """
        self._aggregator = aggregator
        self._i_target_time = i_target_time

    def record_batch(
        self,
        target_data: EnsembleTensorDict,
        gen_data: EnsembleTensorDict,
        target_data_norm: EnsembleTensorDict | None = None,
        gen_data_norm: EnsembleTensorDict | None = None,
        i_time_start: int = 0,
    ):
        """
        Record a specific global timestep of data.

        Call does nothing if the target time is not in this batch.

        Args:
            target_data: Target data, of shape [batch, ensemble, time, ...].
            gen_data: Generated data, of shape [batch, ensemble, time, ...].
            target_data_norm: Normalized target data, same shape as target_data.
            gen_data_norm: Normalized generated data, same shape as gen_data.
            i_time_start: Global time index of the first time step in the batch.
        """
        n_timesteps = next(iter(target_data.values())).shape[2]
        if (
            self._i_target_time >= i_time_start
            and self._i_target_time < i_time_start + n_timesteps
        ):
            batch_i_target_time = self._i_target_time - i_time_start

            def _select(data: EnsembleTensorDict) -> EnsembleTensorDict:
                return EnsembleTensorDict(
                    {
                        key: value[
                            :, :, batch_i_target_time : batch_i_target_time + 1, ...
                        ]
                        for key, value in data.items()
                    }
                )

            self._aggregator.record_batch(
                target_data=_select(target_data),
                gen_data=_select(gen_data),
                target_data_norm=(
                    _select(target_data_norm) if target_data_norm is not None else None
                ),
                gen_data_norm=(
                    _select(gen_data_norm) if gen_data_norm is not None else None
                ),
            )

    def get_logs(self, label: str):
        """
        Returns logs as can be reported to WandB.

        Args:
            label: Label to prepend to all log keys.
        """
        return self._aggregator.get_logs(label)

    def get_dataset(self) -> xr.Dataset:
        return self._aggregator.get_dataset()


@dataclasses.dataclass
class EnsembleMetricConfig:
    """
    Configuration for an ensemble metric (CRPS, SSR bias, ensemble-mean RMSE)
    at a specific forward step.

    Attributes:
        step: Forward step at which to compute the metric.
        name: Name to use for the logged metric. Defaults to
            ``ensemble_step_{step}`` for ``target="denorm"`` and
            ``ensemble_step_{step}_norm`` for ``target="norm"``.
        log_mean_maps: Whether to log per-variable mean maps.
        enabled: Whether the metric is enabled.
        strict: Whether to raise if the metric cannot be built.
        target: Whether to compute metrics on normalized ("norm") or
            denormalized ("denorm") data. ``channel_mean`` is only logged
            when ``target="norm"``, since averaging metrics across variables
            with different physical units is not meaningful.
        channel_mean_names: Names of variables to include in the channel-mean
            metric. If None, falls back to the aggregator-level value passed
            via the build context, and finally to all variables present in
            the data when that is also None. Names not present in the data
            raise KeyError. Ignored when ``target="denorm"``.
    """

    step: int = 20
    name: str | None = None
    log_mean_maps: bool = False
    enabled: bool = True
    strict: bool = False
    target: Literal["norm", "denorm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            base = f"ensemble_step_{self.step}"
            self.name = f"{base}_norm" if self.target == "norm" else base

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        if self.step > ctx.n_forward_steps:
            raise MetricNotSupportedError(
                f"ensemble step {self.step} exceeds "
                f"n_forward_steps={ctx.n_forward_steps}"
            )
        is_norm = self.target == "norm"
        return MetricBuildResult(
            ensemble=get_one_step_ensemble_aggregator(
                gridded_operations=ctx.ops,
                target_time=self.step,
                log_mean_maps=self.log_mean_maps,
                metadata=ctx.variable_metadata,
                target=self.target,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )


@dataclasses.dataclass
class OneStepEnsembleMetricConfig:
    """
    Configuration for the one-step ensemble metric (CRPS, SSR bias,
    ensemble-mean RMSE) at the first forward step.

    Attributes:
        name: Name to use for the logged metric. Defaults to ``ensemble``
            for ``target="denorm"`` and ``ensemble_norm`` for
            ``target="norm"``.
        log_mean_maps: Whether to log per-variable mean maps.
        enabled: Whether the metric is enabled.
        strict: Whether to raise if the metric cannot be built.
        target: Whether to compute metrics on normalized ("norm") or
            denormalized ("denorm") data. ``channel_mean`` is only logged
            when ``target="norm"``, since averaging metrics across variables
            with different physical units is not meaningful.
        channel_mean_names: Names of variables to include in the channel-mean
            metric. If None, falls back to the aggregator-level value passed
            via the build context, and finally to all variables present in
            the data when that is also None. Names not present in the data
            raise KeyError. Ignored when ``target="denorm"``.
    """

    name: str | None = None
    log_mean_maps: bool = True
    enabled: bool = True
    strict: bool = False
    target: Literal["norm", "denorm"] = "denorm"
    channel_mean_names: list[str] | None = None

    def __post_init__(self):
        if self.name is None:
            self.name = "ensemble_norm" if self.target == "norm" else "ensemble"

    def get_name(self) -> str:
        return self.name  # type: ignore[return-value]

    def build(self, ctx: OneStepBuildContext) -> OneStepMetricBuildResult:
        is_norm = self.target == "norm"
        return OneStepMetricBuildResult(
            ensemble=get_one_step_ensemble_aggregator(
                gridded_operations=ctx.ops,
                log_mean_maps=self.log_mean_maps,
                target_time=1,
                metadata=ctx.variable_metadata,
                target=self.target,
                channel_mean_names=(
                    (self.channel_mean_names or ctx.channel_mean_names)
                    if is_norm
                    else None
                ),
            )
        )
