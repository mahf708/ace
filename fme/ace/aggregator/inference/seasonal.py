import logging
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
import xarray as xr

from fme.ace.aggregator.plotting import plot_paneled_data
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping
from fme.core.wandb import Image

from .data import InferenceBatchData


class SeasonalAggregator:
    def __init__(
        self,
        ops: GriddedOperations,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        global_img_shape: tuple[int, int] | None = None,
    ):
        self._area_weighted_mean = ops.area_weighted_mean
        self._variable_metadata = variable_metadata
        self._target_dataset: xr.Dataset | None = None
        self._gen_dataset: xr.Dataset | None = None
        self._global_img_shape = global_img_shape

    @torch.no_grad()
    def record_batch(
        self,
        data: InferenceBatchData,
    ):
        """Record a batch of data for computing time variability statistics."""
        time = data.time
        target_data = {name: value.cpu() for name, value in data.target.items()}
        gen_data = {name: value.cpu() for name, value in data.prediction.items()}
        target_ds = _to_dataset(target_data, time)
        gen_ds = _to_dataset(gen_data, time)

        # must keep a separate dataset for each sample to avoid averaging across
        # samples when we groupby year
        if self._target_dataset is None:
            self._target_dataset = target_ds.groupby(
                target_ds.valid_time.dt.season
            ).sum(dim="stacked_sample_time", skipna=False)
        else:
            self._target_dataset = _add_dataarray(
                self._target_dataset,
                target_ds.groupby(target_ds.valid_time.dt.season).sum(
                    dim="stacked_sample_time", skipna=False
                ),
            )

        if self._gen_dataset is None:
            self._gen_dataset = gen_ds.groupby(gen_ds.valid_time.dt.season).sum(
                dim="stacked_sample_time", skipna=False
            )
        else:
            self._gen_dataset = _add_dataarray(
                self._gen_dataset,
                gen_ds.groupby(gen_ds.valid_time.dt.season).sum(
                    dim="stacked_sample_time", skipna=False
                ),
            )

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        if self._target_dataset is None or self._gen_dataset is None:
            raise ValueError("No data has been recorded yet.")
        dist = Distributed.get_instance()
        # Step 1: data-parallel reduce-sum on per-spatial-rank-local datasets.
        # The result (target_local, gen_local) is the data-summed local-slice.
        if dist.world_size > 1:
            target_local = _reduce_dataset_data_only(dist, self._target_dataset)
            gen_local = _reduce_dataset_data_only(dist, self._gen_dataset)
        else:
            target_local = self._target_dataset
            gen_local = self._gen_dataset

        if len(gen_local.season) < 4 or len(target_local.season) < 4:
            # seasonal metrics undefined when not all seasons are recorded.
            # Need to early-return on every rank or block on collectives.
            return {}

        # normalize by counts
        target_local = cast(xr.Dataset, target_local / target_local["counts"])  # type: ignore
        gen_local = cast(xr.Dataset, gen_local / gen_local["counts"])  # type: ignore
        bias_local = gen_local - target_local

        metric_logs: dict[str, float] = {}
        # Step 2: compute area-weighted RMSE on local-slice tensors. The
        # spatial-aware area_weighted_mean (LatLonOperations) does a spatial
        # all-reduce internally, so each rank ends up with the same global
        # scalar — every rank must call it for the collective to complete.
        for name in gen_local.data_vars.keys():
            if name == "counts":
                continue
            mse_tensor = self._area_weighted_mean(
                torch.as_tensor(bias_local[name].values ** 2),
                name=name,
            )
            for i, season in enumerate(bias_local[name].season.values):
                rmse = float(mse_tensor[i].sqrt().numpy())
                metric_logs[f"time-mean-rmse/{name}-{season}"] = rmse
            rmse = float(
                # must compute area mean and then mean across seasons
                # before sqrt, so we can't use metrics.root_mean_squared_error
                mse_tensor.mean().sqrt().numpy()
            )
            metric_logs[f"time-mean-rmse/{name}"] = rmse

        # Step 3: gather per-pixel target / gen across the spatial group for
        # plotting. Non-root spatial ranks return early (after participating
        # in the spatial-reduce collectives above).
        if self._global_img_shape is not None and dist.has_spatial_parallelism():
            target = _gather_dataset_to_root(dist, target_local, self._global_img_shape)
            gen = _gather_dataset_to_root(dist, gen_local, self._global_img_shape)
            if target is None or gen is None:
                return {}
        else:
            target = target_local
            gen = gen_local

        if not dist.is_root():
            # Non-world-root data ranks (spatial-rank-0 of data-rank>0): we
            # still computed the metrics above and must avoid returning None
            # so the data-parallel reduction collectives complete on every
            # rank, but only world-root produces logs.
            return {}

        bias = gen - target
        plots: dict[str, Image] = {}

        for name in gen.data_vars.keys():
            if name == "counts":
                continue

            if self._variable_metadata is not None and name in self._variable_metadata:
                long_name = self._variable_metadata[name].long_name
                units = self._variable_metadata[name].units
                caption_name = f"{long_name} ({units})"
            else:
                caption_name = name

            target_mean_pattern = target[name].mean(dim="season")
            gen_anomaly = gen[name] - target_mean_pattern
            target_anomaly = target[name] - target_mean_pattern
            r2 = get_r2(gen_anomaly, target_anomaly)

            image = plot_paneled_data(
                [
                    [
                        target_anomaly.sel(season="DJF").values,
                        target_anomaly.sel(season="MAM").values,
                        target_anomaly.sel(season="JJA").values,
                        target_anomaly.sel(season="SON").values,
                    ],
                    [
                        gen_anomaly.sel(season="DJF").values,
                        gen_anomaly.sel(season="MAM").values,
                        gen_anomaly.sel(season="JJA").values,
                        gen_anomaly.sel(season="SON").values,
                    ],
                ],
                diverging=True,
                caption=(
                    f"Seasonal time-mean anomaly of {caption_name} for target (top) "
                    f"and gen (bottom) starting with DJF, R2={r2:.4f}. "
                    "Time-mean of target is subtracted from predictions and target."
                ),
            )
            plots[f"anomaly/{name}"] = image

            image_err = plot_paneled_data(
                [
                    [
                        bias[name].sel(season="DJF").values,
                        bias[name].sel(season="MAM").values,
                    ],
                    [
                        bias[name].sel(season="JJA").values,
                        bias[name].sel(season="SON").values,
                    ],
                ],
                diverging=True,
                caption=(
                    f"Seasonal bias of {caption_name} for DJF (Upper-Left), "
                    "MAM (UR), JJA (LL), and SON (LR). "
                    f"Seasonal anomaly R2={r2:.4f} (excludes time-mean of target)."
                ),
            )
            plots[f"bias/{name}"] = image_err

        if len(label) > 0:
            label = label + "/"
        logs: dict[str, Image | float] = {}
        logs.update({f"{label}{name}": plots[name] for name in plots.keys()})
        logs.update({f"{label}{name}": val for name, val in metric_logs.items()})
        return logs

    def get_dataset(self) -> xr.Dataset:
        logging.debug(
            "get_dataset not implemented for SeasonalAggregator. "
            "Returning an empty dataset."
        )
        return xr.Dataset()


ALL_SEASONS = np.asarray(["DJF", "MAM", "JJA", "SON"])


def _add_dataarray(da1: xr.DataArray, da2: xr.DataArray):
    """
    Perform dataarray addition, assuming any missing season indices
    have zero values.
    """
    if len(da1.season) < 4:
        da1 = da1.reindex(season=ALL_SEASONS, fill_value=0)
    if len(da2.season) < 4:
        da2 = da2.reindex(season=ALL_SEASONS, fill_value=0)
    return da1 + da2


def get_r2(da: xr.DataArray, target: xr.DataArray) -> float:
    """Compute the R2 value of the target compared to the reference."""
    SS_ref = np.sum((target.values - np.mean(target.values)) ** 2)
    SS_pred = np.sum((da - target).values ** 2)
    return float(1 - SS_pred / SS_ref)


def _reduce_dataset_data_only(
    dist: Distributed,
    dataset: xr.Dataset,
) -> xr.Dataset:
    """Reduce-sum the dataset across the data-parallel group only.

    Each spatial coordinate has its own data-parallel group, so the result
    is the per-(h, w) local-slice summed across data co-ranks.
    """
    names = sorted(list(dataset.data_vars))
    names.remove("counts")
    for name in names:
        if dataset[name].shape != dataset[names[0]].shape:
            raise ValueError(
                f"Variable {name} has shape {dataset[name].shape} "
                f"which is not equal to {dataset[names[0]].shape}"
            )
    tensor = torch.stack(
        [torch.as_tensor(dataset[name].values) for name in names],
        dim=0,
    ).to(get_device())
    reduced = dist.reduce_sum(tensor).cpu()
    reduced_counts = dist.reduce_sum(
        torch.as_tensor(dataset["counts"].values).to(get_device())
    ).cpu()
    dataset_out = xr.Dataset(
        {name: (["season", "lat", "lon"], reduced[i]) for i, name in enumerate(names)},
        coords=dataset.coords,
    )
    dataset_out["counts"] = xr.DataArray(reduced_counts, dims=["season"])
    return dataset_out


def _gather_dataset_to_root(
    dist: Distributed,
    dataset: xr.Dataset,
    global_img_shape: tuple[int, int],
) -> xr.Dataset | None:
    """Gather the per-pixel variables of ``dataset`` onto spatial-rank 0.

    Returns ``None`` on non-root spatial ranks. ``counts`` is scalar per
    season (already identical across spatial ranks) and is preserved
    unchanged.
    """
    names = sorted(list(dataset.data_vars))
    names.remove("counts")
    tensor = torch.stack(
        [torch.as_tensor(dataset[name].values) for name in names],
        dim=0,
    ).to(get_device())
    gathered = dist.gather_spatial_to_root(tensor, global_img_shape)
    if gathered is None:
        return None
    gathered_cpu = gathered.cpu()
    out = xr.Dataset(
        {
            name: (["season", "lat", "lon"], gathered_cpu[i])
            for i, name in enumerate(names)
        },
    )
    out["counts"] = dataset["counts"]
    return out


# Kept for backwards compatibility with callers that don't pass
# ``global_img_shape``; equivalent to the data-only reduce path.
def _reduce_datasets(
    dist: Distributed,
    dataset: xr.Dataset,
    global_img_shape: tuple[int, int] | None = None,
) -> xr.Dataset | None:
    reduced = _reduce_dataset_data_only(dist, dataset)
    if global_img_shape is not None and dist.has_spatial_parallelism():
        return _gather_dataset_to_root(dist, reduced, global_img_shape)
    return reduced


@torch.no_grad()
def _to_dataset(data: TensorMapping, time: xr.DataArray) -> xr.Dataset:
    """Convert a dictionary of data to an xarray dataset."""
    assert time.dims == ("sample", "time")  # must be consistent with this module
    data_vars = {}
    for name, tensor in data.items():
        data_vars[name] = (["sample", "time", "lat", "lon"], tensor)
    data_vars["counts"] = (["sample", "time"], np.ones(shape=time.shape))
    return xr.Dataset(data_vars, coords={"valid_time": time})
