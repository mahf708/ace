"""Tests for the configurations spatial parallelism refuses to run.

Each test here pins one combination that is known to produce a *plausible but
wrong* result under spatial parallelism -- a tile statistic reported as a
global one, a file containing one rank's tile, a weight whose gradient is the
sum of unrelated spectral modes. The value of a gate is that the failure is
loud and early, so these assert on the error, not on a workaround.

These run single-process against a backend that only *claims* a spatial
layout; that is enough because a gate is a function of the layout. Tests that
need data to really be split across ranks live in ``parallel_tests``.
"""

import numpy as np
import pytest
import torch

from fme.core.distributed import Distributed
from fme.core.distributed.distributed import SpatialParallelismNotImplemented
from fme.core.gridded_ops import HEALPixOperations, LatLonOperations
from fme.core.testing.distributed import fake_spatial_parallelism


def test_spatial_shape_defaults_to_no_decomposition():
    assert Distributed.get_instance().spatial_shape == (1, 1)


@pytest.mark.parametrize(
    "h_size,w_size,img_shape",
    [
        (2, 1, (45, 90)),  # 45 lat over 2 ranks
        (1, 2, (16, 45)),  # 45 lon over 2 ranks
        (2, 2, (16, 45)),
    ],
)
def test_uneven_spatial_split_is_rejected(h_size, w_size, img_shape):
    with fake_spatial_parallelism(h_size=h_size, w_size=w_size) as dist:
        with pytest.raises(ValueError, match="not divisible"):
            dist.require_even_spatial_split(img_shape)
        with pytest.raises(ValueError, match="not divisible"):
            dist.scatter_spatial({"a": torch.zeros(2, *img_shape)}, img_shape)


def test_even_spatial_split_is_accepted():
    with fake_spatial_parallelism(h_size=2, w_size=2) as dist:
        dist.require_even_spatial_split((16, 32))
        local = dist.scatter_spatial({"a": torch.zeros(2, 16, 32)}, (16, 32))
        assert local["a"].shape == (2, 8, 16)


def test_healpix_operations_rejected_under_spatial_parallelism():
    # its area-weighted reductions are local, so a "global mean" would be a
    # tile mean
    with fake_spatial_parallelism(h_size=2, w_size=1):
        with pytest.raises(SpatialParallelismNotImplemented, match="HEALPix"):
            HEALPixOperations(nside=8)


def test_lat_lon_operations_allowed_under_spatial_parallelism():
    with fake_spatial_parallelism(h_size=2, w_size=1):
        ops = LatLonOperations(area_weights=torch.ones(16, 32))
    # weights are localized to this rank's tile
    assert ops._cpu_area.shape == (8, 32)


def _sfno_module(spectral_lora_rank: int = 0, global_layer_norm: bool = False):
    from fme.core.models.conditional_sfno.sfnonet import (
        SFNONetConfig,
        get_lat_lon_sfnonet,
    )

    return get_lat_lon_sfnonet(
        params=SFNONetConfig(
            embed_dim=4,
            num_layers=1,
            spectral_lora_rank=spectral_lora_rank,
            global_layer_norm=global_layer_norm,
        ),
        in_chans=2,
        out_chans=2,
        img_shape=(16, 32),
    )


def test_global_layer_norm_rejected_under_spatial_parallelism():
    # nn.LayerNorm would normalize over the global (C, H, W) while being fed a
    # local tile
    with fake_spatial_parallelism(h_size=2, w_size=1):
        with pytest.raises(SpatialParallelismNotImplemented, match="global_layer_norm"):
            _sfno_module(global_layer_norm=True)


def test_spectral_lora_rejected_when_latitude_is_decomposed():
    with fake_spatial_parallelism(h_size=2, w_size=1):
        with pytest.raises(
            SpatialParallelismNotImplemented, match="spectral_lora_rank"
        ):
            _sfno_module(spectral_lora_rank=2)


def test_spectral_lora_allowed_when_only_longitude_is_decomposed():
    # lora_A/lora_B are indexed by spectral latitude mode only, so a w-only
    # decomposition leaves them fully replicated and correct
    with fake_spatial_parallelism(h_size=1, w_size=2):
        _sfno_module(spectral_lora_rank=2)


def test_makani_filter_rejected_under_spatial_parallelism():
    from fme.core.models.conditional_sfno.sfnonet import (
        SFNONetConfig,
        get_lat_lon_sfnonet,
    )

    with fake_spatial_parallelism(h_size=2, w_size=1):
        with pytest.raises(SpatialParallelismNotImplemented, match="makani-linear"):
            get_lat_lon_sfnonet(
                params=SFNONetConfig(
                    embed_dim=4, num_layers=1, filter_type="makani-linear"
                ),
                in_chans=2,
                out_chans=2,
                img_shape=(16, 32),
            )


def _data_writer_config(**kwargs):
    from fme.ace.inference.data_writer.main import DataWriterConfig

    return DataWriterConfig(**kwargs)


def _build_writer(config, tmp_path):
    import datetime

    import cftime

    from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata

    return config.build(
        experiment_dir=str(tmp_path),
        initial_condition_times=np.array(
            [cftime.DatetimeProlepticGregorian(2020, 1, 1)]
        ),
        n_timesteps=2,
        timestep=datetime.timedelta(hours=6),
        variable_metadata={},
        coords={"lat": np.arange(16.0), "lon": np.arange(32.0)},
        dataset_metadata=DatasetMetadata(source="test"),
    )


def test_inference_writers_rejected_under_spatial_parallelism(tmp_path):
    config = _data_writer_config(save_prediction_files=True, save_monthly_files=False)
    with fake_spatial_parallelism(h_size=2, w_size=1):
        with pytest.raises(SpatialParallelismNotImplemented, match="data writers"):
            _build_writer(config, tmp_path)


def test_aggregator_only_inference_allowed_under_spatial_parallelism(tmp_path):
    # nothing is written, so there is no file to race over
    config = _data_writer_config(save_prediction_files=False, save_monthly_files=False)
    with fake_spatial_parallelism(h_size=2, w_size=1):
        writer = _build_writer(config, tmp_path)
    assert writer._writers == []
