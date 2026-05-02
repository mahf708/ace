import random

import numpy as np
import pytest
import torch
import torch_harmonics as harmonics

from fme.ace.data_loading.augmentation import (
    NullModifier,
    RotateModifier,
    RoundtripConfig,
    RoundtripModifier,
)
from fme.ace.data_loading.batch_data import BatchData


def rotate(data: torch.Tensor) -> torch.Tensor:
    return torch.flip(data, dims=[-2, -1])


def test_rotate_modifier_all_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=1.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotate(rotated_batch.data["UGRD"]), -1 * batch.data["UGRD"])
    assert torch.allclose(rotate(rotated_batch.data["VGRD"]), -1 * batch.data["VGRD"])
    assert torch.allclose(rotate(rotated_batch.data["PS"]), batch.data["PS"])


def test_rotate_modifier_no_rotation():
    rotate_modifier = RotateModifier(
        rotate_probability=0.0, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data["UGRD"].shape == (1, 2, n_lat, n_lon)
    assert torch.allclose(rotated_batch.data["UGRD"], batch.data["UGRD"])
    assert torch.allclose(rotated_batch.data["VGRD"], batch.data["VGRD"])
    assert torch.allclose(rotated_batch.data["PS"], batch.data["PS"])


def test_rotate_modifier_random_rotation():
    random.seed(0)
    rotate_modifier = RotateModifier(
        rotate_probability=0.5, additional_directional_names=[]
    )
    n_lat = 8
    n_lon = 16
    batch = BatchData.new_for_testing(
        names=["UGRD", "VGRD", "PS"],
        n_samples=40,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    rotated_batch = rotate_modifier(batch)
    assert rotated_batch.data.keys() == batch.data.keys()
    assert rotated_batch.data["UGRD"].shape == (40, 2, n_lat, n_lon)
    rotated = {}
    unrotated = {}
    for name in rotated_batch.data:
        unrotated[name] = np.all(
            torch.abs(batch.data[name] - rotated_batch.data[name]).cpu().numpy() < 1e-6,
            axis=(1, 2, 3),
        )
        if name in ("UGRD", "VGRD"):
            sign = -1
        else:
            sign = 1
        rotated[name] = np.all(
            torch.abs(sign * rotate(batch.data[name]) - rotated_batch.data[name])
            .cpu()
            .numpy()
            < 1e-6,
            axis=(1, 2, 3),
        )
        assert np.all(rotated[name] + unrotated[name] == 1), name
        assert np.sum(rotated[name]) > 0, name
        assert np.sum(unrotated[name]) > 0, name
    for name in ("VGRD", "PS"):
        assert np.all(rotated[name] == rotated["UGRD"]), name
        assert np.all(unrotated[name] == unrotated["UGRD"]), name


@pytest.mark.parametrize(
    "name, additional_directional_names, match_expected",
    [
        ("UGRD", [], True),
        ("VGRD", [], True),
        ("UGRD_10m", [], True),
        ("UGRD_10m", ["UGRD"], True),
        ("VGRD200", [], True),
        ("eastward_wind_3", [], True),
        ("UGRD10m", [], True),
        ("NWIND10m", [], False),
        ("NWIND10m", ["NWIND"], True),
    ],
)
def test_rotate_modifier_pattern(
    name: str, additional_directional_names: list[str], match_expected: bool
):
    rotate_modifier = RotateModifier(
        rotate_probability=1.0,
        additional_directional_names=additional_directional_names,
    )
    assert (rotate_modifier._pattern.match(name) is not None) == match_expected, name


def _build_roundtrip_modifier(
    nlat: int,
    nlon: int,
    fraction_modes_kept: float,
    variables: list[str] | None = None,
    grid: str = "equiangular",
) -> RoundtripModifier:
    """Build a RoundtripModifier directly from torch_harmonics, bypassing
    Distributed (matching the non-distributed singleton behavior)."""
    full_lmax = nlat
    full_mmax = nlon // 2 + 1
    lmax = max(1, int(round(fraction_modes_kept * full_lmax)))
    mmax = max(1, int(round(fraction_modes_kept * full_mmax)))
    sht = harmonics.RealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()
    isht = harmonics.InverseRealSHT(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).float()
    return RoundtripModifier(sht=sht, isht=isht, variables=variables)


def test_roundtrip_config_validates_fraction():
    with pytest.raises(ValueError):
        RoundtripConfig(fraction_modes_kept=0.0)
    with pytest.raises(ValueError):
        RoundtripConfig(fraction_modes_kept=1.5)
    # valid values must not raise
    RoundtripConfig(fraction_modes_kept=None)
    RoundtripConfig(fraction_modes_kept=0.5)
    RoundtripConfig(fraction_modes_kept=1.0)


def test_roundtrip_config_disabled_returns_null_modifier():
    config = RoundtripConfig(fraction_modes_kept=None)
    modifier = config.build_modifier(global_shape=(8, 16))
    assert isinstance(modifier, NullModifier)


def test_roundtrip_modifier_constant_field_unchanged():
    # A constant field has all spectral energy in mode (0, 0), so any
    # truncation that keeps at least one mode must leave it unchanged.
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["PS"],
        n_samples=2,
        n_timesteps=3,
        img_shape=(n_lat, n_lon),
    )
    constant = torch.full_like(batch.data["PS"], 0.42)
    batch = BatchData(
        data={"PS": constant},
        time=batch.time,
        horizontal_dims=batch.horizontal_dims,
    )
    out = modifier(batch)
    assert torch.allclose(out.data["PS"], constant, atol=1e-5)


def test_roundtrip_modifier_idempotent():
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    torch.manual_seed(0)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=2,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
    )
    once = modifier(batch)
    twice = modifier(once)
    assert torch.allclose(once.data["T"], twice.data["T"], atol=1e-5)


def test_roundtrip_modifier_only_selected_variables():
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(
        n_lat, n_lon, fraction_modes_kept=0.5, variables=["T"]
    )
    torch.manual_seed(1)
    batch = BatchData.new_for_testing(
        names=["T", "PS"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
    )
    out = modifier(batch)
    # PS untouched (object identity is fine since modifier passes through)
    assert torch.equal(out.data["PS"], batch.data["PS"])
    # T filtered (so should differ from input for random data)
    assert not torch.allclose(out.data["T"], batch.data["T"], atol=1e-5)


def test_roundtrip_modifier_preserves_batch_metadata():
    n_lat, n_lon = 8, 16
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=2,
        img_shape=(n_lat, n_lon),
        epoch=7,
    )
    out = modifier(batch)
    assert out.horizontal_dims == batch.horizontal_dims
    assert out.epoch == batch.epoch
    assert out.labels == batch.labels
    assert out.time.equals(batch.time)


def test_roundtrip_modifier_rejects_non_latlon_dims():
    n_lat, n_lon = 8, 16
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=0.5)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
        horizontal_dims=["face", "x"],
    )
    with pytest.raises(NotImplementedError):
        modifier(batch)


def test_roundtrip_modifier_full_fraction_is_near_identity():
    # Keeping every mode should leave the field essentially unchanged
    # (up to numerical SHT roundtrip error).
    n_lat, n_lon = 16, 32
    modifier = _build_roundtrip_modifier(n_lat, n_lon, fraction_modes_kept=1.0)
    torch.manual_seed(42)
    batch = BatchData.new_for_testing(
        names=["T"],
        n_samples=1,
        n_timesteps=1,
        img_shape=(n_lat, n_lon),
    )
    out = modifier(batch)
    assert torch.allclose(out.data["T"], batch.data["T"], atol=1e-3)
