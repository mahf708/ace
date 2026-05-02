import abc
import dataclasses
import re
from collections.abc import Sequence

import torch

from fme.ace.data_loading.batch_data import BatchData


@dataclasses.dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Attributes:
        rotate_probability: The probability of rotating the sphere by 180 degrees,
            as a value between 0.0 and 1.0.
        additional_directional_names: Names of variables whose sign is flipped when
            the poles are reversed. By default this includes known directional
            names as stored in RotateModifier.FLIP_NAMES.
    """

    rotate_probability: float = 0.0
    additional_directional_names: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.rotate_probability <= 1.0:
            raise ValueError(
                "rotate_probability must be between 0.0 and 1.0, "
                f"got {self.rotate_probability}"
            )

    def build_modifier(self) -> "BatchModifierABC":
        if self.rotate_probability > 0.0:
            return RotateModifier(
                self.rotate_probability, self.additional_directional_names
            )
        return NullModifier()


class BatchModifierABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchData) -> BatchData: ...


class RotateModifier(BatchModifierABC):
    """
    Modifier that rotates the sphere by 180 degrees so that the poles swap
    places. This is the same as flipping both zonal and meridional axes.

    Also flips the sign of horizontal directional variables such as horizontal
    winds in specific directions, so their new values reflect the rotated axes.
    The names of such variables are stored in the `FLIP_NAMES` class variable.
    Variables not included in this list are not flipped.

    Specifically, the regex pattern r'{name}(_?[0-9]+m?)?$' is used to match the
    names of variables whose sign is flipped when the poles are reversed, for
    each name in `FLIP_NAMES`. This will match both names that end with something
    like "_0", "_1", etc. or something like "10m" or "2m".

    Note that seasons are handled by the fact that solar insolation is a data
    variable, but time is not modified. This means monthly or seasonal averages
    using this data will be affected by the rotation.
    """

    # names of variables whose sign is flipped when the poles are reversed
    FLIP_NAMES = [
        "eastward_wind",
        "northward_wind",
        "UGRD",
        "VGRD",
        "U",
        "V",
    ]

    def __init__(
        self,
        rotate_probability: float,
        additional_directional_names: list[str],
    ):
        self.rotate_probability = rotate_probability
        self.additional_directional_names = additional_directional_names
        self._pattern = re.compile(
            r"({})(_?[0-9]+m?)?$".format(
                "|".join(self.FLIP_NAMES + self.additional_directional_names)
            )
        )

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon to rotate the sphere, got "
                f"{batch.horizontal_dims}"
            )
        example_value = next(iter(batch.data.values()))
        apply = (
            torch.rand(example_value.shape[0]).to(example_value.device)
            < self.rotate_probability
        )
        while len(apply.shape) < len(example_value.shape):
            apply = apply.unsqueeze(-1)
        new_data = {}
        for name, value in batch.data.items():
            new_value = torch.flip(value, dims=[-2, -1])
            if self._pattern.match(name):
                new_value = -1 * new_value
            new_data[name] = torch.where(apply, new_value, value)
        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            labels=batch.labels,
        )


class NullModifier(BatchModifierABC):
    def __call__(self, batch: BatchData) -> BatchData:
        return batch


@dataclasses.dataclass
class RoundtripConfig:
    """
    Configuration for a runtime spherical-harmonics roundtrip filter on data
    batches.

    A forward SHT truncated to ``fraction_modes_kept`` of the modes, followed by
    the corresponding inverse SHT, is applied to selected variables. This
    produces a band-limited version of the field on the same grid. It mirrors
    the offline ``xtorch_harmonics.roundtrip_filter`` used in the data
    preprocessing scripts, but moves the filter into the data loader so the
    cutoff can be changed without regenerating the dataset.

    Under spatial parallelism the modifier runs after ``scatter_spatial`` and
    uses the distributed SHT/ISHT implementations exposed by
    :class:`fme.core.distributed.Distributed`.

    Attributes:
        fraction_modes_kept: Fraction of spherical-harmonic modes to retain in
            (0, 1]. ``None`` disables the filter.
        variables: Names of variables to filter. ``None`` applies to every
            variable in the batch.
        grid: Quadrature grid passed to the SHT/ISHT, e.g. ``"equiangular"`` or
            ``"legendre-gauss"``.
    """

    fraction_modes_kept: float | None = None
    variables: Sequence[str] | None = None
    grid: str = "equiangular"

    def __post_init__(self):
        if self.fraction_modes_kept is not None and not (
            0.0 < self.fraction_modes_kept <= 1.0
        ):
            raise ValueError(
                f"fraction_modes_kept must be in (0, 1], got {self.fraction_modes_kept}"
            )

    def build_modifier(self, global_shape: tuple[int, int]) -> "BatchModifierABC":
        if self.fraction_modes_kept is None:
            return NullModifier()
        from fme.core.distributed import Distributed

        nlat, nlon = global_shape
        full_lmax = nlat
        full_mmax = nlon // 2 + 1
        lmax = max(1, int(round(self.fraction_modes_kept * full_lmax)))
        mmax = max(1, int(round(self.fraction_modes_kept * full_mmax)))
        comm = Distributed.get_instance()
        sht = comm.get_sht(nlat, nlon, lmax=lmax, mmax=mmax, grid=self.grid)
        isht = comm.get_isht(nlat, nlon, lmax=lmax, mmax=mmax, grid=self.grid)
        return RoundtripModifier(
            sht=sht,
            isht=isht,
            variables=(None if self.variables is None else list(self.variables)),
        )


class RoundtripModifier(BatchModifierABC):
    """Apply a spherical-harmonic roundtrip (forward + inverse SHT) to selected
    variables of each batch.

    The forward and inverse transforms must already be configured with matching
    truncated ``lmax``/``mmax``; the filtering happens implicitly via the
    truncation, so the caller is responsible for choosing modes.
    """

    def __init__(
        self,
        sht: torch.nn.Module,
        isht: torch.nn.Module,
        variables: list[str] | None,
    ):
        self._sht = sht
        self._isht = isht
        self._variables = variables
        self._initialized_device: torch.device | None = None

    def _ensure_on_device(self, device: torch.device) -> None:
        if self._initialized_device != device:
            self._sht = self._sht.to(device)
            self._isht = self._isht.to(device)
            self._initialized_device = device

    def _filter(self, value: torch.Tensor) -> torch.Tensor:
        self._ensure_on_device(value.device)
        leading_shape = value.shape[:-2]
        flat = value.reshape(-1, *value.shape[-2:])
        # SHT buffers are float32; cast to match and back, preserving dtype.
        coeffs = self._sht(flat.to(torch.float32))
        out = self._isht(coeffs).to(value.dtype)
        return out.reshape(*leading_shape, *out.shape[-2:])

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon for SHT roundtrip, "
                f"got {batch.horizontal_dims}"
            )
        new_data = {}
        for name, value in batch.data.items():
            if self._variables is not None and name not in self._variables:
                new_data[name] = value
                continue
            new_data[name] = self._filter(value)
        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            labels=batch.labels,
            epoch=batch.epoch,
            n_ensemble=batch.n_ensemble,
        )
