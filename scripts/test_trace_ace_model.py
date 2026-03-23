"""Tests for scripts/trace_ace_model.py.

Uses synthetic steppers (no real checkpoint needed) to verify:
- TraceableACEModule forward pass
- Residual prediction
- torch.jit.trace round-trip (save/load)
- End-to-end match against the original Stepper
- Corrector operations (force-positive, dry-air, moisture advection)
"""

import dataclasses
import datetime
import pathlib

import pytest
import torch

import fme
from fme.ace.stepper.single_module import Stepper, StepperConfig
from fme.core.coordinates import HybridSigmaPressureCoordinate, LatLonCoordinates
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.registry.module import ModuleSelector
from fme.core.step import SingleModuleStepConfig, StepSelector
from fme.core.step.args import StepArgs

# Import from the standalone script (lives in the same directory)
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from trace_ace_model import (  # noqa: E402
    TraceableACEModule,
    area_weighted_mean,
    conserve_dry_air_packed,
    force_positive_channels,
    vertical_integral,
    zero_global_mean_moisture_advection_packed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMG_SHAPE = (5, 5)
TIMESTEP = datetime.timedelta(hours=6)


class _AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _LinearMap(torch.nn.Module):
    """Maps n_in channels to n_out channels via a 1x1 convolution (no bias)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(n_in, n_out, kernel_size=1, bias=False)
        # Initialize to identity-like for predictability
        torch.nn.init.eye_(self.conv.weight.view(n_out, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _scalar_data(names, value):
    return {n: float(value) for n in names}


def _get_dataset_info(n_levels: int = 3) -> DatasetInfo:
    horizontal = LatLonCoordinates(
        lat=torch.linspace(-89.5, 89.5, IMG_SHAPE[0]),
        lon=torch.zeros(IMG_SHAPE[1]),
    )
    ak = torch.linspace(0.0, 100.0, n_levels + 1)
    bk = torch.linspace(0.0, 1.0, n_levels + 1)
    vertical = HybridSigmaPressureCoordinate(ak=ak, bk=bk)
    return DatasetInfo(
        horizontal_coordinates=horizontal,
        vertical_coordinate=vertical,
        timestep=TIMESTEP,
    )


def _get_stepper(
    in_names: list[str],
    out_names: list[str],
    residual_prediction: bool = False,
    corrector_kwargs: dict | None = None,
    next_step_forcing_names: list[str] | None = None,
    module: torch.nn.Module | None = None,
) -> Stepper:
    all_names = list(set(in_names + out_names + (next_step_forcing_names or [])))
    corrector_kwargs = corrector_kwargs or {}
    next_step_forcing_names = next_step_forcing_names or []
    if module is None:
        module = _AddOne()
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt",
                        config={"module": module},
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means=_scalar_data(all_names, 0.0),
                            stds=_scalar_data(all_names, 1.0),
                        ),
                    ),
                    corrector=AtmosphereCorrectorConfig(**corrector_kwargs),
                    residual_prediction=residual_prediction,
                    next_step_forcing_names=next_step_forcing_names,
                )
            ),
        ),
    )
    return config.get_stepper(_get_dataset_info())


def _build_traceable(stepper: Stepper, include_corrector: bool = True):
    """Build a TraceableACEModule from a synthetic stepper using the script's
    load_and_build internals (without needing a .ckpt file)."""
    from trace_ace_model import _unwrap_step, _build_corrector_flags_and_channel_map

    step = _unwrap_step(stepper)
    config = step.config
    normalizer = step.normalizer

    in_names = list(config.in_names)
    out_names = list(config.out_names)

    sec_names: list[str] = []
    sec_module = None
    if config.secondary_decoder is not None:
        sec_names = list(config.secondary_decoder.secondary_diagnostic_names)
    sd = step.secondary_decoder
    if hasattr(sd, "_module"):
        sec_module = sd._module.torch_module
    all_out_names = out_names + sec_names
    forcing_names = list(config.next_step_forcing_names)

    in_means = torch.stack([normalizer.means[n].float().squeeze() for n in in_names])
    in_stds = torch.stack([normalizer.stds[n].float().squeeze() for n in in_names])
    out_means_l, out_stds_l = [], []
    for n in all_out_names:
        if n in normalizer.means:
            out_means_l.append(normalizer.means[n].float().squeeze())
            out_stds_l.append(normalizer.stds[n].float().squeeze())
        else:
            out_means_l.append(torch.tensor(0.0))
            out_stds_l.append(torch.tensor(1.0))
    out_means = torch.stack(out_means_l)
    out_stds = torch.stack(out_stds_l)

    prog_names = list(config.prognostic_names)
    prog_out_idx = torch.tensor(
        [all_out_names.index(n) for n in prog_names if n in all_out_names],
        dtype=torch.long,
    )
    prog_in_idx = torch.tensor(
        [in_names.index(n) for n in prog_names if n in in_names],
        dtype=torch.long,
    )

    corrector = step._corrector
    corr_config = getattr(corrector, "_config", None)
    fp_names: list[str] = []
    if corr_config is not None and hasattr(corr_config, "force_positive_names"):
        fp_names = corr_config.force_positive_names
    force_pos_idx = torch.tensor(
        [all_out_names.index(n) for n in fp_names if n in all_out_names],
        dtype=torch.long,
    )

    if include_corrector:
        flags, cmap = _build_corrector_flags_and_channel_map(
            stepper, in_names, all_out_names
        )
    else:
        flags = {"any_active": False}
        cmap = {}

    area_weights = None
    ak_t = None
    bk_t = None
    timestep_seconds = stepper._dataset_info.timestep.total_seconds()

    if flags.get("any_active", False):
        gridded_ops = stepper._dataset_info.gridded_operations
        if hasattr(gridded_ops, "_cpu_area_global"):
            area_weights = gridded_ops._cpu_area_global.float()
        vc = stepper._dataset_info.atmosphere_vertical_coordinate
        if vc is not None and hasattr(vc, "ak"):
            ak_t = vc.ak.float()
            bk_t = vc.bk.float()

    traceable = TraceableACEModule(
        module=step.module.torch_module,
        secondary_decoder_module=sec_module,
        in_names=in_names,
        out_names=out_names,
        all_out_names=all_out_names,
        forcing_names=forcing_names,
        in_means=in_means,
        in_stds=in_stds,
        out_means=out_means,
        out_stds=out_stds,
        residual_prediction=config.residual_prediction,
        prognostic_input_indices=prog_in_idx,
        prognostic_output_indices=prog_out_idx,
        force_positive_indices=force_pos_idx,
        corrector_flags=flags,
        area_weights=area_weights,
        ak=ak_t,
        bk=bk_t,
        timestep_seconds=timestep_seconds,
        corrector_channel_map=cmap,
    )
    return traceable.to(DEVICE).eval(), in_names, all_out_names, forcing_names


# ---------------------------------------------------------------------------
# Corrector unit tests
# ---------------------------------------------------------------------------


class TestForcePositiveChannels:
    def test_clamps_negative(self):
        x = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]]]).unsqueeze(0)  # [1,1,2,2]
        out = force_positive_channels(x, torch.tensor([0]))
        assert (out >= 0).all()

    def test_no_change_when_positive(self):
        x = torch.rand(1, 3, 4, 4)
        out = force_positive_channels(x, torch.tensor([0, 1, 2]))
        torch.testing.assert_close(out, x)


class TestAreaWeightedMean:
    def test_uniform_weights(self):
        field = torch.ones(2, 3, 3)
        weights = torch.ones(3, 3) / 9.0
        result = area_weighted_mean(field, weights)
        torch.testing.assert_close(result, torch.ones(2))


class TestVerticalIntegral:
    def test_constant_integrand(self):
        B, H, W, K = 1, 2, 2, 3
        integrand = torch.ones(B, H, W, K)
        ps = torch.full((B, H, W), 1000.0)
        ak = torch.linspace(0, 100, K + 1)
        bk = torch.linspace(0, 1, K + 1)
        result = vertical_integral(integrand, ps, ak, bk, gravity=9.80665)
        assert result.shape == (B, H, W)
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# TraceableACEModule tests
# ---------------------------------------------------------------------------


class TestTraceableACEModule:
    def test_forward_basic(self):
        """Basic forward: normalize → AddOne → denormalize."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        stepper = _get_stepper(in_names, out_names)
        traceable, _, all_out, forcing_names = _build_traceable(stepper)

        x = torch.randn(1, len(in_names), *IMG_SHAPE, device=DEVICE)
        f = torch.zeros(1, max(len(forcing_names), 1), *IMG_SHAPE, device=DEVICE)
        with torch.no_grad():
            y = traceable(torch.cat([x, f], dim=1))
        assert y.shape == (1, len(all_out), *IMG_SHAPE)

    def test_residual_prediction(self):
        """With residual_prediction=True, output = input + (input + 1) for prognostic
        channels (in normalized space, denormalized back).  With mean=0, std=1 norms
        this simplifies to output = 2*input + 1."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        stepper = _get_stepper(in_names, out_names, residual_prediction=True)
        traceable, _, _, forcing_names = _build_traceable(stepper)

        x = torch.ones(1, 2, *IMG_SHAPE, device=DEVICE) * 3.0
        f = torch.zeros(1, max(len(forcing_names), 1), *IMG_SHAPE, device=DEVICE)
        with torch.no_grad():
            y = traceable(torch.cat([x, f], dim=1))
        # normalized input = (3-0)/1 = 3, NN output = 4, residual = 4+3 = 7
        # denormalized = 7*1 + 0 = 7
        torch.testing.assert_close(y, torch.full_like(y, 7.0))

    def test_jit_trace_roundtrip(self, tmp_path: pathlib.Path):
        """Trace, save, reload, and verify identical outputs."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        stepper = _get_stepper(in_names, out_names)
        traceable, _, _, forcing_names = _build_traceable(stepper)

        x = torch.randn(1, 2, *IMG_SHAPE, device=DEVICE)
        f = torch.zeros(1, max(len(forcing_names), 1), *IMG_SHAPE, device=DEVICE)

        with torch.no_grad():
            expected = traceable(torch.cat([x, f], dim=1))
            traced = torch.jit.trace(traceable, (torch.cat([x, f], dim=1),))

        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))
        loaded = torch.jit.load(str(pt_path))

        with torch.no_grad():
            actual = loaded(torch.cat([x, f], dim=1))
        torch.testing.assert_close(actual, expected)

    def test_no_corrector(self):
        """With include_corrector=False, correctors are skipped."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        stepper = _get_stepper(in_names, out_names)
        traceable, _, _, forcing_names = _build_traceable(
            stepper, include_corrector=False
        )

        x = torch.randn(1, 2, *IMG_SHAPE, device=DEVICE)
        f = torch.zeros(1, max(len(forcing_names), 1), *IMG_SHAPE, device=DEVICE)
        with torch.no_grad():
            y = traceable(torch.cat([x, f], dim=1))
        assert y.shape == x.shape

    def test_force_positive_in_forward(self):
        """Channels listed in force_positive_names are clamped >= 0."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        stepper = _get_stepper(
            in_names,
            out_names,
            corrector_kwargs={"force_positive_names": ["a"]},
        )
        traceable, _, _, forcing_names = _build_traceable(stepper)

        # Use very negative input so output channel "a" would be negative
        # without clamping.  With AddOne and mean=0, std=1:
        # normalized = -100, NN = -99, denorm = -99 → clamped to 0
        x = torch.full((1, 2, *IMG_SHAPE), -100.0, device=DEVICE)
        f = torch.zeros(1, max(len(forcing_names), 1), *IMG_SHAPE, device=DEVICE)
        with torch.no_grad():
            y = traceable(torch.cat([x, f], dim=1))
        assert (y[:, 0] >= 0).all()


# ---------------------------------------------------------------------------
# End-to-end: TraceableACEModule vs Stepper
# ---------------------------------------------------------------------------


DEVICE = fme.get_device()


def _run_stepper_step(
    stepper: Stepper,
    in_names: list[str],
    forcing_names: list[str],
    input_data: dict[str, torch.Tensor],
    forcing_data: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run one step through the original Stepper and return output dict."""
    args = StepArgs(
        input=input_data,
        next_step_input_data=forcing_data,
    )
    with torch.no_grad():
        return dict(stepper.step(args))


class TestEndToEndMatchesStepper:
    """Verify that TraceableACEModule produces identical output to the
    original Stepper for the same inputs.  This is the definitive test
    that normalization, NN, residual, denormalization, and corrections
    are all faithfully captured."""

    def _compare(
        self,
        in_names: list[str],
        out_names: list[str],
        residual_prediction: bool = False,
        corrector_kwargs: dict | None = None,
        forcing_names: list[str] | None = None,
        module: torch.nn.Module | None = None,
    ):
        forcing_names = forcing_names or []
        stepper = _get_stepper(
            in_names,
            out_names,
            residual_prediction=residual_prediction,
            corrector_kwargs=corrector_kwargs,
            next_step_forcing_names=forcing_names,
            module=module,
        )
        traceable, _, all_out_names, traced_forcing = _build_traceable(stepper)

        # Build dict-based inputs (Stepper uses [B, H, W] per variable)
        torch.manual_seed(42)
        input_data = {n: torch.randn(1, *IMG_SHAPE, device=DEVICE) for n in in_names}
        forcing_data = {
            n: torch.randn(1, *IMG_SHAPE, device=DEVICE) for n in forcing_names
        }

        # Run through original Stepper
        stepper_out = _run_stepper_step(
            stepper, in_names, forcing_names, input_data, forcing_data
        )

        # Build packed tensor for TraceableACEModule: [B, C_in + C_forcing, H, W]
        input_packed = torch.stack(
            [input_data[n] for n in in_names], dim=1
        )  # [B, C_in, H, W]
        if forcing_names:
            forcing_packed = torch.stack(
                [forcing_data[n] for n in forcing_names], dim=1
            )
        else:
            forcing_packed = torch.zeros(1, 0, *IMG_SHAPE, device=DEVICE)
        inputs = torch.cat([input_packed, forcing_packed], dim=1)

        with torch.no_grad():
            traced_out = traceable(inputs)

        # Compare channel-by-channel
        for i, name in enumerate(all_out_names):
            if name not in stepper_out:
                continue
            torch.testing.assert_close(
                traced_out[:, i],
                stepper_out[name],
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch on channel '{name}' (index {i})",
            )

    def test_basic(self):
        """Simple case: no residual, no correctors."""
        self._compare(["a", "b"], ["a", "b"])

    def test_residual(self):
        """With residual prediction enabled."""
        self._compare(["a", "b"], ["a", "b"], residual_prediction=True)

    def test_force_positive(self):
        """With force_positive corrector."""
        self._compare(
            ["a", "b"],
            ["a", "b"],
            corrector_kwargs={"force_positive_names": ["a"]},
        )

    def test_with_forcing(self):
        """With forcing channels included (must be subset of in_names)."""
        in_names = ["a", "b", "forcing_1"]
        out_names = ["a", "b"]
        self._compare(
            in_names,
            out_names,
            forcing_names=["forcing_1"],
            module=_LinearMap(len(in_names), len(out_names)),
        )

    def test_jit_trace_matches_stepper(self, tmp_path: pathlib.Path):
        """Full round-trip: build → trace → save → load → compare to Stepper."""
        in_names = ["a", "b"]
        out_names = ["a", "b"]
        forcing_names: list[str] = []
        stepper = _get_stepper(in_names, out_names)
        traceable, _, all_out_names, _ = _build_traceable(stepper)

        torch.manual_seed(42)
        input_data = {n: torch.randn(1, *IMG_SHAPE, device=DEVICE) for n in in_names}
        forcing_data: dict[str, torch.Tensor] = {}

        input_packed = torch.stack([input_data[n] for n in in_names], dim=1)
        forcing_packed = torch.zeros(1, 0, *IMG_SHAPE, device=DEVICE)
        inputs = torch.cat([input_packed, forcing_packed], dim=1)

        # Trace, save, reload
        with torch.no_grad():
            traced = torch.jit.trace(traceable, (inputs,))
        pt_path = tmp_path / "model.pt"
        torch.jit.save(traced, str(pt_path))
        loaded = torch.jit.load(str(pt_path))

        # Compare loaded model to Stepper
        stepper_out = _run_stepper_step(
            stepper, in_names, forcing_names, input_data, forcing_data
        )
        with torch.no_grad():
            loaded_out = loaded(inputs)

        for i, name in enumerate(all_out_names):
            if name not in stepper_out:
                continue
            torch.testing.assert_close(
                loaded_out[:, i],
                stepper_out[name],
                atol=1e-5,
                rtol=1e-5,
                msg=f"Mismatch on channel '{name}' (index {i})",
            )


# ---------------------------------------------------------------------------
# Dry-air conservation test
# ---------------------------------------------------------------------------


class TestConserveDryAirPacked:
    def test_preserves_global_mean_dry_air(self):
        """After correction, global-mean dry-air surface pressure should match
        between input and output."""
        ak = torch.linspace(0.0, 100.0, 4)
        bk = torch.linspace(0.0, 1.0, 4)
        B, H, W = 1, 5, 5

        area_weights = torch.ones(H, W) / (H * W)

        # 4 channels: ps, water_0, water_1, water_2
        inp = torch.rand(B, 4, H, W) + 1.0
        inp[:, 0] *= 1e5  # surface pressure
        out = inp.clone() + torch.randn_like(inp) * 10.0
        out[:, 0] = inp[:, 0] + 500.0  # perturb ps

        result = conserve_dry_air_packed(
            input_packed=inp,
            output_packed=out,
            area_weights=area_weights,
            ak=ak,
            bk=bk,
            gravity=9.80665,
            ps_out_idx=0,
            water_out_indices=torch.tensor([1, 2, 3]),
            ps_in_idx=0,
            water_in_indices=torch.tensor([1, 2, 3]),
        )

        def _dry_air_mean(packed, ps_idx, water_idx):
            ps = packed[:, ps_idx].double()
            wat = torch.stack(
                [packed[:, i].double() for i in water_idx], dim=-1
            )
            twp = vertical_integral(wat, ps, ak.double(), bk.double(), 9.80665)
            dry = ps - 9.80665 * twp
            return area_weighted_mean(dry, area_weights.double())

        inp_dry_mean = _dry_air_mean(inp, 0, [1, 2, 3])
        out_dry_mean = _dry_air_mean(result, 0, [1, 2, 3])
        torch.testing.assert_close(out_dry_mean, inp_dry_mean, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Zero global-mean moisture advection test
# ---------------------------------------------------------------------------


class TestZeroGlobalMeanMoistureAdvection:
    def test_global_mean_is_zero(self):
        B, C, H, W = 1, 3, 5, 5
        output = torch.randn(B, C, H, W)
        area_weights = torch.ones(H, W) / (H * W)
        result = zero_global_mean_moisture_advection_packed(
            output, area_weights, advection_idx=1
        )
        gm = area_weighted_mean(result[:, 1], area_weights)
        torch.testing.assert_close(gm, torch.zeros(B), atol=1e-6, rtol=1e-6)
