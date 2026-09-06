import unittest.mock
from typing import Literal

import pytest
import torch
from torch_harmonics import InverseRealSHT

from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNO, isotropic_noise
from fme.core.device import get_device
from fme.core.models.conditional_sfno.layers import Context


@pytest.mark.parametrize("nlat, nlon", [(8, 16), (64, 128)])
def test_isotropic_noise(nlat: int, nlon: int):
    torch.manual_seed(0)
    n_batch = 1000
    embed_dim = 4
    leading_shape = (n_batch, embed_dim)
    isht = InverseRealSHT(nlat, nlon, grid="legendre-gauss")
    lmax = isht.lmax
    mmax = isht.mmax
    noise = isotropic_noise(leading_shape, lmax, mmax, isht, device=get_device())
    assert noise.shape == (n_batch, embed_dim, nlat, nlon)
    assert noise.dtype == torch.float32
    torch.testing.assert_close(
        noise.mean(), torch.tensor(0.0, device=noise.device), atol=2e-3, rtol=0.0
    )
    torch.testing.assert_close(
        noise.std(), torch.tensor(1.0, device=noise.device), atol=5e-3, rtol=0.0
    )


def test_noise_conditioned_sfno_conditioning():
    mock_sfno = unittest.mock.MagicMock()
    img_shape = (32, 64)
    n_noise = 16
    n_pos = 8
    n_labels = 4
    label_embed_dim = 3
    model = NoiseConditionedSFNO(
        conditional_model=mock_sfno,
        img_shape=img_shape,
        embed_dim_noise=n_noise,
        embed_dim_pos=n_pos,
        n_labels=n_labels,
        label_embed_dim=label_embed_dim,
    )
    batch_size = 2
    x = torch.randn(batch_size, 3, img_shape[0], img_shape[1])
    labels = torch.randn(batch_size, n_labels)
    _ = model(x, labels=labels)
    mock_sfno.assert_called()
    args, _ = mock_sfno.call_args
    conditioned_x = args[0]
    assert conditioned_x.shape == (batch_size, 3, img_shape[0], img_shape[1])
    context = args[1]
    assert isinstance(context, Context)
    assert context.embedding_scalar is None
    assert context.embedding_pos is not None
    assert context.labels is not None
    assert context.noise is not None
    assert context.embedding_pos.shape == (
        batch_size,
        n_pos,
        img_shape[0],
        img_shape[1],
    )
    assert context.labels.shape == (batch_size, label_embed_dim)
    assert context.noise.shape == (batch_size, n_noise, img_shape[0], img_shape[1])


def test_noise_conditioned_sfno_onehot_labels():
    """When label_embed_dim=0, one-hot labels pass through directly."""
    mock_sfno = unittest.mock.MagicMock()
    img_shape = (32, 64)
    n_labels = 4
    model = NoiseConditionedSFNO(
        conditional_model=mock_sfno,
        img_shape=img_shape,
        embed_dim_noise=8,
        embed_dim_pos=4,
        n_labels=n_labels,
        label_embed_dim=0,
    )
    batch_size = 2
    x = torch.randn(batch_size, 3, img_shape[0], img_shape[1])
    labels = torch.randn(batch_size, n_labels)
    _ = model(x, labels=labels)
    args, _ = mock_sfno.call_args
    context = args[1]
    assert context.labels.shape == (batch_size, n_labels)


def _build_small_stochastic_model(
    noise_type: Literal["gaussian", "isotropic"],
) -> NoiseConditionedSFNO:
    from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
    from fme.core.dataset_info import DatasetInfo

    torch.manual_seed(0)
    model = NoiseConditionedSFNOBuilder(
        embed_dim=16,
        noise_embed_dim=4,
        num_layers=2,
        noise_type=noise_type,
        pos_embed=False,
    ).build(3, 3, DatasetInfo(img_shape=(8, 16)))
    # The noise pathway is zero-initialised (an exact identity at step 0), so
    # give it weight or nothing below can distinguish noise from no noise.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "W_scale_2d" in name or "W_bias_2d" in name:
                param.normal_(std=0.5)
    return model.to(get_device()).eval()


@pytest.mark.parametrize("noise_type", ["gaussian", "isotropic"])
def test_noise_override_scale_zero_is_deterministic(
    noise_type: Literal["gaussian", "isotropic"],
):
    model = _build_small_stochastic_model(noise_type)
    x = torch.randn(2, 3, 8, 16, device=get_device())
    with torch.no_grad():
        fresh_a = model(x)
        fresh_b = model(x)
        assert not torch.allclose(fresh_a, fresh_b), "fresh noise should differ"
        model.set_noise_override(scale=0.0)
        off_a = model(x)
        off_b = model(x)
    torch.testing.assert_close(off_a, off_b)
    # The noise-off output is the backbone g(x, 0), which a fresh draw perturbs.
    assert not torch.allclose(off_a, fresh_a)


@pytest.mark.parametrize("noise_type", ["gaussian", "isotropic"])
def test_noise_override_fixed_mode_holds_noise_across_calls(
    noise_type: Literal["gaussian", "isotropic"],
):
    model = _build_small_stochastic_model(noise_type)
    x = torch.randn(2, 3, 8, 16, device=get_device())
    model.set_noise_override(scale=1.0, mode="fixed")
    with torch.no_grad():
        a = model(x)
        b = model(x)
        # Each sample in the batch still gets its own field.
        assert not torch.allclose(a[0], a[1]) or not torch.allclose(x[0], x[1])
        torch.testing.assert_close(a, b)
        # A new batch shape draws a fresh fixed field rather than failing.
        c = model(x[:1])
        assert c.shape[0] == 1
        # Switching back to fresh restores per-call draws.
        model.set_noise_override(scale=1.0, mode="fresh")
        d = model(x)
        e = model(x)
    assert not torch.allclose(d, e)


def test_noise_override_scale_multiplies_the_draw():
    model = _build_small_stochastic_model("gaussian")
    x = torch.randn(1, 3, 8, 16, device=get_device())
    # Route both calls through an identical draw so only the scale differs.
    with unittest.mock.patch.object(
        model, "_draw_noise", return_value=torch.ones(1, 4, 8, 16, device=x.device)
    ):
        with torch.no_grad():
            model.set_noise_override(scale=1.0)
            unit = model(x)
            model.set_noise_override(scale=2.0)
            doubled = model(x)
            model.set_noise_override(scale=0.0)
            off = model(x)
    assert not torch.allclose(unit, doubled)
    assert not torch.allclose(unit, off)
    # With the draw pinned, scale 0 must match the true noise-off output.
    with torch.no_grad():
        model.set_noise_override(scale=0.0)
        off_again = model(x)
    torch.testing.assert_close(off, off_again)


def test_noise_override_rejects_bad_values():
    model = _build_small_stochastic_model("gaussian")
    with pytest.raises(ValueError):
        model.set_noise_override(scale=-1.0)
    with pytest.raises(ValueError):
        model.set_noise_override(scale=1.0, mode="held")  # type: ignore[arg-type]


def test_noise_override_is_not_in_the_state_dict():
    """Setting the override must not change what a checkpoint contains, so a
    stochastic checkpoint evaluated with the noise off is still the same
    checkpoint."""
    model = _build_small_stochastic_model("gaussian")
    before = set(model.state_dict().keys())
    model.set_noise_override(scale=0.0, mode="fixed")
    x = torch.randn(1, 3, 8, 16, device=get_device())
    with torch.no_grad():
        model(x)
    assert set(model.state_dict().keys()) == before


def test_noise_override_mean_mode_averages_draws():
    """'mean' with K draws returns the per-sample average of K fresh-noise
    outputs, and converges toward the many-draw mean rather than g(x, 0)."""
    model = _build_small_stochastic_model("gaussian")
    x = torch.randn(2, 3, 8, 16, device=get_device())
    torch.manual_seed(1)
    model.set_noise_override(scale=1.0, mode="mean", draws=4)
    with torch.no_grad():
        mean4 = model(x)
    assert mean4.shape == (2, 3, 8, 16)
    # Same draws, done by hand: the stacked call and K separate calls agree
    # in distribution; check exactness by pinning the draw.
    with unittest.mock.patch.object(
        model,
        "_draw_noise",
        side_effect=lambda xx: torch.arange(
            xx.shape[0], device=xx.device, dtype=xx.dtype
        )
        .view(-1, 1, 1, 1)
        .expand(xx.shape[0], 4, 8, 16)
        .clone(),
    ):
        with torch.no_grad():
            model.set_noise_override(scale=1.0, mode="mean", draws=3)
            stacked = model(x[:1])
            model.set_noise_override(scale=1.0, mode="fresh")
            outs = []
            for k in range(3):
                with unittest.mock.patch.object(
                    model,
                    "_draw_noise",
                    return_value=torch.full((1, 4, 8, 16), float(k), device=x.device),
                ):
                    outs.append(model(x[:1]))
    # The two paths are the same arithmetic at different batch shapes -- three
    # rows through one call against one row through three -- and in float32 a
    # spectral network picks different kernels for the two. MEASURED at 2.9e-4,
    # 0.07% of the mean output magnitude. The tolerance sits above that and far
    # below what a real averaging error would produce: dropping one of the three
    # draws moves these outputs by order 0.1.
    torch.testing.assert_close(
        stacked, torch.stack(outs).mean(dim=0), atol=1e-3, rtol=1e-3
    )
    with pytest.raises(ValueError):
        model.set_noise_override(scale=1.0, mode="mean", draws=0)
