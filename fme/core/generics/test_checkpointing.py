"""Tests for DCP-based checkpoint helpers in ``fme.core.generics._checkpointing``.

The helpers are exercised single-process; ``dcp.save`` / ``dcp.load`` work
without an initialized process group via the ``no_dist=True`` path.
"""

import warnings
from pathlib import Path

import pytest
import torch

from fme.core.generics._checkpointing import (
    _is_root_rank,
    is_dcp_checkpoint,
    load_legacy_checkpoint,
    load_state_dict,
    save_state_dict,
)


def _make_state(seed: int = 0) -> dict[str, object]:
    """A nested state dict that mixes tensor and non-tensor leaves."""
    g = torch.Generator().manual_seed(seed)
    return {
        "format_version": 1,
        "epoch": 7,
        "best_validation_loss": 1.25,
        "model": {
            "linear.weight": torch.randn(3, 4, generator=g),
            "linear.bias": torch.randn(4, generator=g),
        },
        "ema": {
            "decay": torch.tensor(0.99),
            "num_updates": torch.tensor(42),
            "module_name_to_ema_name": {"foo": "ema_foo"},
            "ema_params": {
                "linear.weight": torch.randn(3, 4, generator=g),
            },
        },
        "training_history": [{"job": "j1"}, {"job": "j2"}],
    }


def _make_template_like(state: dict[str, object]) -> dict[str, object]:
    """Build a template with same structure / shapes but zeroed tensors and
    placeholder non-tensor values."""
    out: dict[str, object] = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = torch.zeros_like(v)
        elif isinstance(v, dict):
            out[k] = _make_template_like(v)
        elif isinstance(v, list):
            out[k] = []
        else:
            out[k] = type(v)() if not isinstance(v, bool) else False
    return out


def _assert_state_equal(a, b) -> None:
    assert type(a) is type(b), f"type mismatch: {type(a)} vs {type(b)}"
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"key mismatch: {a.keys()} vs {b.keys()}"
        for k in a:
            _assert_state_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _assert_state_equal(x, y)
    else:
        assert a == b, f"value mismatch: {a!r} vs {b!r}"


def test_is_dcp_checkpoint_distinguishes_dir_and_file(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.tar"
    legacy.write_bytes(b"")
    assert not is_dcp_checkpoint(str(legacy))

    plain_dir = tmp_path / "plain_dir"
    plain_dir.mkdir()
    assert not is_dcp_checkpoint(str(plain_dir))  # no .metadata file

    dcp_dir = tmp_path / "dcp_dir"
    dcp_dir.mkdir()
    (dcp_dir / ".metadata").write_bytes(b"")
    assert is_dcp_checkpoint(str(dcp_dir))


def test_dcp_round_trip_preserves_tensors_and_metadata(tmp_path: Path) -> None:
    state = _make_state(seed=0)
    path = tmp_path / "ckpt"
    save_state_dict(state, str(path))
    assert is_dcp_checkpoint(str(path))

    template = _make_template_like(state)
    load_state_dict(template, str(path))
    _assert_state_equal(template, state)


def test_dcp_save_is_atomic_replaces_existing(tmp_path: Path) -> None:
    path = tmp_path / "ckpt"
    save_state_dict(_make_state(seed=0), str(path))
    assert is_dcp_checkpoint(str(path))

    # Save a different state to the same path; the new contents must
    # replace the old, not coexist with it.
    state2 = _make_state(seed=1)
    save_state_dict(state2, str(path))
    template = _make_template_like(state2)
    load_state_dict(template, str(path))
    _assert_state_equal(template, state2)

    # The .tmp directory used during atomic save must have been cleaned up.
    assert not (tmp_path / "ckpt.tmp").exists()


def test_dcp_save_failure_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ckpt"

    # Corrupt the save by patching dcp.save to raise.
    import torch.distributed.checkpoint as dcp

    def _broken_save(*args, **kwargs):
        raise RuntimeError("simulated save failure")

    monkeypatch.setattr(dcp, "save", _broken_save)
    with pytest.raises(RuntimeError, match="simulated save failure"):
        save_state_dict(_make_state(), str(path))

    # Neither the final path nor the tmp path should exist.
    assert not path.exists()
    assert not (tmp_path / "ckpt.tmp").exists()


def test_load_state_dict_rejects_non_dcp_path(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.tar"
    torch.save({"x": torch.zeros(1)}, legacy)
    with pytest.raises(ValueError, match="not a DCP checkpoint"):
        load_state_dict({"x": torch.zeros(1)}, str(legacy))


def test_load_legacy_checkpoint_emits_deprecation_warning(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.tar"
    torch.save({"epoch": 5, "weight": torch.tensor([1.0, 2.0])}, legacy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_legacy_checkpoint(str(legacy))
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert loaded["epoch"] == 5
    torch.testing.assert_close(loaded["weight"], torch.tensor([1.0, 2.0]))


def test_dcp_load_missing_keys_in_template_loads_subset(tmp_path: Path) -> None:
    """A template with a subset of keys reads only those keys from the ckpt."""
    state = _make_state(seed=0)
    path = tmp_path / "ckpt"
    save_state_dict(state, str(path))

    # Template only requests "epoch" and "model"; other saved keys are skipped.
    partial = {
        "epoch": 0,
        "model": {
            "linear.weight": torch.zeros(3, 4),
            "linear.bias": torch.zeros(4),
        },
    }
    load_state_dict(partial, str(path))
    assert partial["epoch"] == 7
    torch.testing.assert_close(
        partial["model"]["linear.weight"], state["model"]["linear.weight"]
    )


def test_is_root_rank_is_true_when_distributed_uninitialized() -> None:
    # In the test environment torch.distributed is not initialized; the
    # helper must report root so single-process saves go through.
    assert _is_root_rank() is True


def test_save_state_dict_skips_writes_on_non_root_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When called on a non-root rank, save_state_dict must be a no-op so
    that a multi-rank caller's contract (root-only save) is preserved."""
    monkeypatch.setattr("fme.core.generics._checkpointing._is_root_rank", lambda: False)
    path = tmp_path / "ckpt"
    save_state_dict(_make_state(), str(path))
    assert not path.exists()
