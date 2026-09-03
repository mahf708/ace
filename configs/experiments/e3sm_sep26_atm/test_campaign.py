"""Tests for the sep26 campaign generator and checker.

    uv run python -m pytest configs/experiments/e3sm_sep26_atm/test_campaign.py

The interesting half is the mutation tests.  A checker that passes a clean
campaign proves nothing -- aug26's checker passed E25 and E26, which cannot
train.  So each mutation below breaks one thing about a generated config and
asserts the checker notices, which is the only evidence that its assertions are
load-bearing rather than decorative.
"""

import copy
import pathlib

import check_campaign as chk
import make_campaign as mk
import pytest
import yaml

HERE = pathlib.Path(__file__).resolve().parent


def _baseline() -> dict:
    return yaml.safe_load(mk.TEMPLATE.read_text())


def _run(**levels) -> mk.Run:
    """A single-seed run at the given delta, with the guards' opt-in on.

    allow_degenerate is set because these helpers are used to exercise the
    *other* guards; the degenerate guards get their own tests below.
    """
    exp = mk.Experiment("T00", mk.Delta.of(**levels), "test", allow_degenerate=True)
    return mk.Run(exp, seed=1)


def _built(**levels) -> dict:
    return mk.build(_baseline(), _run(**levels))


def _check_written(tmp_path: pathlib.Path, config: dict, runid: str) -> list[str]:
    p = tmp_path / f"{runid}.yaml"
    p.write_text(yaml.safe_dump(config, sort_keys=False))
    return chk.check(p)


# ------------------------------------------------------- the naming property --


def test_delta_word_is_sparse_and_sorted():
    assert mk.Delta.of().word() == "base"
    assert mk.Delta.of(mem="2").word() == "base"  # baseline level is dropped
    assert mk.Delta.of(crps="pure").word() == "crps-pure"
    # Written in either order, rendered the same: there is no position to widen.
    a = mk.Delta.of(mem="1", crps="pure", noise="0")
    b = mk.Delta.of(noise="0", crps="pure", mem="1")
    assert a == b
    assert a.word() == "crps-pure_mem-1_noise-0"


def test_adding_an_axis_renames_nothing():
    """The whole reason for this convention over aug26's positional word."""
    before = [mk.Run(e, s).runid for e in mk.RUNLIST for s in e.seeds]
    mk.LEVELS["zzz_probe"] = ("off", "on")
    mk.BASELINE["zzz_probe"] = "off"
    try:
        after = [mk.Run(e, s).runid for e in mk.RUNLIST for s in e.seeds]
        assert before == after
        # ...and the new axis is usable immediately, without touching the rest.
        assert mk.Delta.of(zzz_probe="on").word() == "zzz_probe-on"
    finally:
        del mk.LEVELS["zzz_probe"]
        del mk.BASELINE["zzz_probe"]


def test_unknown_axis_or_level_is_refused():
    with pytest.raises(mk.ConfigError, match="unknown axis"):
        mk.Delta.of(nonsense="1")
    with pytest.raises(mk.ConfigError, match="unknown level"):
        mk.Delta.of(mem="7")


def test_runid_round_trips_through_the_parser():
    for exp in mk.RUNLIST:
        run = mk.Run(exp, seed=1)
        levels, seed = chk.parse_runid(run.runid)
        assert seed == 1
        for axis in mk.BASELINE:
            assert levels[axis] == run.delta.get(axis), (run.runid, axis)


@pytest.mark.parametrize(
    "stem, why",
    [
        ("sep26.atm.mem-1_crps-pure.s01", "not in canonical"),
        ("sep26.atm.mem-2.s01", "must be omitted"),
        ("sep26.atm.mem-1_mem-3.s01", "appears twice"),
        ("sep26.atm.bogus-1.s01", "unknown axis"),
        ("sep26.atm.mem1.s01", "not <axis>-<level>"),
        ("sep26.ocn.base.s01", "not a sep26.atm run"),
        ("sep26.atm.base.S01", "seed field"),
        ("sep26.atm.base.s01.extra", "not <campaign>"),
    ],
)
def test_malformed_run_ids_are_rejected(stem, why):
    with pytest.raises(ValueError, match=why):
        chk.parse_runid(stem)


# ------------------------------------------------------------- the blocker --


def test_energy_score_with_one_member_is_refused():
    """aug26's E25.  Raises on the first training batch; nothing caught it."""
    run = _run(mem="1")  # keeps the template's 0.9/0.1, so energy_score_weight > 0
    complaints = mk.validate(run)
    assert any("get_energy_score supports exactly two" in c for c in complaints)
    with pytest.raises(mk.ConfigError, match="exactly two members"):
        mk.build(_baseline(), run)


def test_energy_score_with_three_members_is_refused():
    """aug26's E26, same failure."""
    with pytest.raises(mk.ConfigError, match="exactly two members"):
        mk.build(_baseline(), _run(mem="3"))


@pytest.mark.parametrize("members", ["1", "2", "3"])
def test_pure_crps_permits_any_member_count(members):
    """The workaround: at energy_score_weight 0 the energy score is never called.

    EnsembleLoss.forward gates it on `energy_score_weight > 0`, so pure CRPS
    skips get_energy_score entirely.  This is what makes the mechanism block
    runnable before the upstream generalization lands.
    """
    config = _built(crps="pure", mem=members)
    kwargs = config["stepper_training"]["loss"]["kwargs"]
    assert kwargs["energy_score_weight"] == 0.0
    assert config["stepper_training"]["n_ensemble"] == int(members)


def test_mse_permits_any_member_count():
    """MSE never reaches the ensemble loss at all."""
    for members in ("1", "2", "3"):
        config = _built(obj="mse", mem=members)
        assert config["stepper_training"]["n_ensemble"] == int(members)


def test_pure_energy_score_is_refused_until_the_upstream_shape_bug_is_fixed():
    """The second blocker, found by running an arm rather than validating it.

    EnergyScoreLoss builds `mode_weights` with `x_hat.ndim - 1` leading
    singleton dims, but get_energy_score has already consumed the ensemble dim,
    so the energy component comes out shaped (1, 1, B, C, n_l, n_m).  With a
    CRPS component present the correctly-shaped one carries the channel
    breakdown; at crps_weight 0 it is the only component and
    get_channel_losses raises on the first training batch.
    """
    with pytest.raises(mk.ConfigError, match="spurious leading"):
        mk.build(_baseline(), _run(crps="energy"))


def test_the_other_crps_splits_still_build():
    """Only crps_weight == 0 is blocked; the mixed weightings are unaffected."""
    for level in ("std", "pure", "half"):
        config = _built(crps=level)
        crps_w, _ = mk.CRPS_WEIGHTS[level]
        assert config["stepper_training"]["loss"]["kwargs"]["crps_weight"] == crps_w


# ------------------------------------------------------- the lying-id guards --


@pytest.mark.parametrize(
    "levels", [{"crps": "pure"}, {"fdcrps": "1"}, {"alpha": "095"}]
)
def test_ensemble_loss_kwargs_on_mse_are_refused(levels):
    """LossConfig.build discards them, so the id would claim what the run lacks."""
    with pytest.raises(mk.ConfigError, match="discards every EnsembleLoss kwarg"):
        mk.build(_baseline(), _run(obj="mse", **levels))


def test_alpha_is_refused_under_pure_energy_because_it_is_inert():
    with pytest.raises(mk.ConfigError, match="inert when"):
        mk.build(_baseline(), _run(crps="energy", alpha="095"))


def test_fdcrps_is_refused_under_pure_energy_because_it_is_not_inert():
    """The asymmetry: alpha is gated on crps_weight, fdcrps on its own weight.

    So crps-energy_fdcrps-1 would quietly run a pooled-CRPS-plus-energy
    objective while the id says the CRPS family is off.
    """
    with pytest.raises(mk.ConfigError, match="gated on its own weight"):
        mk.build(_baseline(), _run(crps="energy", fdcrps="1"))


def test_degenerate_arms_need_an_explicit_opt_in():
    for levels in ({"noise": "0"}, {"obj": "mse", "mem": "1"}):
        plain = mk.Run(mk.Experiment("T00", mk.Delta.of(**levels), "test"), 1)
        assert mk.validate(plain), f"{levels} should need allow_degenerate"
        opted = mk.Run(
            mk.Experiment("T00", mk.Delta.of(**levels), "test", allow_degenerate=True),
            1,
        )
        assert not mk.validate(opted)


def test_the_opt_in_reaches_the_artifacts():
    """The intent has to live in the files, not in someone's memory."""
    run = _run(noise="0")
    env = mk.env_file(run)
    assert "degenerate-by-design" in env
    assert "degenerate by design" in env


# --------------------------------------------------------- config correctness --


def test_zero_noise_forces_gaussian():
    """Isotropic at zero channels dies in the MKL FFT; aug26 reproduced it."""
    builder = _built(noise="0")["stepper"]["step"]["config"]["builder"]["config"]
    assert builder["noise_embed_dim"] == 0
    assert builder["noise_type"] == "gaussian"
    # ...even when the delta explicitly asks for isotropic.
    over = _built(noise="0", ntype="iso")["stepper"]["step"]["config"]["builder"]
    assert over["config"]["noise_type"] == "gaussian"


def test_mse_drops_the_ensemble_kwargs_entirely():
    loss = _built(obj="mse")["stepper_training"]["loss"]
    assert loss["type"] == "MSE"
    assert "kwargs" not in loss


def test_default_alpha_is_not_written_out():
    """Writing a default is how a file grows settings nobody chose."""
    kwargs = _built(crps="pure")["stepper_training"]["loss"]["kwargs"]
    assert "almost_fair_crps_alpha" not in kwargs


@pytest.mark.parametrize("epochs", [3, 10, 11, 12, 15, 20, 29, 30, 31, 60])
def test_inference_always_scores_the_final_epoch(epochs):
    """The off-by-one aug26 hit: solving `start` against a range one too long
    left the last fire a stride short, so the final epoch was never scored."""
    config = _baseline()
    mk.apply_epoch_schedule(config, epochs)
    for block in config["inference"]:
        got = block["epochs"]
        fires = list(range(1, epochs + 1))[got["start"] :: got["step"]]
        assert fires[-1] == epochs, (epochs, got, fires[-3:])


def test_generated_configs_name_nobody_s_scratch():
    for path in sorted((HERE / "runs").glob("*")):
        assert "/pscratch/" not in path.read_text(), path.name


def test_the_whole_campaign_checks_out():
    paths = sorted((HERE / "runs").glob("*.yaml"))
    assert paths, "runs/ is empty; run make_campaign.py --all -o runs"
    assert [c for path in paths for c in chk.check(path)] == []


def test_run_ids_are_unique():
    runs = mk.expand(mk.RUNLIST)
    assert len({r.runid for r in runs}) == len(runs)


def test_duplicate_experiments_are_caught():
    dupe = [
        mk.Experiment("A", mk.Delta.of(crps="pure"), "one"),
        mk.Experiment("B", mk.Delta.of(crps="pure"), "two"),
    ]
    with pytest.raises(mk.ConfigError, match="same run id"):
        mk.expand(dupe)


# ------------------------------------------------------------------ mutation --
# Each of these breaks one thing about a config the generator produced and
# asserts the checker notices.  Without them, "0 complaints" is not evidence.


def _mutate(config: dict, path: list[str], value) -> dict:
    out = copy.deepcopy(config)
    node = out
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return out


MUTATIONS = [
    (["stepper_training", "n_ensemble"], 3, "n_ensemble"),
    (["stepper_training", "n_forward_steps"], 2, "n_forward_steps"),
    (["stepper_training", "optimize_last_step_only"], False, "optimize_last_step_only"),
    (["seed"], 99, "seed"),
    (["max_epochs"], 31, "do not score the last epoch"),
    (["train_loader", "batch_size"], 15, "not a whole node count"),
    (["logging", "entity"], "someone-else", "wandb entity"),
]


@pytest.mark.parametrize("path, value, expect", MUTATIONS)
def test_checker_catches_a_broken_config(tmp_path, path, value, expect):
    runid = "sep26.atm.base.s01"
    config = mk.build(_baseline(), mk.Run(mk.Experiment("T00", mk.Delta.of(), "t"), 1))
    assert _check_written(tmp_path, config, runid) == [], "clean config must pass"
    broken = _mutate(config, path, value)
    complaints = _check_written(tmp_path, broken, runid)
    assert any(expect in c for c in complaints), (expect, complaints)


def test_checker_catches_noise_type_drift(tmp_path):
    """A Z00 config that lost its gaussian override would die in the MKL FFT."""
    config = _built(noise="0")
    runid = "sep26.atm.noise-0.s01"
    assert _check_written(tmp_path, config, runid) == []
    broken = _mutate(
        config,
        ["stepper", "step", "config", "builder", "config", "noise_type"],
        "isotropic",
    )
    complaints = _check_written(tmp_path, broken, runid)
    assert any("MKL FFT" in c for c in complaints), complaints


def test_checker_catches_the_energy_score_blocker_in_a_written_config(tmp_path):
    """Hand-write the config aug26 shipped as E25 and confirm this checker fails it.

    The generator refuses to build it, so it is assembled here by hand -- which
    is exactly the path by which E25 reached runs/ in the first place.
    """
    config = mk.build(_baseline(), mk.Run(mk.Experiment("T", mk.Delta.of(), "t"), 1))
    config["stepper_training"]["n_ensemble"] = 1
    complaints = _check_written(tmp_path, config, "sep26.atm.mem-1.s01")
    assert any("exactly two members" in c for c in complaints), complaints


def test_checker_catches_a_loss_weight_that_disagrees_with_the_id(tmp_path):
    config = _built(crps="pure")
    runid = "sep26.atm.crps-pure.s01"
    assert _check_written(tmp_path, config, runid) == []
    broken = _mutate(
        config,
        ["stepper_training", "loss", "kwargs"],
        {"crps_weight": 0.9, "energy_score_weight": 0.1},
    )
    complaints = _check_written(tmp_path, broken, runid)
    assert any("crps-pure wants" in c for c in complaints), complaints
