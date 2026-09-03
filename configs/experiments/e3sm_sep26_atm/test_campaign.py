"""Tests for the sep26 campaign generator and checker.

    uv run --extra dev python -m pytest \
        configs/experiments/e3sm_sep26_atm/test_campaign.py

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


def _run(exp: str = "LG01", warm_start_from: str = "", **levels) -> mk.Run:
    """A single-seed run at the given word, with the waste guards' opt-in on.

    allow_degenerate is set because these helpers exercise the *other* guards;
    the waste guards get their own tests below.
    """
    e = mk.Experiment(
        exp,
        mk.Word.of(**levels),
        "test",
        allow_degenerate=True,
        warm_start_from=warm_start_from,
    )
    return mk.Run(e, seed=1)


def _built(warm_start_from: str = "", **levels) -> dict:
    return mk.build(_baseline(), _run(warm_start_from=warm_start_from, **levels))


def _check_written(tmp_path: pathlib.Path, config: dict, runid: str) -> list[str]:
    p = tmp_path / f"{runid}.yaml"
    p.write_text(yaml.safe_dump(config, sort_keys=False))
    return chk.check(p)


# ---------------------------------------------------------------- the naming --


def test_word_is_always_written_in_full_and_in_order():
    assert mk.Word.of().word() == "D0_G0_I0_M2_N0_Q0_R0_Y0_Z1"
    assert mk.Word.of(D="1", M="1", Z="0").word() == "D1_G0_I0_M1_N0_Q0_R0_Y0_Z0"
    # Written in any order, rendered in position order.
    a = mk.Word.of(Z="0", M="1", D="1")
    b = mk.Word.of(D="1", M="1", Z="0")
    assert a.word() == b.word()


def test_run_id_carries_the_study_family():
    run = mk.Run(mk.RUNLIST[0], seed=1)
    assert run.runid.startswith("RF02.sep26.atm.")
    assert run.runid.endswith(".S01")


def test_adding_a_level_renames_nothing():
    """Every level of every axis is defined up front, so a new one is additive."""
    before = [mk.Run(e, s).runid for e in mk.RUNLIST for s in e.seeds]
    # LEVELS["Z"] is NOISE_DIM -- the same dict, not a copy.
    mk.NOISE_DIM["9"] = 128
    try:
        after = [mk.Run(e, s).runid for e in mk.RUNLIST for s in e.seeds]
        assert before == after
        assert mk.Word.of(Z="9").word() == "D0_G0_I0_M2_N0_Q0_R0_Y0_Z9"
    finally:
        del mk.NOISE_DIM["9"]


def test_unknown_position_or_level_is_refused():
    with pytest.raises(mk.ConfigError, match="unknown position"):
        mk.Word.of(J="1")
    with pytest.raises(mk.ConfigError, match="unknown level"):
        mk.Word.of(M="7")


def test_experiment_ids_must_be_two_letters_and_two_digits():
    for bad in ("XX01", "LG1", "LG001", "lg01"):
        with pytest.raises(mk.ConfigError):
            mk.Experiment(bad, mk.Word.of(), "test")
    mk.Experiment("LG09", mk.Word.of(), "test")  # valid


def test_runid_round_trips_through_the_parser():
    for e in mk.RUNLIST:
        run = mk.Run(e, seed=1)
        exp, levels, seed = chk.parse_runid(run.runid)
        assert (exp, seed) == (e.exp, 1)
        for pos in mk.POSITIONS:
            assert levels[pos] == run.word.get(pos), (run.runid, pos)


@pytest.mark.parametrize(
    "stem, why",
    [
        ("LG01.sep26.atm.D0_G0_I0_M2_N0_Q0_R0_Y0.S01", "expected 9"),
        ("LG01.sep26.atm.G0_D0_I0_M2_N0_Q0_R0_Y0_Z1.S01", "out of order"),
        ("LG01.sep26.atm.D0_G0_I0_M7_N0_Q0_R0_Y0_Z1.S01", "unknown level"),
        ("XX01.sep26.atm.D0_G0_I0_M2_N0_Q0_R0_Y0_Z1.S01", "two study letters"),
        ("LG1.sep26.atm.D0_G0_I0_M2_N0_Q0_R0_Y0_Z1.S01", "two study letters"),
        ("LG01.sep26.ocn.D0_G0_I0_M2_N0_Q0_R0_Y0_Z1.S01", "not a sep26.atm run"),
        ("LG01.sep26.atm.D0_G0_I0_M2_N0_Q0_R0_Y0_Z1.s01", "seed field"),
        ("LG01.sep26.atm.D0_G0_I0_M2_N0_Q0_R0_Y0_Z1", "not <exp>"),
    ],
)
def test_malformed_run_ids_are_rejected(stem, why):
    with pytest.raises(ValueError, match=why):
        chk.parse_runid(stem)


# --------------------------------------------------------------- the wandb mirror --


def test_every_factor_token_becomes_a_wandb_tag():
    run = _run("LG01", G="1", M="1", Z="0")
    env = mk.env_file(run)
    tags = next(x for x in env.splitlines() if x.startswith("WANDB_TAGS=")).split(
        "=", 1
    )[1]
    for token in run.word.tokens():
        assert token in tags.split(","), token
    assert "LG01" in tags.split(",") and "LG" in tags.split(",")


def test_group_is_the_experiment_and_job_type_is_the_word():
    run = _run("LG01", G="1", M="1", Z="0")
    env = mk.env_file(run)
    assert "WANDB_RUN_GROUP=sep26.atm.LG01" in env
    assert f"WANDB_JOB_TYPE={run.word.word()}" in env


def test_generated_configs_use_the_new_wandb_project():
    config = _built()
    assert config["logging"]["project"] == "ACE2S-sep26-atm"
    assert config["logging"]["project"] != "SamudrACE-E3SMv3"


# ------------------------------------------------------------------ blockers --


def test_energy_score_with_one_or_three_members_is_refused():
    """aug26's E25 and E26.  Raise on the first batch; nothing caught them."""
    for members in ("1", "3"):
        with pytest.raises(mk.ConfigError, match="exactly two members"):
            mk.build(_baseline(), _run(M=members))


@pytest.mark.parametrize("members", ["1", "2", "3"])
def test_pure_crps_permits_any_member_count(members):
    """G1 sets energy_score_weight 0, and forward gates the energy score on it."""
    config = _built(G="1", M=members)
    assert config["stepper_training"]["loss"]["kwargs"]["energy_score_weight"] == 0.0
    assert config["stepper_training"]["n_ensemble"] == int(members)


def test_pure_energy_score_is_refused_until_the_shape_bug_is_fixed():
    """G2 leaves the energy score alone, and its shape breaks get_channel_losses."""
    with pytest.raises(mk.ConfigError, match="spurious leading"):
        mk.build(_baseline(), _run(G="2"))


def test_the_other_splits_still_build():
    for level in ("0", "1", "3"):
        crps_w, _ = mk.SPLIT[level]
        config = _built(G=level)
        assert config["stepper_training"]["loss"]["kwargs"]["crps_weight"] == crps_w


# ------------------------------------------------------- the lying-id guards --


@pytest.mark.parametrize("levels", [{"G": "1"}, {"Q": "1"}, {"Y": "1"}])
def test_ensemble_loss_kwargs_on_mse_are_refused(levels):
    with pytest.raises(mk.ConfigError, match="discards every EnsembleLoss kwarg"):
        mk.build(_baseline(), _run(D="1", M="1", Z="0", **levels))


def test_warm_start_needs_a_parent_and_a_parent_needs_a_warm_start():
    orphan = mk.Experiment("CU01", mk.Word.of(I="1"), "test")
    assert any("without warm_start_from" in c for c in mk.validate(mk.Run(orphan, 1)))
    unused = mk.Experiment("CU01", mk.Word.of(), "test", warm_start_from="RF02")
    assert any("with I0" in c for c in mk.validate(mk.Run(unused, 1)))


def test_warm_start_resolves_to_a_full_run_id_not_an_experiment_id():
    """run-train.sh resolves $CAMPAIGN_ROOT/<parent>/, which is keyed by RUN id.

    An experiment id there points at a directory that never exists, so the arm
    fails closed forever instead of only until its parent finishes.
    """
    cu = next(e for e in mk.RUNLIST if e.exp == "CU01")
    env = mk.env_file(mk.Run(cu, seed=1))
    line = next(x for x in env.splitlines() if x.startswith("FME_WARM_START_FROM="))
    parent = line.split("=", 1)[1]
    assert parent != "RF02"
    assert parent.startswith("RF02.sep26.atm.") and parent.endswith(".S01")
    # ...and it is a run the campaign actually generates.
    assert parent in {r.runid for r in mk.expand(mk.RUNLIST)}


def test_warm_start_parent_must_be_in_the_run_list():
    bogus = [mk.Experiment("CU01", mk.Word.of(I="1"), "t", warm_start_from="ZZ99")]
    with pytest.raises(mk.ConfigError, match="not in the run list"):
        mk.expand(bogus)


def test_degenerate_arms_need_an_explicit_opt_in():
    for levels in ({"Z": "0"}, {"D": "1", "M": "1"}):
        plain = mk.Run(mk.Experiment("LG01", mk.Word.of(**levels), "t"), 1)
        assert mk.validate(plain), f"{levels} should need allow_degenerate"
        opted = mk.Run(
            mk.Experiment("LG01", mk.Word.of(**levels), "t", allow_degenerate=True), 1
        )
        assert not mk.validate(opted)


def test_the_opt_in_reaches_the_artifacts():
    env = mk.env_file(_run("LG01", G="1", M="1", Z="0"))
    assert "degenerate-by-design" in env and "degenerate by design" in env


def test_one_configuration_gets_one_experiment_id():
    dupes = [
        mk.Experiment("LG01", mk.Word.of(G="1"), "one"),
        mk.Experiment("EN01", mk.Word.of(G="1"), "two"),
    ]
    with pytest.raises(mk.ConfigError, match="same configuration"):
        mk.expand(dupes)


# --------------------------------------------------------- config correctness --


def test_zero_noise_forces_gaussian():
    """Isotropic at zero channels dies in the MKL FFT; aug26 reproduced it."""
    for n in ("0", "1"):
        builder = _built(G="1", M="1", Z="0", N=n)["stepper"]["step"]["config"][
            "builder"
        ]
        assert builder["config"]["noise_embed_dim"] == 0
        assert builder["config"]["noise_type"] == "gaussian"


def test_mse_drops_the_ensemble_kwargs_entirely():
    loss = _built(D="1", M="1", Z="0")["stepper_training"]["loss"]
    assert loss["type"] == "MSE" and "kwargs" not in loss


def test_default_alpha_is_not_written_out():
    assert (
        "almost_fair_crps_alpha"
        not in _built(G="1")["stepper_training"]["loss"]["kwargs"]
    )


def test_warm_start_writes_the_placeholder_not_a_path():
    config = _built(I="1", warm_start_from="RF02")
    path = config["stepper_training"]["parameter_init"]["weights_path"]
    assert path == mk.WARM_START_PLACEHOLDER and "/pscratch" not in path


@pytest.mark.parametrize("epochs", [3, 10, 11, 12, 15, 20, 29, 30, 31, 60])
def test_inference_always_scores_the_final_epoch(epochs):
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


def test_rf01_is_still_aug26_e01():
    """The stochastic pole is inherited, not generated.  That claim is checked."""
    assert chk.check_rf01_claim() == []


def test_rf01_is_not_generated():
    """Re-running it would spend ~970 node-hours reproducing three trained seeds."""
    assert all(r.word.word() != chk.RF01_WORD for r in mk.expand(mk.RUNLIST))


# ------------------------------------------------------------------ mutation --


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
    (["train_loader", "batch_size"], 8, "batch_size is 8"),
    (["logging", "entity"], "someone-else", "wandb entity"),
    (["logging", "project"], "SamudrACE-E3SMv3", "wandb project"),
]


@pytest.mark.parametrize("path, value, expect", MUTATIONS)
def test_checker_catches_a_broken_config(tmp_path, path, value, expect):
    runid = f"EN01.sep26.atm.{mk.Word.of(G='1').word()}.S01"
    config = _built(G="1")
    assert _check_written(tmp_path, config, runid) == [], "clean config must pass"
    broken = _mutate(config, path, value)
    complaints = _check_written(tmp_path, broken, runid)
    assert any(expect in c for c in complaints), (expect, complaints)


def test_checker_catches_noise_type_drift(tmp_path):
    """A Z0 config that lost its gaussian override would die in the MKL FFT."""
    config = _built(G="1", M="1", Z="0")
    runid = f"LG01.sep26.atm.{mk.Word.of(G='1', M='1', Z='0').word()}.S01"
    assert _check_written(tmp_path, config, runid) == []
    broken = _mutate(
        config,
        ["stepper", "step", "config", "builder", "config", "noise_type"],
        "isotropic",
    )
    assert any("MKL FFT" in c for c in _check_written(tmp_path, broken, runid))


def test_checker_catches_the_energy_score_blocker_in_a_written_config(tmp_path):
    """Assemble aug26's E25 by hand -- the path by which it reached runs/."""
    config = _built(G="1")
    config["stepper_training"]["n_ensemble"] = 1
    config["stepper_training"]["loss"]["kwargs"] = {
        "crps_weight": 0.9,
        "energy_score_weight": 0.1,
    }
    runid = f"EN01.sep26.atm.{mk.Word.of(M='1').word()}.S01"
    complaints = _check_written(tmp_path, config, runid)
    assert any("exactly two members" in c for c in complaints), complaints


def test_checker_catches_a_loss_weight_that_disagrees_with_the_id(tmp_path):
    config = _built(G="1")
    runid = f"EN01.sep26.atm.{mk.Word.of(G='1').word()}.S01"
    assert _check_written(tmp_path, config, runid) == []
    broken = _mutate(
        config,
        ["stepper_training", "loss", "kwargs"],
        {"crps_weight": 0.9, "energy_score_weight": 0.1},
    )
    assert any("G1 wants" in c for c in _check_written(tmp_path, broken, runid))


def test_checker_catches_a_warm_start_that_lost_its_placeholder(tmp_path):
    config = _built(I="1", warm_start_from="RF02")
    runid = f"CU01.sep26.atm.{mk.Word.of(I='1').word()}.S01"
    assert _check_written(tmp_path, config, runid) == []
    broken = copy.deepcopy(config)
    del broken["stepper_training"]["parameter_init"]
    assert any("I1 but" in c for c in _check_written(tmp_path, broken, runid))
