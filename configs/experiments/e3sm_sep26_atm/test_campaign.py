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
import torch
import yaml

from fme.core.ensemble import get_crps, get_energy_score
from fme.core.gridded_ops import LatLonOperations
from fme.core.loss import EnsembleLoss, LossOutput

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


@pytest.mark.parametrize("members", ["1", "2", "3"])
def test_energy_score_now_permits_any_member_count(members):
    """Was aug26's E25 and E26, which raised on the first batch and were
    refused for it. get_energy_score is generalised on this branch, and M3
    with an energy weight was re-run on a node -- 209 steps, loss 4.0444 ->
    0.3705 -- rather than trusted to unit tests, because the fault it used to
    hit only appeared at training time."""
    config = _built(M=members)
    assert config["stepper_training"]["n_ensemble"] == int(members)
    assert config["stepper_training"]["loss"]["kwargs"]["energy_score_weight"] > 0


@pytest.mark.parametrize("members", ["1", "2", "3"])
def test_pure_crps_permits_any_member_count(members):
    """G1 sets energy_score_weight 0, and forward gates the energy score on it."""
    config = _built(G="1", M=members)
    assert config["stepper_training"]["loss"]["kwargs"]["energy_score_weight"] == 0.0
    assert config["stepper_training"]["n_ensemble"] == int(members)


def test_pure_energy_score_now_builds():
    """G2 leaves the energy score as the only loss component, which used to
    break get_channel_losses on the first batch. Re-run on a node after the
    mode_weights fix landed: 250 steps, loss 1.1952 -> 0.2952, and no
    "Per-channel loss has" error."""
    config = _built(G="2")
    kwargs = config["stepper_training"]["loss"]["kwargs"]
    assert kwargs["crps_weight"] == 0.0 and kwargs["energy_score_weight"] == 1.0


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
    """Isotropic at zero channels dies in the MKL FFT; aug26 reproduced it.

    So Z0 resolves the builder to gaussian even though the word says N0
    (isotropic). The token is inert rather than wrong: no noise of either
    type is drawn from a zero-channel tensor.
    """
    builder = _built(G="1", M="1", Z="0")["stepper"]["step"]["config"]["builder"]
    assert builder["config"]["noise_embed_dim"] == 0
    assert builder["config"]["noise_type"] == "gaussian"


def test_non_default_noise_type_at_zero_width_is_refused():
    """N1 at Z0 would name a setting that has no effect, so it is not a word."""
    with pytest.raises(mk.ConfigError, match="no noise of either type is drawn"):
        _built(G="1", M="1", Z="0", N="1")


def test_pure_crps_with_identical_members_is_refused():
    """MEASURED: at Z0 the members are bit-identical, so CRPS is exactly MAE
    and the extra members buy nothing at all (analysis/z0_degeneracy.py)."""
    for m in ("2", "3"):
        with pytest.raises(mk.ConfigError, match="bit-identical"):
            _built(G="1", M=m, Z="0")


@pytest.mark.parametrize("members", ["1", "2", "3"])
def test_almost_fair_alpha_now_permitted_at_any_member_count(members):
    """epsilon scales with the ensemble size on this branch, so almost-fair
    CRPS is the AIFS definition everywhere, not only at two members."""
    config = _built(G="1", M=members, Y="1")
    assert (
        config["stepper_training"]["loss"]["kwargs"]["almost_fair_crps_alpha"] == 0.95
    )


def test_checker_accepts_alpha_at_three_members(tmp_path):
    """The alpha restriction is lifted, so a Y1/M3 config must now pass the
    checker rather than be reported as a lie."""
    config = _built(G="1", M="3", Y="1")
    runid = _run(G="1", M="3", Y="1").runid
    assert _check_written(tmp_path, config, runid) == []


def test_checker_catches_bit_identical_members(tmp_path):
    """Mutation: relabel an M1/Z0 run as M2 and the checker must object."""
    config = _built(G="1", M="1", Z="0")
    config["stepper_training"]["n_ensemble"] = 2
    runid = _run(G="1", M="1", Z="0").runid.replace("_M1_", "_M2_")
    assert any("bit-identical" in c for c in _check_written(tmp_path, config, runid))


def test_checker_catches_an_inert_noise_type_token(tmp_path):
    """Mutation: relabel a Z0 run as N1, which cannot mean anything there."""
    config = _built(G="1", M="1", Z="0")
    runid = _run(G="1", M="1", Z="0").runid.replace("_N0_", "_N1_")
    assert any(
        "no noise of either type" in c for c in _check_written(tmp_path, config, runid)
    )


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


def test_checker_accepts_an_energy_score_at_one_member(tmp_path):
    """This shape was aug26's E25, and it used to be refused. It is now a
    legal config, so the checker must pass it -- while still catching an
    n_ensemble that disagrees with the run id, which is a different fault."""
    config = _built(M="1")
    runid = f"EN01.sep26.atm.{mk.Word.of(M='1').word()}.S01"
    assert _check_written(tmp_path, config, runid) == []
    config["stepper_training"]["n_ensemble"] = 2
    assert any("M1" in c for c in _check_written(tmp_path, config, runid))


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


# ---------------------------------------------- RF01 loss-code comparability --
#
# RF01 is inherited: it is aug26's E01, trained before B1/B2/B5 were ported onto
# this branch. check_rf01_claim() asserts the CONFIG still matches, but the loss
# CODE changed underneath it, and a config check cannot see that. These pin the
# numerical claim that made the port safe -- that all three fixes are inert for
# RF01's exact settings (EnsembleLoss 0.9/0.1, two members, alpha 1.0).
#
# NOTE ON WHAT THESE DO AND DO NOT GUARD. The two epsilon tests deliberately do
# NOT fail if B5 is reverted: they assert values at alpha 1.0 and at M2, which
# is exactly where the old and new definitions agree. That is the claim being
# pinned -- the port could not have moved this campaign -- not the fix. The fix
# itself is guarded upstream by
# fme/core/test_ensemble.py::test_almost_fair_crps_matches_its_definition,
# which fails at M3 and M5 without it. Mutation-checked 2026-09-03: reverting
# B2's normalisation or B1's ndim DOES fail the two tests below.


def _rf01_loss_inputs():
    """A fixed synthetic batch shaped like RF01's: two members, 3 channels."""
    torch.manual_seed(0)
    gen = torch.randn(4, 2, 3, 8, 16)
    target = torch.randn(4, 1, 3, 8, 16)
    return gen, target


@pytest.mark.parametrize("n_ensemble", [2, 3])
def test_almost_fair_epsilon_is_inert_at_alpha_one(n_ensemble):
    """B5 changed epsilon to (1-alpha)/M. Every arm but OI04 runs at alpha 1.0,
    where epsilon is zero and CRPS is the plain fair estimator -- so the change
    cannot have moved them, at any ensemble size."""
    torch.manual_seed(0)
    gen = torch.randn(16, n_ensemble, 6)
    target = torch.randn(16, 1, 6)
    ordered_pairs = sum(
        (gen[:, i] - gen[:, j]).abs()
        for i in range(n_ensemble)
        for j in range(n_ensemble)
        if i != j
    )
    fair = (gen - target).abs().mean(dim=1) - ordered_pairs / (
        2 * n_ensemble * (n_ensemble - 1)
    )
    torch.testing.assert_close(get_crps(gen, target, alpha=1.0), fair)


def test_almost_fair_epsilon_agrees_with_the_old_constant_at_two_members():
    """OI04 is the one alpha != 1 arm, and it runs at M2, where the old
    hard-coded (1-alpha)/2 and the new (1-alpha)/M are the same number."""
    torch.manual_seed(0)
    gen, target = torch.randn(16, 2, 6), torch.randn(16, 1, 6)
    alpha = 0.95
    old_epsilon = (1.0 - alpha) / 2.0
    pair = (gen[:, 0] - gen[:, 1]).abs()
    old = (gen - target).abs().mean(dim=1) - (1.0 - old_epsilon) * 0.5 * pair
    torch.testing.assert_close(get_crps(gen, target, alpha=alpha), old)


def test_energy_score_is_bit_identical_at_two_members():
    """B2 generalised get_energy_score past two members. Every arm that uses it
    runs at M2, where the generalisation must reproduce the old expression."""
    torch.manual_seed(0)
    shape = (8, 2, 3, 5)
    gen = torch.randn(*shape) + 1j * torch.randn(*shape)
    target = torch.randn(8, 1, 3, 5) + 1j * torch.randn(8, 1, 3, 5)
    old = (gen - target).abs().mean(dim=1) - 0.5 * (gen[:, 0] - gen[:, 1]).abs()
    torch.testing.assert_close(get_energy_score(gen, target), old, rtol=0, atol=0)


def test_energy_component_is_per_channel_not_constant():
    """B1 repaired the energy component's shape. The bug made its per-channel
    contribution a constant; RF01's scalar total was a mean over that constant
    and so did not move, but the breakdown was meaningless."""
    gen, target = _rf01_loss_inputs()
    sht = LatLonOperations(torch.ones((8, 16))).get_real_sht()
    loss = EnsembleLoss(crps_weight=0.9, energy_score_weight=0.1, sht=sht)
    components = loss(gen, target)
    total = LossOutput(components, ["a", "b", "c"]).total()
    assert torch.isfinite(total)
    # The bug made the energy term constant across channels; it must now vary.
    energy = [c for c in components if c.loss.shape[-1] != gen.shape[-1]]
    assert energy, "expected a spectral energy component"
    per_channel = energy[0].reduce_to_channel()
    assert per_channel.shape == (4, 3), per_channel.shape
    assert (
        len(set(per_channel[0].tolist())) > 1
    ), "energy term is constant across channels"


# ------------------------------------------------------ the eval generator --
#
# The offline evaluation is a launch gate rather than a nice-to-have: every
# inline rollout block runs one member per initial condition, so nothing in
# training measures calibration or spread.  These tests cover the three ways
# an eval config can be wrong in a way that only shows up on a node.


def _eval(
    *,
    which_pass: str = "scores",
    noise: str = "keep",
    word: mk.Word | None = None,
    members: int = 4,
    n_ics: int = 8,
    nodes: int = 2,
    years: int = 1,
    data_root: str | None = None,
) -> dict:
    import make_eval_config as ev

    return ev.build(
        runid="RF01.S01",
        checkpoint="/nowhere",
        word=word,
        which_pass=which_pass,
        noise=noise,
        members=members,
        n_ics=n_ics,
        nodes=nodes,
        years=years,
        seed=1,
        out_dir="/tmp/eval",
        data_root=data_root,
    )


def test_eval_refuses_initial_conditions_the_loader_cannot_split():
    """8 ICs over 12 ranks asserts inside __getitem__, minutes into a job."""
    import make_eval_config as ev

    with pytest.raises(ev.EvalError, match="remainder"):
        _eval(n_ics=8, nodes=3)


def test_eval_names_the_node_counts_that_would_work():
    import make_eval_config as ev

    with pytest.raises(ev.EvalError, match=r"divide 8: \[1, 2\]"):
        _eval(n_ics=8, nodes=3)


def test_eval_refuses_a_noise_mode_on_a_deterministic_arm():
    """Z0 has no noise pathway; the stepper raises, but only after loading."""
    import make_eval_config as ev

    with pytest.raises(ev.EvalError, match="no noise pathway"):
        _eval(noise="off", word=mk.Word.of(D="1", M="1", Z="0"))


def test_eval_takes_a_noise_mode_on_a_stochastic_arm():
    config = _eval(noise="mean", word=mk.Word.of())
    assert config["stepper_override"]["noise"]["mode"] == "mean"
    assert config["stepper_override"]["noise"]["draws"] > 1


def test_eval_keep_writes_no_stepper_override():
    """`keep` must leave the checkpoint's own behaviour alone."""
    assert "stepper_override" not in _eval(noise="keep")


def test_scores_pass_scores_the_ensemble_and_writes_no_trajectories():
    config = _eval(which_pass="scores", members=4)
    assert config["n_ensemble_per_ic"] == 4
    assert config["aggregator"]["ensembles"]
    assert config["data_writer"]["save_prediction_files"] is False


def test_trajectory_pass_writes_trajectories_and_scores_no_ensemble():
    config = _eval(which_pass="traj", members=1)
    assert config["n_ensemble_per_ic"] == 1
    assert "ensembles" not in config["aggregator"]
    assert config["data_writer"]["save_prediction_files"] is True


def test_eval_reads_its_data_from_the_training_template():
    """An eval config that names its own dataset can drift from the runs."""
    import make_eval_config as ev

    template = ev._template()
    block = ev._test_block(template)
    config = _eval(n_ics=8)
    template_dataset = dict(block["loader"]["dataset"])
    eval_dataset = dict(config["loader"]["dataset"])
    # the file glob is deliberately narrowed to the reachable years; every
    # other key -- paths, renames, unit conversions -- comes straight across
    assert eval_dataset.pop("file_pattern") != template_dataset.pop("file_pattern")
    assert eval_dataset == template_dataset
    assert (
        config["loader"]["start_indices"]["times"]
        == block["loader"]["start_indices"]["times"][:8]
    )


def test_eval_refuses_more_initial_conditions_than_the_template_holds():
    import make_eval_config as ev

    with pytest.raises(ev.EvalError, match="fewer than"):
        _eval(n_ics=32, nodes=8)


def test_eval_ids_use_campaign_names_not_inherited_ones():
    """RF01's weights are aug26's E01; every table calls it RF01."""
    import make_eval_config as ev

    assert ev.eval_id("RF01.S01", "scores", "keep") == "RF01.S01.eval-scores"
    assert ev.eval_id("RF01.S01", "traj", "off") == "RF01.S01.eval-traj-off"


def test_eval_narrows_the_file_glob_to_the_reachable_years():
    """1,501 files open in 13+ minutes, and every eval job pays it."""
    import make_eval_config as ev

    pattern = "v3.LR.historical_0101.aigo.eam.h0.*.nc"
    ics = ["2040-01-03T12:00:00", "2047-07-03T12:00:00"]
    assert ev.narrow_file_pattern(pattern, ics, years=5) == (
        "v3.LR.historical_0101.aigo.eam.h0.20[4-5]*.nc"
    )


def test_eval_glob_covers_the_end_of_the_rollout():
    """A July 2047 start plus five years runs into 2052."""
    import make_eval_config as ev

    narrowed = ev.narrow_file_pattern("x.h0.*.nc", ["2047-07-03T12:00:00"], years=5)
    assert narrowed == "x.h0.20[4-5]*.nc"
    # one year from 2040 stays inside one decade
    assert (
        ev.narrow_file_pattern("x.h0.*.nc", ["2040-01-03T12:00:00"], 1)
        == "x.h0.204*.nc"
    )


def test_eval_leaves_an_unexpected_pattern_alone():
    """Better a slow open than a glob that silently drops the target years."""
    import make_eval_config as ev

    assert (
        ev.narrow_file_pattern("something.zarr", ["2040-01-03"], 5) == "something.zarr"
    )


def test_eval_config_uses_the_narrowed_pattern():
    """Five years from the 2047 start of the full block reaches 2053."""
    both = _eval(n_ics=16, years=5)["loader"]["dataset"]["file_pattern"]
    assert "20[4-5]" in both
    # the first eight initial conditions stop in 2043, so one decade covers
    # even a five-year rollout from them
    assert "204*" in _eval(n_ics=8, years=5)["loader"]["dataset"]["file_pattern"]


def test_scores_pass_stops_at_the_last_scored_lead():
    """The scores pass is read at fixed leads, so rolling out past the last of
    them costs compute and yields no ensemble metric. Its default length is
    exactly the last lead; the trajectory pass, which measures drift, is
    longer."""
    import make_eval_config as ev

    assert ev.SCORES_YEARS * mk.STEPS_PER_YEAR == max(ev.SCORE_STEPS)
    assert ev.TRAJ_YEARS > ev.SCORES_YEARS
    scores = _eval(which_pass="scores", years=ev.SCORES_YEARS)
    assert scores["n_forward_steps"] == max(ev.SCORE_STEPS)
    # every scored lead is still reachable, so nothing was lost by shortening
    assert len(scores["aggregator"]["step_means"]) == len(ev.SCORE_STEPS)


def test_pass_chooses_its_own_default_length(tmp_path):
    """--years is optional and resolves per pass, so the cheap default is the
    one you get by not thinking about it."""
    import make_eval_config as ev

    for which_pass, expected in (
        ("scores", ev.SCORES_YEARS),
        ("traj", ev.TRAJ_YEARS),
    ):
        out = tmp_path / which_pass
        assert (
            ev.main(
                [
                    "RF01.S01",
                    "--pass",
                    which_pass,
                    "--out",
                    str(out),
                    "--no-wandb",
                ]
            )
            == 0
        )
        written = list(out.glob("*/config.yaml"))
        assert len(written) == 1
        config = yaml.safe_load(written[0].read_text())
        assert config["n_forward_steps"] == expected * mk.STEPS_PER_YEAR


def test_eval_refuses_an_incomplete_staged_data_root(tmp_path):
    """Staging is a copy, and a copy missing the years a rollout reaches gives
    a short dataset rather than an error. The generator checks the staged root
    against the template's before it will point a config at it."""
    import make_eval_config as ev

    with pytest.raises(ev.EvalError, match="stage the rest"):
        _eval(data_root=str(tmp_path))


def test_eval_env_records_which_weights_were_read(tmp_path):
    """An arm still training rewrites best_ckpt.tar whenever validation
    improves, so two evaluations of one arm hours apart can be two models --
    which turns a seed spread into a comparison of epochs. The env records the
    weights' size and mtime so that is checkable rather than assumed."""
    import make_eval_config as ev

    ckpt = tmp_path / "training_checkpoints"
    ckpt.mkdir()
    (ckpt / "best_ckpt.tar").write_bytes(b"weights")
    lines = ev.checkpoint_provenance(str(ckpt / "best_ckpt.tar"))
    assert any(line.startswith("# checkpoint_bytes 7") for line in lines)
    assert any(line.startswith("# checkpoint_mtime 20") for line in lines)
    # a checkpoint that does not exist yet is the normal state before an arm
    # finishes, and must not stop a config being generated
    assert ev.checkpoint_provenance(str(tmp_path / "nope")) == [
        "# checkpoint not present at generation time"
    ]


def test_noise_ladder_brackets_the_trained_amplitude():
    """A ladder with only a downward rung can conclude "not smaller" and
    nothing else. RF01 is mildly under-dispersed at its trained amplitude, so
    the interesting direction is upward and the rungs must bracket 1.0."""
    below = _eval(noise="half")["stepper_override"]["noise"]["scale"]
    above = _eval(noise="double")["stepper_override"]["noise"]["scale"]
    assert below == 0.5
    assert above == 2.0
    assert below < 1.0 < above
