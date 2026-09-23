#!/usr/bin/env python3
"""Generate the sep26v3 atmosphere PILOT campaign into runs/.

    ./make_campaign.py --list          # the run list and the budget
    ./make_campaign.py --all -o runs   # write every run
    ./make_campaign.py --exp BL01      # one experiment, to stdout

sep26v3 trains the same arms as sep26v2 (two arms already defined in the
sep26 campaign -- RO04's word and RO03's R2 counterpart -- plus the ablations
added since) on the FULL continuous 1940-1970 and 1980-1990 record, now that
sep26v2's minimal ~11-year sparse-subset check showed a cheaper dataset was
worth pursuing further. See config-train-atm.template.yaml's header for the
exact train/validation periods and why they are disjoint. The wandb run
group/tag stay pinned to the original sep26v2 identity (WANDB_CAMPAIGN) even
though runids/filenames/output dirs have moved to sep26v3.

    <exp>.<campaign>.<realm>.<factor word>.S<seed>
    BL01 .sep26v3   .atm    .D0_G0_I0_M2_N0_Q0_R4_Y0_Z1.S01

The experiment id is TWO LETTERS naming the study family plus two digits, so a
run id says which question it belongs to before any table is consulted:

    BL  baseline

The factor word is fixed-order and always written out in full, alphabetical by
position (D G I M N Q R Y Z), unchanged from sep26 -- this pilot only varies
the data, not the objective axes, so the same schema and guards apply as-is.

Both halves mirror into W&B: the experiment id is the run group (so seeds
collapse), the factor word is the job type (so arms group), and every factor
token is its own tag (so "every M1 run" is a filter, not a regex).
"""

import argparse
import copy
import dataclasses
import pathlib
import sys
from collections.abc import Mapping

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "sep26v3"
# The wandb run group/tag stay pinned to the sep26v2 identity so the fuller
# sep26v3 dataset re-run of the campaign keeps landing in the same wandb
# "folder" as the original pilot, even though runids/filenames/output dirs
# have moved on to sep26v3 (see env_file()).
WANDB_CAMPAIGN = "sep26v2"
REALM = "atm"
TEMPLATE = HERE / "config-train-atm.template.yaml"

# A W&B project of its own, separate from sep26's, so a pilot run never mixes
# into the real campaign's workspace.
WANDB_PROJECT = "ACE2S-sep26v2-atm"
WANDB_ENTITY = "e3sm-aig"

# Sizing.  The template is built for local batch 1, so ranks == batch_size.
# BATCH is 16 (sep26's full batch size): sep26v3 moved off the minimal-data
# pilot's cheaper BATCH=8 footprint once it started training on the full
# dataset, and 16 is also what makes the 8-point inference blocks divide the
# rank count exactly (see apply_sizing).
BATCH = 16
LOCAL_BATCH = 1
GPUS_PER_NODE = 4
DEFAULT_EPOCHS = 30

INFERENCE_EVALUATIONS = 10
# Per-block rollout length in years, keyed by the inference block's `name` in
# the template. The train-window block scores 2-year rollouts (per the pilot
# design). There is no held-out test block in sep26v3 (removed along with the
# sparse-dataset pilot's test period).
INFERENCE_YEARS_BY_NAME = {"inference": 2}
STEPS_PER_YEAR = 1460  # 6-hourly, noleap

# Inherited from sep26's measurement on aug26's E01 (63.6 h training + 14.2 h
# inline inference + ~3 h setup on 4 nodes / batch 16 / 30 epochs, full
# continuous data) -- the same sizing/data footprint sep26v3 now uses, unlike
# sep26v2's batch 8 / sparse 11-year pilot. Still not re-measured on this
# campaign's own arms, so treat run_hours here as a rough placeholder, not a
# budget you can rely on, until a real run measures it.
TRAIN_HOURS_AT_REL_1 = 63.6
FIXED_HOURS = 17.0


WARM_START_CKPT = "training_checkpoints/best_ckpt.tar"
WARM_START_PLACEHOLDER = "OVERRIDE_ME_WARM_START"

# Isotropic noise at zero channels calls an inverse SHT on a zero-channel tensor
# and dies in the MKL FFT; gaussian at zero channels is a randn of zero size and
# returns cleanly.  Reproduced in aug26.  So Z0 sets the type as well as the
# width, whatever N says.
NOISE_TYPE_AT_ZERO = "gaussian"


class ConfigError(Exception):
    """A word that would produce a config whose run id is a lie, or cannot run."""


# ------------------------------------------------------------ the factor word --
#
# Fixed order, alphabetical by position.  Each entry maps a level DIGIT to the
# value it means in the yaml.  check_campaign.py duplicates these tables on
# purpose and must keep duplicating them.

OBJECTIVE = {"0": "EnsembleLoss", "1": "MSE"}  # D
# G -- (crps_weight, energy_score_weight).  The energy score is taken in
# spectral space through an SHT, so G2 is a different objective in a different
# basis rather than a reweighting of G0.
#
# It is NOT the textbook multivariate energy score.  get_energy_score applies a
# complex modulus at EACH spherical-harmonic coefficient independently and the
# loss averages over coefficients, so it is a sum of per-coefficient marginal
# (bivariate) energy scores.  MEASURED: permuting which member holds which
# value independently at each (channel, mode) leaves the score BIT-IDENTICAL,
# while a true joint score with an L2 norm over all modes moves by 0.19%
# (analysis/loss_semantics.py).  It cannot see cross-mode or cross-channel
# dependence.  Read G as "how much per-mode spectral score", not "joint".
SPLIT = {"0": (0.9, 0.1), "1": (1.0, 0.0), "2": (0.0, 1.0), "3": (0.5, 0.5)}
INIT = {"0": "scratch", "1": "warm"}  # I
MEMBERS = {"1": 1, "2": 2, "3": 3}  # M
NOISE_TYPE = {"0": "isotropic", "1": "gaussian"}  # N
# Q -- multiscale finite-difference CRPS levels, at FDCRPS_WEIGHT.  NOT
# "spatially pooled CRPS": _get_finite_difference_crps_loss takes CRPS of the
# lat and lon ARRAY-INDEX differences, then avg_pool2d by 2 and recurses.  It
# scores grid texture, not an area-weighted physical length scale, and
# FiniteDifferenceCRPSLoss divides by `levels`, so Q3 spreads the same 0.1
# across three scales rather than tripling it.
FDCRPS = {"0": 0, "1": 1, "3": 3}
FDCRPS_WEIGHT = 0.1
# R -- (n_forward_steps, optimize_last_step_only).
#
# NONE of these is backpropagation through time.  optimize_last_step_only runs
# the unscored steps under torch.no_grad (single_module.py:1706), and with
# use_gradient_accumulation (which the template sets) predict_generator detaches
# the state between every step (single_module.py:1167) and accumulate_loss
# backwards each step separately.  So:
#   R1  two steps, first DETACHED under no_grad, only the 12 h state scored
#   R2  two steps, both scored, losses SUMMED (not averaged) over the two
#       horizons, gradients never crossing the step boundary
# R2 - R1 therefore adds the 6 h loss on top of the 12 h one and raises the
# total objective scale; it is not "the second scored step" in isolation.
ROLLOUT: dict[str, tuple[int | dict[int, float], bool]] = {
    "0": (1, True),
    "1": (2, True),
    "2": (2, False),
    "3": ({1: 0.6, 2: 0.2, 4: 0.2}, True),
    "4": ({1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}, True),
}
# Y -- almost_fair_crps_alpha.  get_crps used to hard-code epsilon =
# (1 - alpha) / 2, which agrees with the AIFS-CRPS definition
# (arXiv:2412.15832, (1 - alpha) / n_ensemble) only at M2; it now scales with
# the ensemble size (B5, ported), so Y1 is valid at any member count.
ALPHA = {"0": 1.0, "1": 0.95}
NOISE_DIM = {"0": 0, "1": 32, "2": 64}  # Z

POSITIONS = ("D", "G", "I", "M", "N", "Q", "R", "Y", "Z")
LEVELS: dict[str, Mapping[str, object]] = {
    "D": OBJECTIVE,
    "G": SPLIT,
    "I": INIT,
    "M": MEMBERS,
    "N": NOISE_TYPE,
    "Q": FDCRPS,
    "R": ROLLOUT,
    "Y": ALPHA,
    "Z": NOISE_DIM,
}
# The template's own levels.  RF01 is exactly this word.
BASELINE = {
    "D": "0",
    "G": "0",
    "I": "0",
    "M": "2",
    "N": "0",
    "Q": "0",
    "R": "0",
    "Y": "0",
    "Z": "1",
}

# Relative training cost.  Inherited from sep26 (analysis/card-sweep.sh,
# measured 2026-09-03 on batch 16 / the full continuous dataset) -- the same
# footprint sep26v3 now uses. Still not re-measured on this campaign's own
# arms -- see the TRAIN_HOURS_AT_REL_1 note above. Kept only so
# `rel`/`run_hours` print something in --list; do not treat them as a real
# budget for this pilot.
REL_MEMBERS = {"1": 0.476, "2": 1.0, "3": 1.435}
REL_ROLLOUT = {
    "0": 1.0,
    "1": 1.21,
    "2": 1.89,
    "3": 1.21,
    "4": 1.52,
}

STUDIES = {
    "BL": "baseline",
    "LG": "loss geometry",
    "RO": "rollout",
    "EN": "ensemble size",
    "NC": "noise conditioning",
}


def _rollout_steps(level: str) -> object:
    """The yaml value for R.

    A fixed rollout is a bare int.  A sampled one is a
    TimeLengthProbabilities, whose schema is an `outcomes` list of
    {steps, probability} -- NOT the bare {steps: probability} mapping the
    table below is written as.  fme.ace.validate_config catches the bare
    mapping with an unhelpful UnionMatchError, so the conversion lives here.
    """
    steps, _ = ROLLOUT[level]
    if isinstance(steps, int):
        return steps
    return {
        "outcomes": [{"steps": n, "probability": p} for n, p in sorted(steps.items())]
    }


@dataclasses.dataclass(frozen=True)
class Word:
    """A full factor word.  Unspecified positions take the template's level."""

    levels: dict[str, str]

    @classmethod
    def of(cls, **kw: str) -> "Word":
        levels = dict(BASELINE)
        for pos, level in kw.items():
            pos = pos.upper()
            if pos not in LEVELS:
                raise ConfigError(f"unknown position {pos!r}; known: {POSITIONS}")
            if level not in LEVELS[pos]:
                raise ConfigError(
                    f"unknown level {pos}{level}; known: "
                    f"{sorted(pos + k for k in LEVELS[pos])}"
                )
            levels[pos] = level
        return cls(levels)

    def get(self, pos: str) -> str:
        return self.levels[pos]

    def word(self) -> str:
        return "_".join(f"{p}{self.levels[p]}" for p in POSITIONS)

    def tokens(self) -> list[str]:
        return [f"{p}{self.levels[p]}" for p in POSITIONS]

    @property
    def is_baseline(self) -> bool:
        return self.levels == BASELINE


@dataclasses.dataclass(frozen=True)
class Experiment:
    exp: str  # <2 letters><2 digits>, e.g. LG01
    word: Word
    note: str
    seeds: tuple[int, ...] = (1,)
    priority: int = 3
    epochs: int = DEFAULT_EPOCHS
    warm_start_from: str = ""  # an experiment id, resolved at submit time
    # The two waste guards below refuse arms that spend a whole run on a
    # configuration that cannot learn what it looks like it is learning.  A
    # mechanism probe deliberately spends that, and says so here rather than in
    # someone's memory: the flag reaches the run notes and the .env.
    allow_degenerate: bool = False

    def __post_init__(self):
        if len(self.exp) != 4 or self.exp[:2] not in STUDIES:
            raise ConfigError(
                f"experiment id {self.exp!r} must be two study letters "
                f"{sorted(STUDIES)} plus two digits"
            )
        if not self.exp[2:].isdigit():
            raise ConfigError(f"experiment id {self.exp!r} must end in two digits")


@dataclasses.dataclass(frozen=True)
class Run:
    experiment: Experiment
    seed: int

    @property
    def word(self) -> Word:
        return self.experiment.word

    @property
    def runid(self) -> str:
        return (
            f"{self.experiment.exp}.{CAMPAIGN}.{REALM}."
            f"{self.word.word()}.S{self.seed:02d}"
        )

    @property
    def ranks(self) -> int:
        return BATCH // LOCAL_BATCH

    @property
    def nodes(self) -> int:
        nodes, rem = divmod(self.ranks, GPUS_PER_NODE)
        if rem or nodes < 1:
            raise ConfigError(f"{self.runid}: {self.ranks} ranks is not whole nodes")
        return nodes

    @property
    def rel(self) -> float:
        return REL_MEMBERS[self.word.get("M")] * REL_ROLLOUT[self.word.get("R")]

    @property
    def run_hours(self) -> float:
        scale = self.experiment.epochs / DEFAULT_EPOCHS
        return TRAIN_HOURS_AT_REL_1 * self.rel * scale + FIXED_HOURS


# ------------------------------------------------------------------ the runs --
#
# The two arms this pilot opened with (BL01, BL02), now at three seeds each so
# the sweep below has an error bar to read against, plus a first sweep across
# the objective's other axes -- loss geometry, rollout depth, ensemble size and
# noise conditioning -- all on sep26v3's full continuous 1940-1970/1980-1990
# dataset (sep26v2's minimal 11-year sparse subset, superseded here).
# Every word below reuses a level combination already understood from sep26
# (see that campaign's RUNLIST for the LG/RO/EN/NC precedents); the only new
# thing here is running them against this pilot's dataset, at R4
# (BL01's sampled 20-step rollout) rather than sep26's R0 default, so each new
# arm differs from a baseline by exactly the one axis its study name claims.

RUNLIST: list[Experiment] = [
    # -- baselines ------------------------------------------------------
    Experiment(
        "BL01",
        Word.of(R="4"),
        "sep26's RO04 word (sampled rollout to 20 steps, pure CRPS at two "
        "members) on sep26v3's full continuous dataset -- one of this "
        "pilot's two baselines, rerun here after sep26v2's minimal 11-year "
        "sparse-dataset check. Three seeds for an error bar.",
        seeds=(1, 47, 82),
        priority=1,
    ),
    Experiment(
        "BL02",
        Word.of(D="1", M="1", R="2", Z="0"),
        "sep26's RO03 word with R2 instead of R1 (two steps, both scored) on "
        "the deterministic row (MSE, one member, no noise), on sep26v3's "
        "full continuous dataset -- this pilot's other baseline, rerun here "
        "after sep26v2's minimal 11-year sparse-dataset check. Three seeds "
        "for an error bar.",
        seeds=(1, 47, 82),
        priority=1,
    ),
    # -- loss geometry: MAE (M1 CRPS) with noise wired but nothing in the
    # M1 objective able to reward it, at BL01's rollout.
    Experiment(
        "LG02",
        Word.of(G="1", M="1", R="4"),
        "CRPS with a single ensemble member (MAE), with noise wired in but "
        "nothing in the M1 objective able to reward it, at BL01's rollout.",
        seeds=(1, 47, 82),
        priority=2,
        allow_degenerate=True,
    ),
    # -- rollout: the stochastic and deterministic poles at R2 ----------
    Experiment(
        "RO01",
        Word.of(R="2"),
        "stochastic model (BL01's members/objective) at two steps, both "
        "scored, losses summed -- mimicking ACE2 training. RO01 - BL01 "
        "reads the sampled-vs-fixed rollout choice at a fixed two-step depth.",
        seeds=(1, 47, 82),
        priority=2,
    ),
    Experiment(
        "RO02",
        Word.of(D="1", M="1", R="4", Z="0"),
        "deterministic baseline (BL02's objective) extended to BL01's "
        "sampled 20-step rollout instead of BL02's fixed two steps -- "
        "RO02 - BL02 reads the rollout-depth effect on the deterministic "
        "pole, the same axis RO01 reads on the stochastic one.",
        seeds=(1, 47, 82),
        priority=2,
    ),
    # -- ensemble size, at BL01's rollout --------------------------------
    Experiment(
        "EN02",
        Word.of(M="3", R="4"),
        "three-member stochastic ensemble at BL01's full objective (CRPS + "
        "spectral energy score) and rollout -- EN02 - BL01 is ensemble size "
        "alone.",
        seeds=(1, 47, 82),
        priority=3,
    ),
    # -- noise conditioning, at BL01's rollout ---------------------------
    Experiment(
        "NC01",
        Word.of(Z="2", R="4"),
        "64-dim noise embedding added to the stochastic model (BL01's "
        "objective), against BL01's 32-dim default -- NC01 - BL01 is noise "
        "width alone.",
        seeds=(1, 47, 82),
        priority=3,
    ),
]


# ------------------------------------------------------------------- guards --


def validate(run: Run) -> list[str]:
    """Complaints about a word, empty if it is sound.

    Everything here is a property of the code, verified against this branch on
    2026-09-03.  A word that trips one of these produces a config whose run id
    claims something the run does not do, or that raises on its first batch.
    """
    bad: list[str] = []
    w = run.word
    d, g, i = w.get("D"), w.get("G"), w.get("I")
    m, q = w.get("M"), w.get("Q")
    y, z = w.get("Y"), w.get("Z")

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(f"{run.runid}: {msg}")

    crps_w, energy_w = SPLIT[g]
    ensemble = d == "0"

    # BLOCKER 1 and BLOCKER 2 -- LIFTED 2026-09-03.  Both used to refuse
    # configurations that raise on the FIRST TRAINING BATCH, after config
    # validation, after dataset construction, after the model is built:
    #
    #   * any member count but two with an energy-score weight
    #     (get_energy_score raised NotImplementedError -- aug26's E25 and E26),
    #   * crps_weight 0, leaving the energy score alone with a shape carrying
    #     two spurious leading dims (get_channel_losses raised "Per-channel
    #     loss has 1 elements but 50 channel names were provided").
    #
    # The upstream fixes are now on this branch, and because both faults were
    # first-batch faults, unit tests were not the standard for lifting the
    # guards.  Both were re-run for real on a GPU node:
    #
    #   M3 + energy_score_weight 0.1   209 steps, loss 4.0444 -> 0.3705,
    #                                  zero NotImplementedError
    #   crps_weight 0 / energy 1.0     250 steps, loss 1.1952 -> 0.2952,
    #                                  zero "Per-channel loss has" errors
    #
    # See PLAN.md 12.  What remains refused below is degeneracy and
    # truth-in-labelling, not upstream breakage.

    # LossConfig.build discards every EnsembleLoss kwarg for MSE, so these would
    # parse, run, and read as a lie in the file.
    if not ensemble:
        for pos, level in (("G", g), ("Q", q), ("Y", y)):
            want(
                level == BASELINE[pos],
                f"D1 with {pos}{level}: LossConfig.build discards every "
                f"EnsembleLoss kwarg for MSE, so the id would claim a setting "
                f"the run does not have",
            )

    # almost_fair_crps_alpha only parameterizes the CRPS module, which
    # EnsembleLoss.forward gates on crps_weight > 0.  The finite-difference term
    # is gated on its own weight alone.  The asymmetry is easy to get backwards.
    want(
        not (ensemble and crps_w == 0.0 and y != BASELINE["Y"]),
        f"Y{y} at crps_weight 0: almost_fair_crps_alpha is inert there",
    )

    # A warm start needs a parent, and the parent has to be in the run list.
    if i == "1":
        want(
            bool(run.experiment.warm_start_from),
            "I1 without warm_start_from: the .env would carry no parent and "
            "run-train.sh would have nothing to resolve",
        )
    else:
        want(
            not run.experiment.warm_start_from,
            f"warm_start_from={run.experiment.warm_start_from!r} with I0: the "
            f"parent would be recorded and then not used",
        )

    # BLOCKER 3.  With Z0 there are no noise channels, so the model is a
    # deterministic function of its input and broadcast_ensemble hands every
    # member the same input: the members come out BIT-IDENTICAL.  MEASURED on
    # CPU, where kernels are deterministic: max|member0 - member1| is exactly
    # 0.0, CRPS(M=2) equals MAE to the last bit, and the energy score's
    # dispersion term is exactly zero (analysis/z0_degeneracy.py).
    #
    # So pure CRPS (energy_w == 0) at Z0 with more than one member optimises
    # bit-for-bit the same objective as its M1 twin at M times the cost.  There
    # is no opt-in for this one: it buys literally nothing.
    want(
        not (ensemble and z == "0" and energy_w == 0.0 and m != "1"),
        f"M{m} with Z0 and G{g}: no noise channels means the members are "
        f"bit-identical, so the pairwise CRPS term is exactly zero and this is "
        f"the M1 arm's objective at {m}x the cost. Use M1",
    )

    # N is inert at Z0 -- a zero-channel noise tensor is drawn no matter what
    # the type says -- so a word claiming N1 there would be a lie.  (Z0 also
    # forces the builder to gaussian: isotropic at zero channels dies in the
    # MKL FFT.)
    want(
        not (z == "0" and w.get("N") != BASELINE["N"]),
        f"N{w.get('N')} with Z0: there are no noise channels, so no noise of "
        f"either type is drawn and the token would claim a setting that has no "
        f"effect. Leave N at {BASELINE['N']}",
    )

    # The Y1-only-at-M2 guard is LIFTED too: get_crps now scales epsilon with
    # the ensemble size, so almost-fair CRPS is the AIFS definition at every
    # member count, checked against it analytically at M = 1, 2, 3 and 5.
    #
    # Weaker evidence than the two above, and deliberately labelled as such:
    # this one was not re-run on a node.  It is a scalar coefficient inside the
    # CRPS term, which every D0 arm already exercises on every batch, so there
    # is no unexercised code path for a first-batch fault to hide in -- unlike
    # BLOCKER 1 and 2, which gated whole branches of the loss.

    # Two waste guards.  Both are right about the waste, and the LG block
    # deliberately spends it, so they are gated on an explicit opt-in that
    # reaches the artifacts rather than deleted.
    if ensemble and z == "0" and not run.experiment.allow_degenerate:
        bad.append(
            f"{run.runid}: Z0 with D0 scores a degenerate ensemble -- the "
            f"members are bit-identical, so CRPS collapses to MAE and the "
            f"energy score to a spectral L1 distance. Set allow_degenerate="
            f"True if that is the point"
        )
    if not ensemble and m == "1" and z != "0" and not run.experiment.allow_degenerate:
        bad.append(
            f"{run.runid}: Z{z} with D1/M1 wires up noise conditioning that "
            f"nothing in the objective can reward. Set allow_degenerate=True "
            f"if that is the point"
        )
    return bad


# -------------------------------------------------------------------- build --


def apply_word(config: dict, run: Run) -> None:
    """Map the factor word onto the yaml.  The only place that knows the schema."""
    w = run.word
    training = config["stepper_training"]
    builder = config["stepper"]["step"]["config"]["builder"]["config"]

    training["n_ensemble"] = MEMBERS[w.get("M")]
    _, last_only = ROLLOUT[w.get("R")]
    training["n_forward_steps"] = _rollout_steps(w.get("R"))
    training["optimize_last_step_only"] = last_only

    noise = NOISE_DIM[w.get("Z")]
    builder["noise_embed_dim"] = noise
    # Z0 forces the type regardless of N: isotropic at zero channels dies in the
    # MKL FFT.  The checker asserts the resolved value, not N.
    builder["noise_type"] = NOISE_TYPE_AT_ZERO if noise == 0 else NOISE_TYPE[w.get("N")]

    loss = training["loss"]
    loss["type"] = OBJECTIVE[w.get("D")]
    if w.get("D") == "1":
        # LossConfig.build ignores kwargs for MSE; leaving EnsembleLoss weights
        # in the file would be dead text that reads as if it applied.
        loss.pop("kwargs", None)
    else:
        crps_w, energy_w = SPLIT[w.get("G")]
        kwargs: dict[str, object] = {
            "crps_weight": crps_w,
            "energy_score_weight": energy_w,
        }
        if w.get("Q") != BASELINE["Q"]:
            kwargs["finite_difference_crps_weight"] = FDCRPS_WEIGHT
            kwargs["finite_difference_crps_levels"] = FDCRPS[w.get("Q")]
        if w.get("Y") != BASELINE["Y"]:
            kwargs["almost_fair_crps_alpha"] = ALPHA[w.get("Y")]
        loss["kwargs"] = kwargs

    if w.get("I") == "1":
        # A run id in the .env, never a path: runs/ must be byte-identical for
        # every teammate, and the parent's checkpoint lives under whichever
        # $CAMPAIGN_ROOT owns it.  run-train.sh resolves it at submit time and
        # refuses if the checkpoint is absent.
        training.setdefault("parameter_init", {})["weights_path"] = (
            WARM_START_PLACEHOLDER
        )


def apply_epoch_schedule(config: dict, epochs: int) -> None:
    """Give every inference block the same schedule, ending on the final epoch.

    FME fires a block on ``list(range(1, max_epochs + 1))[start::step]``.  The
    range starts at 1 because evaluate_before_training is off; solving `start`
    against a range one element longer is how aug26 once arranged for the final
    epoch never to be scored.
    """
    config["max_epochs"] = epochs
    step = max(1, epochs // INFERENCE_EVALUATIONS)
    start = (epochs - 1) % step
    fires = list(range(1, epochs + 1))[start::step]
    if not fires or fires[-1] != epochs:
        raise ConfigError(f"epoch schedule misses the last epoch: {fires[-3:]}")
    for block in config.get("inference", []):
        block["epochs"] = {"start": start, "step": step}


def apply_sizing(config: dict, run: Run) -> None:
    """Batch sizes, and make every inference IC count divide the rank count.

    Done here, on a login node, because the evaluator checks it too late to be
    useful: InlineInferenceConfig raises readably but InferenceEvaluatorConfig
    has no such check at all, so a mismatch there surfaces as a bare
    AssertionError inside the data loader, minutes into an allocation.
    """
    config["train_loader"]["batch_size"] = BATCH
    config["validation"]["loader"]["batch_size"] = BATCH
    for block in config.get("inference", []):
        n = len(block["loader"]["start_indices"]["times"])
        if n % run.ranks:
            raise ConfigError(
                f"{run.runid}: block {block.get('name')!r} has {n} initial "
                f"conditions, which do not divide {run.ranks} ranks"
            )


def build(baseline: dict, run: Run) -> dict:
    complaints = validate(run)
    if complaints:
        raise ConfigError("\n".join(complaints))
    config = copy.deepcopy(baseline)
    config["seed"] = run.seed
    config["experiment_dir"] = "OVERRIDE_ME"
    config["logging"]["project"] = WANDB_PROJECT
    config["logging"]["entity"] = WANDB_ENTITY
    apply_word(config, run)
    apply_sizing(config, run)
    apply_epoch_schedule(config, run.experiment.epochs)
    for block in config.get("inference", []):
        name = block.get("name")
        if name not in INFERENCE_YEARS_BY_NAME:
            raise ConfigError(
                f"{run.runid}: inference block {name!r} has no entry in "
                f"INFERENCE_YEARS_BY_NAME"
            )
        block["n_forward_steps"] = INFERENCE_YEARS_BY_NAME[name] * STEPS_PER_YEAR
    return config


def parent_runid(
    exp_id: str, seed: int, runlist: list[Experiment] | None = None
) -> str:
    """Resolve a warm-start parent from an experiment id to a full run id.

    The run list names a parent by experiment id (RF02) because that is what a
    human writing an arm knows.  run-train.sh resolves
    $CAMPAIGN_ROOT/<parent>/training_checkpoints/, which is keyed by RUN id, so
    the .env has to carry the full one -- otherwise the resolution points at a
    directory that never exists and the arm fails closed forever.

    A parent with several seeds contributes the matching seed where it has one,
    and its first otherwise, so a seeded curriculum stays paired.
    """
    for e in runlist if runlist is not None else RUNLIST:
        if e.exp == exp_id:
            return Run(e, seed if seed in e.seeds else e.seeds[0]).runid
    return exp_id  # expand() rejects an unknown parent before this matters


def env_file(run: Run) -> str:
    """W&B provenance, read from the environment by wandb rather than the yaml.

    Deliberately free of identity: who submitted a run and where its output
    landed are properties of the submission, not of the run list.  Baking either
    in made aug26's runs/ differ for every teammate, which dirtied the worktree
    the moment anyone regenerated -- and the submit script refuses a dirty
    worktree, so only the generator's author could launch.
    """
    e = run.experiment
    # Every factor token is its own tag, so "every M1 run" is a filter rather
    # than a regex, and the two-letter study prefix groups a whole question.
    tags = [
        WANDB_CAMPAIGN,
        REALM,
        e.exp,
        e.exp[:2],
        f"S{run.seed:02d}",
        f"P{e.priority}",
        *run.word.tokens(),
    ]
    if e.allow_degenerate:
        tags.append("degenerate-by-design")
    warm = (
        [
            f"FME_WARM_START_FROM={parent_runid(e.warm_start_from, run.seed)}",
            f"FME_WARM_START_CKPT={WARM_START_CKPT}",
        ]
        if e.warm_start_from
        else []
    )
    note = e.note
    if e.allow_degenerate:
        note += (
            " [degenerate by design: this arm deliberately spends the waste the "
            "generator's guards exist to prevent]"
        )
    return "\n".join(
        [
            f"# generated by make_campaign.py -- {e.exp}, {STUDIES[e.exp[:2]]}",
            f"FME_NODES={run.nodes}",
            f"FME_RANKS={run.ranks}",
            f"FME_PRIORITY={e.priority}",
            *warm,
            f"WANDB_NAME={run.runid}",
            # The run group collapses seeds; the job type groups arms. Pinned
            # to WANDB_CAMPAIGN (sep26v2), not CAMPAIGN, so this stays the
            # same wandb group across the sep26v2 -> sep26v3 dataset change.
            f"WANDB_RUN_GROUP={WANDB_CAMPAIGN}.{REALM}.{e.exp}",
            f"WANDB_JOB_TYPE={run.word.word()}",
            f"WANDB_TAGS={','.join(tags)}",
            f'WANDB_NOTES="{note} | {run.nodes} nodes, {run.ranks} ranks"',
            "",
        ]
    )


def expand(runlist: list[Experiment]) -> list[Run]:
    runs = [Run(e, s) for e in runlist for s in e.seeds]
    seen: dict[str, str] = {}
    ids: dict[str, str] = {}
    for r in runs:
        if r.runid in seen:
            raise ConfigError(
                f"two experiments render the same run id {r.runid}: "
                f"{seen[r.runid]} and {r.experiment.exp}"
            )
        seen[r.runid] = r.experiment.exp
        key = r.word.word() + f".S{r.seed:02d}"
        if key in ids and ids[key] != r.experiment.exp:
            raise ConfigError(
                f"{ids[key]} and {r.experiment.exp} are the same configuration "
                f"({key}); give one experiment id to one configuration"
            )
        ids[key] = r.experiment.exp
    parents = {e.exp for e in runlist}
    for e in runlist:
        if e.warm_start_from and e.warm_start_from not in parents:
            raise ConfigError(
                f"{e.exp} warm-starts from {e.warm_start_from}, which is not in "
                f"the run list"
            )
    return runs


def manifest(runs: list[Run]) -> str:
    head = (
        "runid\texp\tstudy\tword\tseed\tpriority\tepochs\tnodes\trel\t"
        "run_hours\twarm_start_from\tnote"
    )
    rows = [
        "\t".join(
            [
                r.runid,
                r.experiment.exp,
                STUDIES[r.experiment.exp[:2]],
                r.word.word(),
                f"{r.seed:02d}",
                str(r.experiment.priority),
                str(r.experiment.epochs),
                str(r.nodes),
                f"{r.rel:.2f}",
                f"{r.run_hours:.0f}",
                r.experiment.warm_start_from,
                r.experiment.note.replace("\t", " ").replace("\n", " "),
            ]
        )
        for r in runs
    ]
    return "\n".join([head, *rows]) + "\n"


def report(runs: list[Run]) -> None:
    print(f"{CAMPAIGN} {REALM} -- {len(runs)} runs, W&B project {WANDB_PROJECT}\n")
    w = max(len(r.runid) for r in runs)
    print(
        f"{'exp':<5} {'run id':<{w}} {'pri':>3} {'nodes':>5} {'rel':>5} "
        f"{'run h':>6} {'node h':>7}  study"
    )
    for r in sorted(
        runs, key=lambda r: (r.experiment.priority, r.experiment.exp, r.seed)
    ):
        print(
            f"{r.experiment.exp:<5} {r.runid:<{w}} {r.experiment.priority:>3} "
            f"{r.nodes:>5} {r.rel:>5.2f} {r.run_hours:>6.0f} "
            f"{r.nodes * r.run_hours:>7.0f}  {STUDIES[r.experiment.exp[:2]]}"
        )
    nodes = sum(r.nodes for r in runs)
    node_h = sum(r.nodes * r.run_hours for r in runs)
    print(
        f"\n{len(runs)} runs, {nodes} nodes concurrent, {node_h:,.0f} node-hours, "
        f"critical path {max(r.run_hours for r in runs):.0f} h"
    )
    by_pri: dict[int, float] = {}
    for r in runs:
        by_pri[r.experiment.priority] = (
            by_pri.get(r.experiment.priority, 0) + r.nodes * r.run_hours
        )
    cum = 0.0
    for p in sorted(by_pri):
        cum += by_pri[p]
        print(f"  P{p}: {by_pri[p]:>7,.0f} node-h   cumulative {cum:>7,.0f}")
    print(
        "\nrun_hours are inherited sep26 estimates, NOT measured for this "
        "campaign's own arms -- treat them as a rough placeholder until a "
        "real run measures it."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="write every run")
    p.add_argument("--exp", action="append", default=[], help="one experiment id")
    p.add_argument("--list", action="store_true", help="print the run list only")
    p.add_argument("-o", "--out", default=None, help="output directory")
    p.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    args = p.parse_args(argv)

    runlist = RUNLIST
    if args.exp:
        runlist = [e for e in RUNLIST if e.exp in args.exp]
        if not runlist:
            print(f"no experiment matches {args.exp}", file=sys.stderr)
            return 2
    if args.epochs:
        runlist = [dataclasses.replace(e, epochs=args.epochs) for e in runlist]

    runs = expand(runlist)
    if args.list:
        report(runs)
        return 0

    complaints = [c for r in runs for c in validate(r)]
    if complaints:
        print("\n".join(complaints), file=sys.stderr)
        return 1

    baseline = yaml.safe_load(TEMPLATE.read_text())
    if args.out is None:
        for r in runs:
            yaml.safe_dump(build(baseline, r), sys.stdout, sort_keys=False, width=100)
        return 0

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for r in runs:
        (out / f"{r.runid}.yaml").write_text(
            yaml.safe_dump(
                build(baseline, r), sort_keys=False, default_flow_style=False, width=100
            )
        )
        (out / f"{r.runid}.env").write_text(env_file(r))
    (out / "MANIFEST.tsv").write_text(manifest(runs))
    print(f"wrote {len(runs)} runs to {out}")
    report(runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
