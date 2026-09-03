#!/usr/bin/env python3
"""Generate the sep26 atmosphere ablation campaign into runs/.

    ./make_campaign.py --list          # the run list and the budget
    ./make_campaign.py --all -o runs   # write every run
    ./make_campaign.py --label M1      # one experiment, to stdout

sep26 varies only the *training objective*: the loss family, its internals, the
member count, the noise conditioning and the rollout.  Everything else -- the
channels, the batch size, the learning rate, the loss weighting, the data -- is
held at the aug26 E01 tuning set, which is what makes aug26's E01 and E21 the
references for this campaign and why no baseline has to be re-run here.

Run ids are a **sparse, canonically sorted delta** from the template:

    sep26.atm.base.s01
    sep26.atm.crps-pure.s01
    sep26.atm.crps-pure_mem-1_noise-0.s01

Only axes that differ from the template appear.  That is the whole design: a
new axis added to LEVELS does not appear in any id that does not use it, so
adding one cannot rename an existing run.  aug26 needed a rule and a checker to
get that property; here it falls out of the encoding.  See PLAN.md section 3.

The delta is also the identity -- there is no experiment number.  Inserting an
arm renumbers nothing.  Short human handles live in the `label` field, which
reaches MANIFEST.tsv and the W&B tags but never the id or the directory name,
so a handle can be corrected without moving a byte on disk.
"""

import argparse
import copy
import dataclasses
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "sep26"
REALM = "atm"
TEMPLATE = HERE / "config-train-atm.template.yaml"

# Sizing.  The template is built for local batch 1, so ranks == batch_size.
LOCAL_BATCH = 1
GPUS_PER_NODE = 4
DEFAULT_EPOCHS = 30

# Inference blocks all fire the same number of times, ending on the last epoch.
INFERENCE_EVALUATIONS = 10
INFERENCE_YEARS = 5
STEPS_PER_YEAR = 1460  # 6-hourly, noleap

# Cost model, measured on aug26's E01: 63.6 h of training plus 14.2 h of inline
# inference plus ~3 h of setup = 81 h on 4 nodes for 30 epochs.  `rel` scales
# the training term only, because inline inference is a single-member rollout --
# neither inference block sets n_ensemble_per_ic, so it defaults to 1 and is
# independent of stepper_training.n_ensemble.  Verified 2026-09-03.
TRAIN_HOURS_AT_REL_1 = 63.6
FIXED_HOURS = 17.0

WARM_START_CKPT = "training_checkpoints/best_ckpt.tar"
WARM_START_PLACEHOLDER = "OVERRIDE_ME_WARM_START"

# Isotropic noise at zero channels calls an inverse SHT on a zero-channel tensor
# and dies in the MKL FFT ("Intel oneMKL DFTI ERROR: Inconsistent configuration
# parameters"); gaussian at zero channels is a randn of zero size and returns
# cleanly.  Reproduced in aug26.  So noise-0 sets the type as well as the width.
NOISE_TYPE_AT_ZERO = "gauss"


class ConfigError(Exception):
    """A delta that would produce a config whose id is a lie, or that cannot run."""


# ------------------------------------------------------------------ the axes --

# LEVELS is the id vocabulary: the ordered level names for each axis, first
# entry being the template's value and therefore the one omitted from a run id.
# The typed tables below map a level to what it means in the yaml.  Keeping the
# two apart is what lets `mypy` see real types instead of `object`, and it
# mirrors check_campaign.py, which duplicates these tables on purpose.
LEVELS: dict[str, tuple[str, ...]] = {
    "obj": ("crps", "mse"),
    "crps": ("std", "pure", "energy", "half"),
    "mem": ("2", "1", "3"),
    "noise": ("32", "0", "64"),
    "ntype": ("iso", "gauss"),
    "roll": ("f1", "c2", "f2", "s04", "s20"),
    "fdcrps": ("0", "1", "3"),
    "alpha": ("100", "095"),
}

LOSS_TYPE: dict[str, str] = {"crps": "EnsembleLoss", "mse": "MSE"}
# (crps_weight, energy_score_weight).  The energy score is computed in spectral
# space through an SHT, so `energy` is a different objective in a different
# basis rather than a reweighting of `crps`.
CRPS_WEIGHTS: dict[str, tuple[float, float]] = {
    "std": (0.9, 0.1),
    "pure": (1.0, 0.0),
    "energy": (0.0, 1.0),
    "half": (0.5, 0.5),
}
MEMBERS: dict[str, int] = {"1": 1, "2": 2, "3": 3}
NOISE: dict[str, int] = {"0": 0, "32": 32, "64": 64}
NTYPE: dict[str, str] = {"iso": "isotropic", "gauss": "gaussian"}
# (n_forward_steps, optimize_last_step_only).  optimize_last_step_only runs the
# unscored steps under torch.no_grad, so an n-step sample costs (n-1) forwards
# plus one forward+backward.  Measured, that forward is 0.24 of a scored step;
# see REL_ROLL below.
ROLL: dict[str, tuple[object, bool]] = {
    "f1": (1, True),
    "c2": (2, True),
    "f2": (2, False),
    "s04": ({1: 0.6, 2: 0.2, 4: 0.2}, True),
    "s20": ({1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}, True),
}
# finite_difference_crps_levels, at weight 0.1.  Level "0" is the term off.
FDCRPS: dict[str, int] = {"0": 0, "1": 1, "3": 3}
FDCRPS_WEIGHT = 0.1
# almost_fair_crps_alpha x 100.
ALPHA: dict[str, float] = {"100": 1.0, "095": 0.95}

BASELINE = {axis: names[0] for axis, names in LEVELS.items()}

# Relative training cost per level, against the template.  MEASURED 2026-09-03
# by analysis/card-sweep.sh, not assumed: the two card types agree on every
# figure below to within 2%, which is the check that these are properties of the
# model rather than of the node.
#
# Fitting  batch_time = fixed + n_scored * step + n_unscored * forward  to the
# sweep gives, on the 40 GB card, fixed = 0.093 s, a scored step of 0.810 s and
# a no_grad forward of 0.195 s -- so
#
#     a no_grad forward is 0.24 of a scored step, not the 1/3 aug26 assumed.
#
# That makes every last-step-only rollout cheaper than the arithmetic budgeted,
# and it is the difference between roll-c2 costing 102 h and 94 h.
REL_MEM = {"1": 0.476, "2": 1.0, "3": 1.435}  # measured; arithmetic said 0.5 / 1.5
REL_ROLL = {
    "f1": 1.0,
    "c2": 1.21,  # measured; arithmetic said 1.33
    "f2": 1.89,  # measured; arithmetic said 2.00
    # Not measured directly.  The 1->2 step slope (0.195 s per extra no_grad
    # forward) UNDER-predicts at depth: a fixed 20-step rollout was measured at
    # 5.355 s/batch against a linear 4.608, so the per-forward cost grows with
    # rollout length.  Fitting through the two extreme measured points
    # (1 step 0.903 s, 20 steps 5.355 s) gives 0.234 s per extra forward, which
    # reproduces the measured 2-step point to 3.5%.  The sampled schedules are
    # then the probability-weighted sum over that curve, which is what the
    # arithmetic version of this table got wrong in both directions.
    "s04": 1.21,  # {1:.6, 2:.2, 4:.2};              arithmetic said 1.20
    "s20": 1.52,  # {1:.6, 2:.2, 4:.1, 12:.05, 20:.05}; arithmetic said 1.67
}
# Per-level multipliers for axes whose cost is not captured above.  Anything
# absent is 1.0.  fdcrps and ntype are still arithmetic; the sweep variants that
# settle them were queued behind the rollout arms.
REL_EXTRA: dict[tuple[str, str], float] = {}


@dataclasses.dataclass(frozen=True)
class Delta:
    """A sparse set of non-baseline axis levels."""

    levels: tuple[tuple[str, str], ...] = ()

    @classmethod
    def of(cls, **kw: str) -> "Delta":
        for k, v in kw.items():
            if k not in LEVELS:
                raise ConfigError(f"unknown axis {k!r}; known: {sorted(LEVELS)}")
            if v not in LEVELS[k]:
                raise ConfigError(
                    f"unknown level {v!r} for axis {k!r}; "
                    f"known: {sorted(LEVELS[k])}"
                )
        # Drop baseline levels and sort, so the same delta always renders the
        # same word however it was written.  This is what removes the ordering
        # question that forced aug26 into a second and third factor word.
        sparse = tuple(sorted((k, v) for k, v in kw.items() if BASELINE[k] != v))
        return cls(sparse)

    def get(self, axis: str) -> str:
        return dict(self.levels).get(axis, BASELINE[axis])

    def word(self) -> str:
        return "_".join(f"{k}-{v}" for k, v in self.levels) or "base"

    @property
    def is_default(self) -> bool:
        return not self.levels


@dataclasses.dataclass(frozen=True)
class Experiment:
    label: str  # a short handle for slides; never part of the id
    delta: Delta
    note: str
    seeds: tuple[int, ...] = (1,)
    priority: int = 2
    epochs: int = DEFAULT_EPOCHS
    warm_start_from: str = ""
    # The two guards below refuse arms that waste a whole run.  A mechanism
    # probe deliberately trips them, and says so here rather than in someone's
    # memory: the flag reaches the run notes and the .env.
    allow_degenerate: bool = False


@dataclasses.dataclass(frozen=True)
class Run:
    exp: Experiment
    seed: int

    @property
    def delta(self) -> Delta:
        return self.exp.delta

    @property
    def runid(self) -> str:
        return f"{CAMPAIGN}.{REALM}.{self.delta.word()}.s{self.seed:02d}"

    @property
    def batch(self) -> int:
        return 16  # the E01 tuning set; sep26 does not vary it

    @property
    def ranks(self) -> int:
        return self.batch // LOCAL_BATCH

    @property
    def nodes(self) -> int:
        nodes, rem = divmod(self.ranks, GPUS_PER_NODE)
        if rem or nodes < 1:
            raise ConfigError(f"{self.runid}: {self.ranks} ranks is not whole nodes")
        return nodes

    @property
    def rel(self) -> float:
        r = REL_MEM[self.delta.get("mem")] * REL_ROLL[self.delta.get("roll")]
        for k, v in self.delta.levels:
            r *= REL_EXTRA.get((k, v), 1.0)
        return r

    @property
    def run_hours(self) -> float:
        scale = self.exp.epochs / DEFAULT_EPOCHS
        return TRAIN_HOURS_AT_REL_1 * self.rel * scale + FIXED_HOURS


# ------------------------------------------------------------------ the runs --

RUNLIST: list[Experiment] = [
    # -- Tier 1: decompose REF-S minus REF-D ---------------------------------
    # aug26's E01 (stochastic pole) and E21 (deterministic pole) are the
    # references and are NOT re-run here.  E01-E21 moves the loss family, the
    # noise conditioning and the member count at once; this block holds the
    # member count at one and crosses the other two, so the row difference is
    # loss geometry and the column difference is noise conditioning with nothing
    # in the objective rewarding it.
    #
    # The D0 row must be crps-pure, not the template's 0.9/0.1: get_energy_score
    # raises at any member count but two, so a one- or three-member arm with an
    # energy-score weight cannot train.  See PLAN.md section 4.
    Experiment(
        "M01",
        Delta.of(obj="mse", mem="1"),
        "noise conditioning under MSE; the D1/Z32 cell of the mechanism square",
        priority=10,
        allow_degenerate=True,
    ),
    Experiment(
        "M02",
        Delta.of(crps="pure", mem="1", noise="0"),
        "loss geometry with no noise on either side; MAE against REF-D's MSE",
        priority=9,
        allow_degenerate=True,
    ),
    Experiment(
        "M03",
        Delta.of(crps="pure", mem="1"),
        "noise conditioning under MAE; also M1 of the pure-CRPS member sweep",
        priority=9,
        allow_degenerate=True,
    ),
    Experiment(
        "M04",
        Delta.of(crps="pure"),
        "pure CRPS at two members; what the 0.1 energy-score term buys, and M2 "
        "of the member sweep",
        priority=9,
    ),
    Experiment(
        "M05",
        Delta.of(crps="pure", mem="3"),
        "three members on a pure-CRPS objective; the runnable replacement for "
        "aug26 E26",
        priority=10,
    ),
    # -- Tier 2: the objective internals, all at two members ------------------
    # L01 -- Delta.of(crps="energy"), the spectral-space objective alone -- is
    # BLOCKED, not dropped.  It is the sharpest test in the campaign for a model
    # sold on spatial structure, and it fails on the first training batch for a
    # reason that has nothing to do with the science: see the mode_weights ndim
    # bug in validate().  Restore it the moment that lands upstream.
    Experiment(
        "L02",
        Delta.of(fdcrps="1"),
        "spatially pooled CRPS as a training objective (Alet et al. 2025), one "
        "coarsening level",
        priority=9,
    ),
    Experiment(
        "L03",
        Delta.of(alpha="095"),
        "almost-fair CRPS; at two members the pairwise term is a single pair, "
        "which is where this is supposed to pay",
        priority=11,
    ),
    Experiment(
        "N01",
        Delta.of(ntype="gauss"),
        "the noise's spatial correlation: i.i.d. per grid point against "
        "isotropic drawn in spherical-harmonic space",
        priority=9,
    ),
    # -- Tier 3: separating two steps from two scored steps -------------------
    Experiment(
        "R01",
        Delta.of(roll="c2"),
        "two forward steps, one scored; with REF-S (1 step) and aug26 E18 "
        "(2 steps, both scored) the two moves separate",
        priority=11,
    ),
]


# ------------------------------------------------------------------- guards --


def validate(run: Run) -> list[str]:
    """Complaints about a delta, empty if it is sound.

    Everything here is a property of the *code*, verified against this branch on
    2026-09-03, not a matter of taste.  A delta that trips one of these produces
    a config whose run id claims something the run does not do.
    """
    bad: list[str] = []
    d = run.delta
    obj, crps, mem = d.get("obj"), d.get("crps"), d.get("mem")
    noise, fd, alpha = d.get("noise"), d.get("fdcrps"), d.get("alpha")

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(f"{run.runid}: {msg}")

    # THE BLOCKER.  get_energy_score (fme/core/ensemble.py:80) opens with
    # `if gen.shape[1] != 2: raise NotImplementedError`, EnergyScoreLoss.forward
    # calls it unconditionally, and EnsembleLoss.forward calls EnergyScoreLoss
    # whenever energy_score_weight > 0.  So a one- or three-member arm with any
    # energy-score weight raises on the FIRST BATCH -- after config validation,
    # after dataset construction, after the model is built.  Reproduced on a GPU
    # node 2026-09-03; this is what aug26's E25 and E26 do.
    energy_w = CRPS_WEIGHTS[crps][1] if obj == "crps" else 0.0
    want(
        not (obj == "crps" and energy_w > 0 and mem != "2"),
        f"mem-{mem} with an energy-score weight of {energy_w}: get_energy_score "
        f"supports exactly two members and raises on the first training batch "
        f"otherwise. Use crps-pure, which gates the energy score off entirely, "
        f"or land the upstream generalization first",
    )

    # THE SECOND BLOCKER, found the same way as the first -- by running it.
    # EnergyScoreLoss.forward builds `mode_weights` with `x_hat.ndim - 1`
    # leading singleton dimensions, but get_energy_score has already consumed
    # the ensemble dimension, so the energy component comes out with TWO
    # spurious leading dims: (1, 1, B, C, n_l, n_m) instead of (B, C, n_l, n_m).
    # With a CRPS component present the correctly-shaped one carries the channel
    # breakdown and nothing fails.  At crps_weight 0 the energy score is the
    # only component, and single_module.py:1757 raises
    #   RuntimeError: Per-channel loss has 1 elements but 50 channel names
    # on the first training batch.  Reproduced on a GPU node 2026-09-03.
    want(
        not (obj == "crps" and CRPS_WEIGHTS[crps][0] == 0.0),
        f"crps-{crps} sets crps_weight to 0, leaving the energy score as the "
        f"only loss component. Its shape carries two spurious leading "
        f"dimensions, so get_channel_losses raises on the first training "
        f"batch. Blocked until the mode_weights ndim bug is fixed upstream",
    )

    # LossConfig.build drops every EnsembleLoss kwarg for MSE, so these would
    # parse, run, and read as a lie in the file.
    for axis, level in (("crps", crps), ("fdcrps", fd), ("alpha", alpha)):
        want(
            not (obj == "mse" and level != BASELINE[axis]),
            f"obj-mse with {axis}-{level}: LossConfig.build discards every "
            f"EnsembleLoss kwarg for MSE, so the id would claim a setting the "
            f"run does not have",
        )

    # almost_fair_crps_alpha only parameterizes the CRPS module, and
    # EnsembleLoss.forward gates that on crps_weight > 0.  At crps-energy the
    # alpha is inert.
    want(
        not (crps == "energy" and alpha != BASELINE["alpha"]),
        f"crps-energy with alpha-{alpha}: almost_fair_crps_alpha is inert when "
        f"crps_weight is 0, so the id would claim a setting with no effect",
    )
    # ...but finite_difference_crps_weight is NOT inert: forward gates it on
    # `self.diff_crps_loss is not None`, i.e. on its own weight alone.  So
    # crps-energy_fdcrps-1 would quietly run a pooled-CRPS-plus-energy objective
    # while the id says the CRPS family is off.  The asymmetry is easy to get
    # backwards, which is why both directions are spelled out.
    want(
        not (crps == "energy" and fd != BASELINE["fdcrps"]),
        f"crps-energy with fdcrps-{fd}: the finite-difference CRPS term is "
        f"gated on its own weight, not on crps_weight, so it would still run "
        f"and the id would understate the objective",
    )

    # Two wasteful-arm guards.  Both are right about the waste, and a mechanism
    # probe deliberately spends it -- so they are gated on an explicit opt-in
    # that reaches the artifacts, not deleted.
    degenerate = (
        obj == "crps" and noise == "0",
        obj == "mse" and mem == "1" and noise != "0",
    )
    if degenerate[0] and not run.exp.allow_degenerate:
        bad.append(
            f"{run.runid}: noise-0 with an ensemble loss scores a degenerate "
            f"ensemble at full ensemble cost. Set allow_degenerate=True if that "
            f"is the point of the arm"
        )
    if degenerate[1] and not run.exp.allow_degenerate:
        bad.append(
            f"{run.runid}: noise-{noise} with obj-mse at one member wires up "
            f"noise conditioning that nothing in the objective can reward. Set "
            f"allow_degenerate=True if that is the point of the arm"
        )
    return bad


# -------------------------------------------------------------------- build --


def apply_delta(config: dict, run: Run) -> None:
    """Map the delta onto the yaml.  The only place that knows the schema."""
    d = run.delta
    training = config["stepper_training"]
    builder = config["stepper"]["step"]["config"]["builder"]["config"]

    training["n_ensemble"] = MEMBERS[d.get("mem")]

    steps, last_only = ROLL[d.get("roll")]
    training["n_forward_steps"] = steps
    training["optimize_last_step_only"] = last_only

    noise = NOISE[d.get("noise")]
    builder["noise_embed_dim"] = noise
    # noise-0 forces the type regardless of the ntype level: isotropic at zero
    # channels dies in the MKL FFT.  A delta that asks for both is not an error,
    # it is over-specified, and the checker asserts the resolved value.
    ntype = NOISE_TYPE_AT_ZERO if noise == 0 else d.get("ntype")
    builder["noise_type"] = NTYPE[ntype]

    loss = training["loss"]
    loss["type"] = LOSS_TYPE[d.get("obj")]
    if d.get("obj") == "mse":
        # LossConfig.build ignores kwargs for MSE; leaving EnsembleLoss weights
        # in the file would be dead text that reads as if it applied.
        loss.pop("kwargs", None)
        return

    crps_w, energy_w = CRPS_WEIGHTS[d.get("crps")]
    kwargs = {"crps_weight": crps_w, "energy_score_weight": energy_w}
    if d.get("fdcrps") != BASELINE["fdcrps"]:
        kwargs["finite_difference_crps_weight"] = FDCRPS_WEIGHT
        kwargs["finite_difference_crps_levels"] = FDCRPS[d.get("fdcrps")]
    if d.get("alpha") != BASELINE["alpha"]:
        kwargs["almost_fair_crps_alpha"] = ALPHA[d.get("alpha")]
    loss["kwargs"] = kwargs


def apply_epoch_schedule(config: dict, epochs: int) -> None:
    """Give every inference block the same schedule, ending on the final epoch.

    FME fires a block on ``list(range(1, max_epochs + 1))[start::step]``.  The
    range starts at 1 because evaluate_before_training is off, and solving
    `start` against a range one element longer is how aug26 once arranged for
    the final epoch never to be scored.  The arithmetic has to be re-solved per
    run, not per campaign, because `ep` is an axis.
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

    The evaluator does this check too late to be useful -- InlineInferenceConfig
    raises readably but InferenceEvaluatorConfig has no such check at all, so a
    mismatch there surfaces as a bare AssertionError inside the data loader,
    minutes into an allocation.  Do the arithmetic here, on a login node.
    """
    config["train_loader"]["batch_size"] = run.batch
    config["validation"]["loader"]["batch_size"] = run.batch
    if run.batch % run.ranks:
        raise ConfigError(f"{run.runid}: batch {run.batch} does not divide ranks")
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
    apply_delta(config, run)
    apply_sizing(config, run)
    apply_epoch_schedule(config, run.exp.epochs)
    for block in config.get("inference", []):
        block["n_forward_steps"] = INFERENCE_YEARS * STEPS_PER_YEAR
    if run.exp.warm_start_from:
        # A run id in the .env, never a path: runs/ must be byte-identical for
        # every teammate, and the parent's checkpoint lives under whichever
        # $CAMPAIGN_ROOT owns it.  run-train.sh resolves it at submit time and
        # refuses if the checkpoint is absent.
        config["stepper_training"].setdefault("parameter_init", {})["weights_path"] = (
            WARM_START_PLACEHOLDER
        )
    return config


def env_file(run: Run) -> str:
    """W&B provenance, read from the environment by wandb rather than the yaml.

    Deliberately free of identity: who submitted a run and where its output
    landed are properties of the submission, not the run list.  Baking either in
    made aug26's runs/ differ for every teammate, which dirtied the worktree the
    moment anyone regenerated -- and the submit script refuses a dirty worktree,
    so only the generator's author could launch.
    """
    tags = [CAMPAIGN, REALM, run.exp.label, f"s{run.seed:02d}", f"P{run.exp.priority}"]
    # Every varied axis is its own tag, so "every crps-pure run" is a filter
    # rather than a regex -- and a run acquires no tag for an axis it holds at
    # the template value.
    tags += [f"{k}-{v}" for k, v in run.delta.levels]
    if run.exp.allow_degenerate:
        tags.append("degenerate-by-design")
    warm = (
        [
            f"FME_WARM_START_FROM={run.exp.warm_start_from}",
            f"FME_WARM_START_CKPT={WARM_START_CKPT}",
        ]
        if run.exp.warm_start_from
        else []
    )
    note = run.exp.note
    if run.exp.allow_degenerate:
        note += " [degenerate by design: this arm deliberately spends the waste "
        note += "the generator's guards exist to prevent]"
    return "\n".join(
        [
            f"# generated by make_campaign.py -- {run.exp.label}",
            f"FME_NODES={run.nodes}",
            f"FME_RANKS={run.ranks}",
            f"FME_PRIORITY={run.exp.priority}",
            *warm,
            f"WANDB_NAME={run.runid}",
            f"WANDB_RUN_GROUP={CAMPAIGN}.{REALM}.{run.exp.label}",
            # The job type is what a W&B workspace groups arms by, so it carries
            # the delta word: without it every arm would share one job type and
            # the grouping would show one arm where there are ten.
            f"WANDB_JOB_TYPE={run.delta.word()}",
            f"WANDB_TAGS={','.join(tags)}",
            f'WANDB_NOTES="{note} | {run.nodes} nodes, {run.ranks} ranks"',
            "",
        ]
    )


def expand(runlist: list[Experiment]) -> list[Run]:
    runs = [Run(e, s) for e in runlist for s in e.seeds]
    seen: dict[str, str] = {}
    for r in runs:
        if r.runid in seen:
            raise ConfigError(
                f"two experiments render the same run id {r.runid}: "
                f"{seen[r.runid]} and {r.exp.label}"
            )
        seen[r.runid] = r.exp.label
    return runs


def manifest(runs: list[Run]) -> str:
    head = "runid\tlabel\tdelta\tseed\tpriority\tepochs\tnodes\trel\trun_hours\tnote"
    rows = [
        "\t".join(
            [
                r.runid,
                r.exp.label,
                r.delta.word(),
                f"{r.seed:02d}",
                str(r.exp.priority),
                str(r.exp.epochs),
                str(r.nodes),
                f"{r.rel:.2f}",
                f"{r.run_hours:.0f}",
                r.exp.note.replace("\t", " "),
            ]
        )
        for r in runs
    ]
    return "\n".join([head, *rows]) + "\n"


def report(runs: list[Run]) -> None:
    print(f"{CAMPAIGN} {REALM} -- {len(runs)} runs\n")
    w = max(len(r.runid) for r in runs)
    print(
        f"{'label':<5} {'run id':<{w}} {'pri':>3} {'ep':>3} {'nodes':>5} "
        f"{'rel':>5} {'run h':>6} {'node h':>7}"
    )
    for r in sorted(runs, key=lambda r: (r.exp.priority, r.exp.label, r.seed)):
        print(
            f"{r.exp.label:<5} {r.runid:<{w}} {r.exp.priority:>3} "
            f"{r.exp.epochs:>3} {r.nodes:>5} {r.rel:>5.2f} "
            f"{r.run_hours:>6.0f} {r.nodes * r.run_hours:>7.0f}"
        )
    nodes = sum(r.nodes for r in runs)
    node_h = sum(r.nodes * r.run_hours for r in runs)
    crit = max(r.run_hours for r in runs)
    print(
        f"\n{len(runs)} runs, {nodes} nodes concurrent, {node_h:,.0f} node-hours, "
        f"critical path {crit:.0f} h"
    )
    print(
        "references are aug26 E01 (stochastic pole) and E21 (deterministic "
        "pole), 3 seeds each, not re-run here"
    )
    print(
        "member and rollout costs are measured (analysis/card-sweep.sh, "
        "2026-09-03); fdcrps and ntype are still assumed at 1.0"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="write every run")
    p.add_argument("--label", action="append", default=[], help="one experiment")
    p.add_argument("--list", action="store_true", help="print the run list only")
    p.add_argument("-o", "--out", default=None, help="output directory")
    p.add_argument("--epochs", type=int, default=None, help="override max_epochs")
    args = p.parse_args(argv)

    runlist = RUNLIST
    if args.label:
        runlist = [e for e in RUNLIST if e.label in args.label]
        if not runlist:
            print(f"no experiment matches {args.label}", file=sys.stderr)
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
    out = pathlib.Path(args.out) if args.out else None
    if out is None:
        for r in runs:
            yaml.safe_dump(build(baseline, r), sys.stdout, sort_keys=False, width=100)
        return 0

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
