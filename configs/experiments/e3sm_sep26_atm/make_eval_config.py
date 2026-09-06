#!/usr/bin/env python3
"""Generate offline evaluation configs for a sep26 run.

Training writes inline rollouts, but every inline block runs ONE member per
initial condition: it can score a single realisation's RMSE, spectra and
stability and it cannot score calibration, spread, or any proper
finite-ensemble quantity.  That is the campaign's own main question, so the
offline pass is a launch gate rather than a nice-to-have (`TODO.md` D1).

Two passes, because they want opposite things from the same weights:

    scores   many members per initial condition, no trajectory files.
             Produces CRPS, spread-skill ratio and ensemble-mean RMSE at a
             ladder of leads, plus the usual time-mean / spectrum / histogram
             diagnostics.  This is the pass that answers "is the learned
             stochasticity any good".
    traj     one member per initial condition, prediction files written.
             Produces the per-trajectory statistics -- temporal variance,
             persistence, quantiles, wet-day frequency and intensity -- which
             MUST be computed within a trajectory and only then averaged
             across realisations.  Averaging members first reports the model
             as smoother than it is: measured at 8-41% too little variance on
             RF01 (`analysis/noise_decomp/results/ens4_mean_vs_member.txt`).

Both passes read the dataset block and the held-out initial conditions out of
`config-train-atm.template.yaml`, so an eval config cannot drift from the
training data it is meant to score.

Noise modes (stochastic arms only; `Z0` arms have no noise pathway and the
stepper refuses the override).  Measured on RF01 in
`analysis/noise_decomp/REVIEW.md`:

    keep     the trained behaviour -- fresh noise every step
    off      scale 0: the learned backbone g(x, 0).  NOT a deterministic
             control.  It is a worse one-step operator than the model's own
             noise-averaged mean by 14-16% on all three RF01 seeds, and its
             one-year climate is 4-8x further off in the time mean.  The
             trained deterministic control is RF02.
    mean     the iterated conditional mean E_Z g(x, Z): the honest
             deterministic proxy of a stochastic checkpoint, and the right
             thing to put next to RF02 alongside individual members.
    fixed    one latent field held for the whole rollout.  A probe of
             temporal whiteness, not a model -- nothing in training ever saw
             it.  Adds +0.2 to +0.4 dex of spurious small-scale power.
    half     amplitude 0.5, the downward half of the calibration bracket.

Usage:

    ./make_eval_config.py RF02.sep26.atm.D1_G0_I0_M1_N0_Q0_R0_Y0_Z0.S01
    ./make_eval_config.py RF01.S01 --noise mean --pass traj
    ./make_eval_config.py --all --pass scores --dry-run
"""

import argparse
import copy
import datetime
import glob
import os
import pathlib
import subprocess
import sys

import make_campaign as mc
import yaml

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = HERE / "config-train-atm.template.yaml"

# RF01 is not generated: it is aug26's E01, three seeds already trained.
# `check_campaign.py` asserts the template still matches that config.
RF01_RUNID = "E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0"
RF01_ROOT = "{pscratch}/aug26"
RF01_SEEDS = ("S01", "S02", "S03")

# The default is 2 rather than 4 nodes because the useful unit here is one
# whole arm per node-pair: 8 initial conditions over 8 ranks is one IC per
# rank, which is what the 16-IC stall (below) leaves as the largest shape that
# reliably completes.
DEFAULT_NODES = 2

# Rollout length, in years, defaulted per pass rather than shared.  The scores
# pass is read at fixed leads (SCORE_STEPS below) and rolling out past the last
# of them buys no ensemble metric -- only a better-sampled climatology, which
# is the trajectory pass's job and costs five times as much here.  The
# trajectory pass wants the length, because drift is what it measures.
# `--years` overrides either.
SCORES_YEARS = 1
TRAJ_YEARS = 5

# 16 ICs is the template's whole held-out block, and it is the default again.
#
# It was not, for a while.  Three separate 16-IC attempts parked ranks in
# uninterruptible D state inside the DVS client (`dvsipc_wait_for_resp`,
# `ipclower_tx_request`) and never returned, while 8- and 4-IC runs reading the
# same files finished (`analysis/noise_decomp/REVIEW.md` 3.2).  Then on
# 2026-09-04 a 4-IC run stalled the same way under concurrent load, which said
# the initial-condition count was a correlate rather than the cause.
#
# RESOLVED 2026-09-05: the cause is DVS, and staging the dataset to Lustre
# removes it.  A 16-IC run on four nodes against a staged copy completed in
# 17.5 minutes with every GPU at 93-100% and nothing in D state.  The count was
# never the variable; it was how much traffic the shape put through DVS.
#
# Why the full block rather than half of it, now that both work: at 8 ICs the
# skill metrics are already stable to 0.2-3%, but the calibration statistics
# are not -- going 8 -> 16 moves `ssr_bias` at one year by 0.08 on both Tat2m
# and PS.  Calibration is a second-moment statement about a small sample, and
# it wants the samples.  A stalled job still holds its allocation silently, so
# `run-eval.sh` keeps its deadline.
DEFAULT_ICS = 16

# Arms are scored at a FIXED EPOCH with averaged weights, not at whatever
# `best_ckpt.tar` happens to hold.
#
# Two measurements force this.  `best_ckpt.tar` is rewritten whenever
# validation loss improves, so its epoch differs from arm to arm; scoring three
# seeds of one arm that way gives a 33% spread in 90-day temperature error
# against 14.5% at a common epoch, so half the campaign's resolving power goes
# to an accident of when each run last improved.  And the epoch is not a
# neutral choice: swept from 10 to 23 on one seed, one-day error improves
# 10-27% while one-year error degrades 6-96%, on every variable, pivoting at
# 30 days -- so validation loss, which tracks the improving end, selects close
# to the worst checkpoint available for the climate range.
#
# `--epoch N` therefore folds the running average of `ckpt_NNNN.tar` into the
# weights the evaluator loads, caching the result, and scores that.  Leaving it
# unset keeps `best_ckpt.tar`, which is right for a one-off look at an arm and
# wrong for any comparison between arms.
SCORING_EPOCH_ENV = "SEP26_SCORING_EPOCH"

# The epoch itself, chosen 2026-09-06 from a sweep on two seeds.
#
# One-year error has a broad minimum and one-day error falls monotonically, so
# the two ranges want different checkpoints and the choice is a trade.  What
# picks 10 rather than 14 is that the seeds disagree about 14: S01 is 5.7% off
# its one-year minimum there and S02 is 31% off, while at 10 both sit at their
# minimum (+0.1% and +0.0%) and within 0.6% of their best 90-day.  One seed's
# basin extends further than the other's, and 10 is inside both.
#
# It costs 17-22% of one-day skill against the fully-trained checkpoint, which
# is the right side of a trade whose other end is 92-135% of one-year skill;
# 30-day is flat across the whole range either way.
#
# CAVEAT worth carrying: at epoch 10 the model is short of its final one-day
# skill, so a *weather-range* comparison here partly measures which objective
# converges faster.  Every arm is equally early, which contains that but does
# not remove it -- score a second time at the converged end for weather claims,
# and say which epoch a number came from.
SCORING_EPOCH = 10

# Where the evaluator reads its data.  The training template points at the
# project tree on CFS, which the compute nodes see through DVS -- and the
# evaluator's read pattern (a window of every variable, every 20 steps, per
# initial condition) is the one DVS is worst at.  MEASURED on 2026-09-05, the
# same two concurrent 8-IC evaluations either way: 84 s per window against CFS,
# with ranks parked in `dvsipc_wait_for_response` while the GPUs that had data
# sat at 100%, and 13.5 s per window against a staged copy on Lustre.  A single
# run against CFS managed ~25 s, so the filesystem was the bottleneck and
# concurrency made it worse.  Copying the decade costs 77 s at 3.3 GB/s
# (sbatch-scripts/stage-data.sh), repaying itself inside the first run.  Set
# EVAL_DATA_ROOT, or pass --data-root, to use one.
STAGED_DATA_ROOT_ENV = "EVAL_DATA_ROOT"

# The amplitude rungs bracket the trained value rather than landing on it.
# MEASURED on RF01.S01 at one day: the trained amplitude is 7-9%
# under-dispersed by both spread-skill and rank, so the calibrated scale is
# near 1.1 if spread grows linearly with it. `half` overshoots downward hard
# (ssr_bias -0.09 -> -0.37, CRPS +31%), which is the point: it says the
# ensemble is not too wide. `double` is the other bracket, and without it the
# ladder could only ever conclude "not smaller".
NOISE_MODES = {
    "keep": None,
    "off": {"scale": 0.0, "mode": "fresh"},
    "fresh": {"scale": 1.0, "mode": "fresh"},
    "half": {"scale": 0.5, "mode": "fresh"},
    "double": {"scale": 2.0, "mode": "fresh"},
    "fixed": {"scale": 1.0, "mode": "fixed"},
    "mean": {"scale": 1.0, "mode": "mean", "draws": 8},
}

# Leads at which the ensemble scores are taken, in 6-hourly steps: 6 h, 1 d,
# 5 d, 30 d, 90 d, 1 y.  The first three are the weather range where the
# spread should track the error; the last three are where a stochastic model
# either holds its climate or does not.
SCORE_STEPS = (1, 4, 20, 120, 360, 1460)

PLOT_VARS = ["Tat2m", "surface_precipitation_rate", "PS", "FLUT", "U_6", "T_7"]
HIST_VARS = [
    "PS", "TS", "LHFLX", "SHFLX", "surface_precipitation_rate", "FLUT",
    "FSNS", "TAUX", "Qat2m", "Uat10m", "Tat2m", "T_1", "T_4", "T_7",
    "STW_4", "STW_7", "U_1", "U_4", "U_6", "V_6",
]  # fmt: skip
# The trajectory pass writes three fields, not fifty: at 6-hourly output over
# five years a single 180x360 float32 field is 5.7 GB per initial condition.
TRAJ_VARS = ["Tat2m", "surface_precipitation_rate", "PS"]


class EvalError(Exception):
    pass


def narrow_file_pattern(pattern: str, ics: list[str], years: int) -> str:
    """Restrict the file glob to the years the rollout can reach.

    The template's pattern matches all 1,501 monthly history files, 1940
    through 2065, because training reads all of them. An evaluation reads
    only from its initial conditions forward, and opening the full set costs
    upward of 13 minutes of wall clock -- paid once per eval job, of which
    this campaign has at least 26. Measured here at 20+ minutes with three
    jobs sharing the filesystem, every rank in uninterruptible I/O wait.

    2040-2052 is 240 files rather than 1,501. The glob keeps a character
    class on the decade digit, which is the widest narrowing that stays a
    single fsspec-compatible pattern; if the span crosses a century or the
    pattern is not the expected shape, it is returned unchanged rather than
    guessed at.
    """
    marker = "h0.*"
    if marker not in pattern:
        return pattern
    try:
        years_used = [int(ic[:4]) for ic in ics]
    except (ValueError, IndexError):
        return pattern
    first = min(years_used)
    # +1: a rollout starting in July of the last year runs into the next one.
    last = max(years_used) + years + 1
    if first // 100 != last // 100:
        return pattern
    century, lo, hi = first // 100, (first % 100) // 10, (last % 100) // 10
    if lo == hi:
        return pattern.replace(marker, f"h0.{century}{lo}*")
    return pattern.replace(marker, f"h0.{century}[{lo}-{hi}]*")


def _template() -> dict:
    with open(TEMPLATE) as f:
        return yaml.safe_load(f)


def _test_block(template: dict) -> dict:
    """The held-out inference block, by name rather than by position."""
    blocks = template["inference"]
    for block in blocks:
        if block.get("name") == "5yr_test":
            return block
    raise EvalError(
        "config-train-atm.template.yaml has no inference block named "
        "'5yr_test'; the eval generator reads its dataset and initial "
        "conditions from there so the two cannot drift apart"
    )


def resolve_run(runid: str) -> tuple[mc.Run | None, str, str]:
    """Return (run, runid, checkpoint_dir) for a campaign or inherited run."""
    pscratch = os.environ.get("PSCRATCH", "/pscratch/sd/m/mahf708")
    root = os.environ.get("CAMPAIGN_ROOT", f"{pscratch}/{mc.CAMPAIGN}")
    if runid.startswith("RF01"):
        # RF01.S01 -- inherited from aug26, under its own campaign root.
        seed = runid.split(".")[-1]
        if not (seed.startswith("S") and seed[1:].isdigit()):
            raise EvalError(f"RF01 must be named RF01.S01 .. RF01.S03, got {runid!r}")
        full = f"{RF01_RUNID}.{seed}"
        return None, full, os.path.join(RF01_ROOT.format(pscratch=pscratch), full)
    for run in mc.expand(mc.RUNLIST):
        if run.runid == runid or f"{run.experiment.exp}.S{run.seed:02d}" == runid:
            return run, run.runid, os.path.join(root, run.runid)
    known = sorted(e.exp for e in mc.RUNLIST)
    raise EvalError(f"unknown run id {runid!r}; experiments are {known} (or RF01.S01)")


def fixed_epoch_checkpoint(checkpoint_dir: str, runid: str, epoch: int) -> str:
    """Averaged weights at `epoch`, built once and cached.

    `ckpt_NNNN.tar` keeps raw weights in the slot the evaluator reads and the
    running average beside them, while `best_ckpt.tar` has folded the average
    in.  Scoring the raw ones is not a smaller version of the same comparison:
    on RF01.S01 at epoch 22 the two differ by more than the entire epoch effect
    (1.750 K averaged against 4.130 K raw, 90-day ensemble-mean RMSE).  So the
    fold is not optional, and `analysis/ema_checkpoint.py` is what does it.
    """
    pscratch = os.environ.get("PSCRATCH", "/pscratch/sd/m/mahf708")
    pin_root = os.environ.get("SEP26_PIN_ROOT", f"{pscratch}/sep26-pin")
    pinned = os.path.join(pin_root, f"{runid}.ema{epoch}")
    target = os.path.join(pinned, "training_checkpoints", "best_ckpt.tar")
    if os.path.exists(target):
        return pinned
    source = os.path.join(
        checkpoint_dir, "training_checkpoints", f"ckpt_{epoch:04d}.tar"
    )
    if not os.path.exists(source):
        raise EvalError(
            f"no checkpoint for epoch {epoch} at {source}. Available epochs "
            "are the ckpt_NNNN.tar files in that directory; an arm that has "
            "not reached the scoring epoch cannot be scored at it."
        )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fold = os.path.join(os.path.dirname(__file__), "analysis", "ema_checkpoint.py")
    result = subprocess.run(
        [sys.executable, fold, source, target], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise EvalError(f"could not fold {source}: {result.stderr.strip()}")
    return pinned


def check_divisible(n_ics: int, nodes: int) -> int:
    """Return the rank count, refusing a shape the loader would assert on.

    `InferenceDataset.__getitem__` asserts that every rank received
    `n_initial_conditions // total_data_parallel_ranks` samples.  That assert
    now has a config-time guard upstream (`InferenceEvaluatorConfig`), but the
    arithmetic belongs here too: this is where the node count is chosen, and
    an allocation is the expensive place to discover it.
    """
    ranks = nodes * mc.GPUS_PER_NODE
    if n_ics % ranks:
        options = [n for n in (1, 2, 4, 8) if n_ics % (n * mc.GPUS_PER_NODE) == 0]
        raise EvalError(
            f"{n_ics} initial conditions over {ranks} ranks ({nodes} nodes x "
            f"{mc.GPUS_PER_NODE} GPUs) leaves a remainder; the loader deals "
            f"them out evenly or asserts. Node counts that divide {n_ics}: "
            f"{options or 'none -- change --ics'}"
        )
    return ranks


def noise_override(name: str, word: mc.Word | None) -> dict | None:
    override = NOISE_MODES[name]
    if override is None:
        return None
    if word is not None and mc.NOISE_DIM[word.get("Z")] == 0:
        raise EvalError(
            f"--noise {name} on a Z{word.get('Z')} arm: the checkpoint has no "
            "noise pathway, so there is nothing to override. Only stochastic "
            "arms take a noise mode; the deterministic ones are already the "
            "control."
        )
    return dict(override)


def build(
    *,
    runid: str,
    checkpoint: str,
    word: mc.Word | None,
    which_pass: str,
    noise: str,
    members: int,
    n_ics: int,
    nodes: int,
    years: int,
    seed: int,
    out_dir: str,
    wandb: bool = True,
    data_root: str | None = None,
) -> dict:
    template = _template()
    block = _test_block(template)
    ics = list(block["loader"]["start_indices"]["times"])[:n_ics]
    if len(ics) != n_ics:
        raise EvalError(
            f"the template's held-out block has {len(ics)} initial conditions, "
            f"fewer than the {n_ics} requested"
        )
    check_divisible(n_ics, nodes)
    steps = years * mc.STEPS_PER_YEAR
    is_scores = which_pass == "scores"
    dataset = copy.deepcopy(block["loader"]["dataset"])
    if isinstance(dataset, dict) and "file_pattern" in dataset:
        dataset["file_pattern"] = narrow_file_pattern(
            dataset["file_pattern"], ics, years
        )
    if data_root is not None:
        if not isinstance(dataset, dict):
            raise EvalError("the template's dataset is not a single path to move")
        # A staged copy that is missing the years this rollout reaches gives an
        # empty or short dataset rather than an error, so check it here.
        staged = glob.glob(os.path.join(data_root, dataset["file_pattern"]))
        original = glob.glob(
            os.path.join(dataset["data_path"], dataset["file_pattern"])
        )
        if len(staged) < len(original):
            raise EvalError(
                f"{data_root} holds {len(staged)} of the {len(original)} files "
                f"matching {dataset['file_pattern']}; stage the rest with "
                "sbatch-scripts/stage-data.sh before using --data-root"
            )
        dataset["data_path"] = data_root

    aggregator: dict = {
        "histogram": {"enabled": True, "variables": HIST_VARS},
        "zonal_mean": {"enabled": False},
        "video": {"enabled": False},
        "trend": {"enabled": False},
        "seasonal": {"enabled": True},
        "near_zero_fraction": {"enabled": False},
        "enso_coefficient": {"enabled": False},
        "ipo_index": {"enabled": False},
        # The clamp is applied after the network at every step, so a model
        # whose tails come from clamping has not learned them. Track the
        # burden so noise-on and noise-off rollouts can be compared on it.
        "step_diagnostics": {"correction_scalars": True, "correction_maps": False},
        "time_mean_denorm": {"plot_variables": PLOT_VARS},
        "time_mean_norm": {"target": "norm", "plot_variables": PLOT_VARS},
        "power_spectrum": {"plot_variables": PLOT_VARS},
        "step_means": [{"step": s} for s in SCORE_STEPS if s <= steps],
    }
    if is_scores and members > 1:
        aggregator["ensembles"] = [
            {"step": s, "strict": False, "target": t}
            for s in SCORE_STEPS
            if s <= steps
            for t in ("denorm", "norm")
        ]

    config = {
        "experiment_dir": out_dir,
        "n_forward_steps": steps,
        "forward_steps_in_memory": 20,
        "checkpoint_path": os.path.join(
            checkpoint, "training_checkpoints/best_ckpt.tar"
        ),
        "logging": {
            "log_to_screen": True,
            "log_to_file": True,
            "log_to_wandb": wandb,
            "project": mc.WANDB_PROJECT,
            "entity": mc.WANDB_ENTITY,
        },
        "loader": {
            "start_indices": {"times": ics},
            "dataset": dataset,
            # In-process reads. Forked loader workers stalled in an
            # uninterruptible DVS wait at the same window on three nodes.
            "num_data_workers": 0,
        },
        "n_ensemble_per_ic": members,
        "seed": seed,
        "data_writer": {
            "save_prediction_files": not is_scores,
            "save_monthly_files": False,
            "names": TRAJ_VARS,
        },
        "aggregator": aggregator,
    }
    override = noise_override(noise, word)
    if override is not None:
        config["stepper_override"] = {"noise": override}
    return config


def eval_id(label: str, which_pass: str, noise: str) -> str:
    """`<exp>.S<seed>.eval-<pass>[-<noise>]`, in campaign names.

    RF01's weights are aug26's E01, but the campaign calls it RF01 and so
    does every table that will read these directories.
    """
    suffix = "" if noise == "keep" else f"-{noise}"
    return f"{label}.eval-{which_pass}{suffix}"


def checkpoint_provenance(path: str) -> list[str]:
    """Identify the exact weights an evaluation read.

    An arm still training rewrites ``best_ckpt.tar`` whenever validation
    improves, so two evaluations of "the same" arm hours apart can be two
    different models -- which turns a seed spread or a noise ladder into a
    comparison of epochs. Recording size and mtime makes that checkable after
    the fact instead of assumed; use ``--checkpoint`` to point at a pinned copy
    when it matters.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return ["# checkpoint not present at generation time"]
    stamp = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
    return [
        f"# checkpoint {path}",
        f"# checkpoint_bytes {stat.st_size}",
        f"# checkpoint_mtime {stamp}",
    ]


def env_file(
    evalid: str,
    runid: str,
    nodes: int,
    which_pass: str,
    noise: str,
    checkpoint_path: str = "",
) -> str:
    ranks = nodes * mc.GPUS_PER_NODE
    tags = ["sep26", "atm", "eval", which_pass, noise, evalid.split(".")[0]]
    return "\n".join(
        [
            f"# generated by make_eval_config.py -- {evalid}",
            *checkpoint_provenance(checkpoint_path),
            f"FME_NODES={nodes}",
            f"FME_RANKS={ranks}",
            f"WANDB_NAME={evalid}",
            f"WANDB_RUN_GROUP=sep26.atm.eval.{evalid.split('.')[0]}",
            f"WANDB_JOB_TYPE=eval-{which_pass}{'' if noise == 'keep' else '-' + noise}",
            f"WANDB_TAGS={','.join(tags)}",
            f'WANDB_NOTES="offline {which_pass} pass for {runid}, noise {noise}"',
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("runid", nargs="?", help="run id, experiment id, or RF01.S01")
    parser.add_argument("--all", action="store_true", help="every run in the campaign")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the campaign's arm labels, one per line, and stop",
    )
    parser.add_argument(
        "--pass", dest="which_pass", default="scores", choices=["scores", "traj"]
    )
    parser.add_argument("--noise", default="keep", choices=sorted(NOISE_MODES))
    parser.add_argument(
        "--members",
        type=int,
        default=None,
        help="members per IC (default 4 for scores, 1 for traj)",
    )
    parser.add_argument("--ics", type=int, default=DEFAULT_ICS)
    parser.add_argument("--nodes", type=int, default=DEFAULT_NODES)
    parser.add_argument(
        "--years",
        type=int,
        default=None,
        help=f"rollout length (default {SCORES_YEARS} for scores, "
        f"{TRAJ_YEARS} for traj)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="override the weights path")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="score at this epoch with averaged weights, building the folded "
        f"checkpoint if needed (default 10, or ${SCORING_EPOCH_ENV}). "
        "0 means best_ckpt.tar, whose epoch differs per arm and doubles the "
        "seed floor -- fine for a one-off look at an arm, wrong for any "
        "comparison between them",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="read the dataset from here instead of the template's path "
        f"(default ${STAGED_DATA_ROOT_ENV})",
    )
    parser.add_argument("--out", default=None, help="output root (default $EVAL_ROOT)")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="for a development run; the campaign wants WandB on",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)  # fmt: skip

    if args.list:
        # RF01 first, and by name: it is not in RUNLIST because it is
        # inherited from aug26 rather than generated, but it is the
        # stochastic pole and every other arm differences against it.
        for seed in RF01_SEEDS:
            print(f"RF01.{seed}")
        for listed in mc.expand(mc.RUNLIST):
            print(f"{listed.experiment.exp}.S{listed.seed:02d}")
        return 0
    if args.all == bool(args.runid):
        parser.error("give a run id or --all, not both")
    members = args.members
    if members is None:
        members = 4 if args.which_pass == "scores" else 1
    years = args.years
    if years is None:
        years = SCORES_YEARS if args.which_pass == "scores" else TRAJ_YEARS
    if args.which_pass == "traj" and members > 1:
        parser.error(
            "the trajectory pass writes one member per initial condition: "
            "per-trajectory statistics must be computed inside a trajectory, "
            "and an ensemble mean is 8-41% short of the target variance"
        )

    pscratch = os.environ.get("PSCRATCH", "/pscratch/sd/m/mahf708")
    out_root = args.out or os.environ.get("EVAL_ROOT", f"{pscratch}/sep26-eval")
    data_root = args.data_root or os.environ.get(STAGED_DATA_ROOT_ENV) or None
    epoch = args.epoch
    if epoch is None:
        epoch = int(os.environ.get(SCORING_EPOCH_ENV) or SCORING_EPOCH)

    if args.all:
        targets = [r.runid for r in mc.expand(mc.RUNLIST)]
    else:
        targets = [args.runid]

    for target in targets:
        run, runid, ckpt_dir = resolve_run(target)
        if epoch and args.checkpoint is None:
            ckpt_dir = fixed_epoch_checkpoint(ckpt_dir, runid, epoch)
        word = run.word if run is not None else None
        if run is None:
            label = f"RF01.{runid.split('.')[-1]}"
        else:
            label = f"{run.experiment.exp}.S{run.seed:02d}"
        evalid = eval_id(label, args.which_pass, args.noise)
        out_dir = os.path.join(out_root, evalid)
        try:
            if os.path.islink(out_dir):
                # Writing through a symlink puts this config in some other
                # run's directory and overwrites its record of which weights
                # produced the results already sitting there. That happened
                # once, to a finished run, and only the fact that nothing
                # rewrites the netCDFs saved the results.
                raise EvalError(
                    f"{out_dir} is a symlink; refusing to write a config "
                    "through it. Point --out at a real directory."
                )
            config = build(
                runid=runid,
                checkpoint=args.checkpoint or ckpt_dir,
                word=word,
                which_pass=args.which_pass,
                noise=args.noise,
                members=members,
                n_ics=args.ics,
                nodes=args.nodes,
                years=years,
                seed=args.seed,
                out_dir=out_dir,
                wandb=not args.no_wandb,
                data_root=data_root,
            )
        except EvalError as err:
            if args.all and "noise pathway" in str(err):
                continue  # a deterministic arm simply has no noise ladder
            print(f"error: {err}", file=sys.stderr)
            return 2
        if args.dry_run:
            print(f"{evalid}\t{args.nodes} nodes\t{config['n_forward_steps']} steps")
            continue
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        with open(os.path.join(out_dir, "eval.env"), "w") as f:
            f.write(
                env_file(
                    evalid,
                    runid,
                    args.nodes,
                    args.which_pass,
                    args.noise,
                    config["checkpoint_path"],
                )
            )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
