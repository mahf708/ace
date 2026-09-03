#!/usr/bin/env python3
"""Assert that every generated run config agrees with its own run id.

    ./check_campaign.py               # checks runs/
    ./check_campaign.py --dir DIR

The tables below **duplicate** make_campaign.py's on purpose, and must keep
duplicating them.  A checker that imports the generator's mapping can only
prove the generator is self-consistent; one that re-derives the expected config
from the id independently catches a typo in that mapping, which is the failure
this file exists for.

What it does not do is prove a config *runs*.  aug26's checker passed E25 and
E26, which raise on their first training batch, because it verified that a
config agreed with its id and nothing more.  The member/energy-score rule below
is the specific lesson from that, but the general point stands: a real forward
and backward pass is the only thing that proves a config trains.
"""

import argparse
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "sep26"
REALM = "atm"
LOCAL_BATCH = 1
GPUS_PER_NODE = 4
INFERENCE_EVALUATIONS = 10
INFERENCE_YEARS = 5
STEPS_PER_YEAR = 1460

# --- duplicated from make_campaign.py, deliberately -------------------------
BASELINE = {
    "obj": "crps",
    "crps": "std",
    "mem": "2",
    "noise": "32",
    "ntype": "iso",
    "roll": "f1",
    "fdcrps": "0",
    "alpha": "100",
}
LOSS_TYPE = {"crps": "EnsembleLoss", "mse": "MSE"}
CRPS_WEIGHTS = {
    "std": (0.9, 0.1),
    "pure": (1.0, 0.0),
    "energy": (0.0, 1.0),
    "half": (0.5, 0.5),
}
MEMBERS = {"1": 1, "2": 2, "3": 3}
NOISE = {"0": 0, "32": 32, "64": 64}
NTYPE = {"iso": "isotropic", "gauss": "gaussian"}
ROLL = {
    "f1": (1, True),
    "c2": (2, True),
    "f2": (2, False),
    "s04": ({1: 0.6, 2: 0.2, 4: 0.2}, True),
    "s20": ({1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}, True),
}
FDCRPS = {"0": 0, "1": 1, "3": 3}
ALPHA = {"100": 1.0, "095": 0.95}
NOISE_TYPE_AT_ZERO = "gaussian"

WANDB = {"project": "SamudrACE-E3SMv3", "entity": "e3sm-aig"}
EMBED = {"embed_dim": 384}
# The training window.  A weighted inference block's WHOLE trajectory has to
# stay out of it, not just its first timestep -- a 7300-step rollout from 1980
# runs to 1992, into the 1990-95 validation split.
VALIDATION_WINDOW = ("1990-01-01", "1995-01-01")


def parse_runid(stem: str) -> tuple[dict[str, str], int]:
    """<campaign>.<realm>.<delta>.s<seed> -> resolved levels, seed.

    The delta is sparse: an axis absent from the id is a *claim* that the run
    sits at the template value for it, and that claim is checked below exactly
    as hard as an explicit level is.  Without that, omission would be a way to
    hide a setting.
    """
    fields = stem.split(".")
    if len(fields) != 4:
        raise ValueError(f"not <campaign>.<realm>.<delta>.s<seed>: {stem}")
    campaign, realm, word, seed = fields
    if campaign != CAMPAIGN or realm != REALM:
        raise ValueError(f"not a {CAMPAIGN}.{REALM} run: {stem}")
    if not seed.startswith("s") or not seed[1:].isdigit():
        raise ValueError(f"seed field is not s<digits>: {seed}")

    levels = dict(BASELINE)
    if word != "base":
        pairs: list[tuple[str, str]] = []
        for part in word.split("_"):
            axis, sep, level = part.partition("-")
            if not sep:
                raise ValueError(f"delta field {part!r} is not <axis>-<level>")
            if axis not in BASELINE:
                raise ValueError(f"unknown axis {axis!r} in {word}")
            if axis in [a for a, _ in pairs]:
                raise ValueError(f"axis {axis!r} appears twice in {word}")
            pairs.append((axis, level))
            levels[axis] = level
        # Canonical form: sorted, and no axis at its baseline value.  Two ids
        # for one configuration is how a W&B workspace ends up showing the same
        # arm under two names.
        if pairs != sorted(pairs):
            raise ValueError(f"delta {word} is not in canonical (sorted) order")
        for axis, level in pairs:
            if level == BASELINE[axis]:
                raise ValueError(
                    f"{axis}-{level} is the template value, so it must be "
                    f"omitted from the run id"
                )
    return levels, int(seed[1:])


def check(path: pathlib.Path) -> list[str]:
    """Complaints about one run config, empty if it is sound."""
    bad: list[str] = []
    try:
        levels, seed = parse_runid(path.stem)
    except ValueError as e:
        return [f"{path.name}: {e}"]

    d = yaml.safe_load(path.read_text())
    training = d["stepper_training"]
    step = d["stepper"]["step"]["config"]
    builder = step["builder"]["config"]

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(f"{path.stem}: {msg}")

    want(d["seed"] == seed, f"id says seed {seed}, config says {d['seed']}")

    # -- the delta, axis by axis --------------------------------------------
    want(
        training["n_ensemble"] == MEMBERS[levels["mem"]],
        f"mem-{levels['mem']} but n_ensemble is {training['n_ensemble']}",
    )
    steps, last_only = ROLL[levels["roll"]]
    want(
        training["n_forward_steps"] == steps,
        f"roll-{levels['roll']} but n_forward_steps is "
        f"{training['n_forward_steps']}",
    )
    want(
        training["optimize_last_step_only"] == last_only,
        f"roll-{levels['roll']} wants optimize_last_step_only={last_only}, got "
        f"{training['optimize_last_step_only']}",
    )
    noise = NOISE[levels["noise"]]
    want(
        builder["noise_embed_dim"] == noise,
        f"noise-{levels['noise']} but noise_embed_dim is "
        f"{builder['noise_embed_dim']}",
    )
    # Isotropic noise at zero channels calls an inverse SHT on a zero-channel
    # tensor and dies in the MKL FFT, so noise-0 must also carry the type.
    want_type = NOISE_TYPE_AT_ZERO if noise == 0 else NTYPE[levels["ntype"]]
    want(
        builder["noise_type"] == want_type,
        f"expected noise_type {want_type}, got {builder['noise_type']}"
        + (" (isotropic at zero channels dies in the MKL FFT)" if noise == 0 else ""),
    )

    # -- the loss ------------------------------------------------------------
    loss = training["loss"]
    want(
        loss["type"] == LOSS_TYPE[levels["obj"]],
        f"obj-{levels['obj']} but loss type is {loss['type']}",
    )
    if levels["obj"] == "mse":
        # LossConfig.build discards every EnsembleLoss kwarg for MSE, so kwargs
        # left in the file are dead text that reads as if it applied.
        want(
            not loss.get("kwargs"),
            f"obj-mse but loss.kwargs is {loss.get('kwargs')}; MSE discards "
            f"them, so the file would claim settings the run does not have",
        )
        for axis in ("crps", "fdcrps", "alpha"):
            want(
                levels[axis] == BASELINE[axis],
                f"obj-mse with {axis}-{levels[axis]}: unreachable under MSE",
            )
    else:
        kwargs = loss.get("kwargs") or {}
        crps_w, energy_w = CRPS_WEIGHTS[levels["crps"]]
        want(
            kwargs.get("crps_weight") == crps_w
            and kwargs.get("energy_score_weight") == energy_w,
            f"crps-{levels['crps']} wants ({crps_w}, {energy_w}), got "
            f"({kwargs.get('crps_weight')}, {kwargs.get('energy_score_weight')})",
        )

        # THE ONE THAT WOULD HAVE CAUGHT aug26's E25 AND E26.
        # get_energy_score raises unless there are exactly two members, and
        # EnsembleLoss.forward calls it whenever energy_score_weight > 0, so
        # this combination dies on the first training batch -- after config
        # validation, after dataset construction, after the model is built.
        # Verified on a GPU node 2026-09-03.
        want(
            not (energy_w > 0 and MEMBERS[levels["mem"]] != 2),
            f"mem-{levels['mem']} with energy_score_weight {energy_w}: "
            f"get_energy_score supports exactly two members and raises on the "
            f"first training batch otherwise",
        )

        fd = FDCRPS[levels["fdcrps"]]
        if fd:
            want(
                kwargs.get("finite_difference_crps_weight") == 0.1
                and kwargs.get("finite_difference_crps_levels") == fd,
                f"fdcrps-{levels['fdcrps']} but kwargs are "
                f"{kwargs.get('finite_difference_crps_weight')} / "
                f"{kwargs.get('finite_difference_crps_levels')}",
            )
        else:
            want(
                "finite_difference_crps_weight" not in kwargs,
                "fdcrps-0 but finite_difference_crps_weight is set",
            )
        alpha = ALPHA[levels["alpha"]]
        if levels["alpha"] == BASELINE["alpha"]:
            want(
                "almost_fair_crps_alpha" not in kwargs,
                "alpha-100 is the default, so it must not be written out",
            )
        else:
            want(
                kwargs.get("almost_fair_crps_alpha") == alpha,
                f"alpha-{levels['alpha']} but almost_fair_crps_alpha is "
                f"{kwargs.get('almost_fair_crps_alpha')}",
            )
        # almost_fair_crps_alpha only parameterizes the CRPS module, which
        # forward gates on crps_weight > 0; the finite-difference term is gated
        # on its own weight alone.  So at crps-energy the first is inert and the
        # second is not, and only one of them can be quietly wrong.
        if levels["crps"] == "energy":
            want(
                levels["alpha"] == BASELINE["alpha"],
                "crps-energy with a non-default alpha: inert at crps_weight 0",
            )
            want(
                levels["fdcrps"] == BASELINE["fdcrps"],
                "crps-energy with fdcrps set: that term is gated on its own "
                "weight, so it would still run and the id would understate the "
                "objective",
            )

    # -- sizing --------------------------------------------------------------
    batch = d["train_loader"]["batch_size"]
    want(
        d["validation"]["loader"]["batch_size"] == batch,
        "validation batch_size differs from train batch_size",
    )
    ranks, rem = divmod(batch, LOCAL_BATCH)
    want(rem == 0, f"batch {batch} is not a multiple of local batch {LOCAL_BATCH}")
    nodes, node_rem = divmod(ranks, GPUS_PER_NODE)
    want(node_rem == 0 and nodes >= 1, f"{ranks} ranks is not a whole node count")

    # -- inference -----------------------------------------------------------
    epochs = d["max_epochs"]
    stride = max(1, epochs // INFERENCE_EVALUATIONS)
    start = (epochs - 1) % stride
    for block in d.get("inference", []):
        name = block.get("name")
        n = len(block["loader"]["start_indices"]["times"])
        want(
            n % ranks == 0,
            f"block {name!r} has {n} initial conditions, which do not divide "
            f"{ranks} ranks",
        )
        want(
            block["n_forward_steps"] == INFERENCE_YEARS * STEPS_PER_YEAR,
            f"block {name!r} rollout is {block['n_forward_steps']}, not "
            f"{INFERENCE_YEARS} years",
        )
        got = block.get("epochs") or {}
        want(
            got.get("start") == start and got.get("step") == stride,
            f"block {name!r} epochs {got} do not score the last epoch at "
            f"max_epochs {epochs} (want start {start}, step {stride})",
        )
        fires = list(range(1, epochs + 1))[got.get("start", 0) :: got.get("step", 1)]
        want(
            bool(fires) and fires[-1] == epochs,
            f"block {name!r} last fires at epoch {fires[-1] if fires else None}, "
            f"not {epochs}",
        )

    # -- things that must not drift -----------------------------------------
    for k, v in EMBED.items():
        want(builder.get(k) == v, f"builder {k} is {builder.get(k)}, not {v}")
    for field, expected in WANDB.items():
        want(
            d["logging"].get(field) == expected,
            f"wandb {field} is {d['logging'].get(field)}, not {expected}",
        )
    want(
        "/pscratch/" not in path.read_text(),
        "a generated file names someone's scratch; runs/ has to be "
        "byte-identical for every teammate",
    )

    # A weighted inference block selects the checkpoint, so its whole
    # trajectory has to stay out of the validation window.
    for block in d.get("inference", []):
        if not block.get("weight"):
            continue
        span_years = block["n_forward_steps"] / STEPS_PER_YEAR
        for t in block["loader"]["start_indices"]["times"]:
            want(
                not _overlaps(t, span_years),
                f"weighted block {block.get('name')!r} starts at {t} and runs "
                f"{span_years:g} years, crossing the {VALIDATION_WINDOW[0][:4]}-"
                f"{VALIDATION_WINDOW[1][:4]} validation window",
            )
    return bad


def _overlaps(start: str, span_years: float) -> bool:
    lo = int(VALIDATION_WINDOW[0][:4])
    hi = int(VALIDATION_WINDOW[1][:4])
    s = int(str(start)[:4])
    e = s + span_years
    return s < hi and e > lo


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE / "runs"))
    args = p.parse_args(argv)

    paths = sorted(pathlib.Path(args.dir).glob("*.yaml"))
    if not paths:
        print(f"no configs in {args.dir}", file=sys.stderr)
        return 2
    bad = [c for path in paths for c in check(path)]
    for c in bad:
        print(c, file=sys.stderr)
    print(f"checked {len(paths)} configs, {len(bad)} complaints")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
