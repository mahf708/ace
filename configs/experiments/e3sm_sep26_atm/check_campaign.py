#!/usr/bin/env python3
"""Assert that every generated run config agrees with its own run id.

    ./check_campaign.py               # checks runs/, and the RF01 claim
    ./check_campaign.py --dir DIR

The tables below **duplicate** make_campaign.py's on purpose, and must keep
duplicating them.  A checker that imports the generator's mapping can only
prove the generator is self-consistent; one that re-derives the expected config
from the id independently catches a typo in that mapping, which is the failure
this file exists for.

What it does not do is prove a config *runs*.  aug26's checker passed E25 and
E26, which raise on their first training batch, because it verified that a
config agreed with its id and nothing more.  The two blocker rules below are the
specific lessons; the general point stands: only a real forward and backward
pass proves a config trains.
"""

import argparse
import pathlib
import sys
from collections.abc import Mapping

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CAMPAIGN = "sep26"
REALM = "atm"
BATCH = 16
LOCAL_BATCH = 1
GPUS_PER_NODE = 4
INFERENCE_EVALUATIONS = 10
INFERENCE_YEARS = 5
STEPS_PER_YEAR = 1460

# The stochastic pole is not generated here -- it is aug26's E01, already
# trained at three seeds.  That claim is only safe while the template still
# matches that config, so it is checked rather than asserted in prose.
AUG26_BASELINE = HERE.parent / "e3sm_hist_v20260812" / "config-train-atm.yaml"
RF01_WORD = "D0_G0_I0_M2_N0_Q0_R0_Y0_Z1"

# --- duplicated from make_campaign.py, deliberately -------------------------
POSITIONS = ("D", "G", "I", "M", "N", "Q", "R", "Y", "Z")
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
OBJECTIVE = {"0": "EnsembleLoss", "1": "MSE"}
SPLIT = {"0": (0.9, 0.1), "1": (1.0, 0.0), "2": (0.0, 1.0), "3": (0.5, 0.5)}
INIT = {"0": "scratch", "1": "warm"}
MEMBERS = {"1": 1, "2": 2, "3": 3}
NOISE_TYPE = {"0": "isotropic", "1": "gaussian"}
FDCRPS = {"0": 0, "1": 1, "3": 3}
FDCRPS_WEIGHT = 0.1
ROLLOUT: dict[str, tuple[int | dict[int, float], bool]] = {
    "0": (1, True),
    "1": (2, True),
    "2": (2, False),
    "3": ({1: 0.6, 2: 0.2, 4: 0.2}, True),
    "4": ({1: 0.6, 2: 0.2, 4: 0.1, 12: 0.05, 20: 0.05}, True),
}
ALPHA = {"0": 1.0, "1": 0.95}
NOISE_DIM = {"0": 0, "1": 32, "2": 64}
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
STUDIES = ("RF", "LG", "NC", "EN", "OI", "RO", "CU")
NOISE_TYPE_AT_ZERO = "gaussian"
WARM_START_PLACEHOLDER = "OVERRIDE_ME_WARM_START"

WANDB = {"project": "ACE2S-sep26-atm", "entity": "e3sm-aig"}
EMBED = {"embed_dim": 384}
VALIDATION_WINDOW = (1990, 1995)


def parse_runid(stem: str) -> tuple[str, dict[str, str], int]:
    """<exp>.<campaign>.<realm>.<word>.S<seed> -> exp, levels, seed."""
    fields = stem.split(".")
    if len(fields) != 5:
        raise ValueError(f"not <exp>.<campaign>.<realm>.<word>.S<seed>: {stem}")
    exp, campaign, realm, word, seed = fields
    if campaign != CAMPAIGN or realm != REALM:
        raise ValueError(f"not a {CAMPAIGN}.{REALM} run: {stem}")
    if len(exp) != 4 or exp[:2] not in STUDIES or not exp[2:].isdigit():
        raise ValueError(
            f"experiment id {exp!r} is not two study letters {list(STUDIES)} "
            f"plus two digits"
        )
    if not (seed.startswith("S") and seed[1:].isdigit()):
        raise ValueError(f"seed field is not S<digits>: {seed}")

    parts = word.split("_")
    if len(parts) != len(POSITIONS):
        raise ValueError(
            f"factor word {word!r} has {len(parts)} positions, expected "
            f"{len(POSITIONS)} ({'_'.join(p + '?' for p in POSITIONS)})"
        )
    levels: dict[str, str] = {}
    for part, pos in zip(parts, POSITIONS):
        if not part.startswith(pos):
            raise ValueError(
                f"factor word {word!r} is out of order: expected position "
                f"{pos} where {part!r} is"
            )
        level = part[len(pos) :]
        if level not in LEVELS[pos]:
            raise ValueError(f"unknown level {part!r} in {word}")
        levels[pos] = level
    return exp, levels, int(seed[1:])


def check(path: pathlib.Path) -> list[str]:
    """Complaints about one run config, empty if it is sound."""
    bad: list[str] = []
    try:
        exp, levels, seed = parse_runid(path.stem)
    except ValueError as e:
        return [f"{path.name}: {e}"]

    d = yaml.safe_load(path.read_text())
    training = d["stepper_training"]
    builder = d["stepper"]["step"]["config"]["builder"]["config"]

    def want(cond: bool, msg: str) -> None:
        if not cond:
            bad.append(f"{path.stem}: {msg}")

    want(d["seed"] == seed, f"id says seed {seed}, config says {d['seed']}")

    # -- the word, position by position --------------------------------------
    want(
        training["n_ensemble"] == MEMBERS[levels["M"]],
        f"M{levels['M']} but n_ensemble is {training['n_ensemble']}",
    )
    steps, last_only = ROLLOUT[levels["R"]]
    if isinstance(steps, int):
        want_steps: object = steps
    else:
        # A sampled rollout is a TimeLengthProbabilities: an `outcomes` list of
        # {steps, probability}, not the bare mapping the table is written as.
        want_steps = {
            "outcomes": [
                {"steps": n, "probability": p} for n, p in sorted(steps.items())
            ]
        }
    want(
        training["n_forward_steps"] == want_steps,
        f"R{levels['R']} but n_forward_steps is {training['n_forward_steps']}",
    )
    want(
        training["optimize_last_step_only"] == last_only,
        f"R{levels['R']} wants optimize_last_step_only={last_only}, got "
        f"{training['optimize_last_step_only']}",
    )
    noise = NOISE_DIM[levels["Z"]]
    want(
        builder["noise_embed_dim"] == noise,
        f"Z{levels['Z']} but noise_embed_dim is {builder['noise_embed_dim']}",
    )
    # Isotropic noise at zero channels calls an inverse SHT on a zero-channel
    # tensor and dies in the MKL FFT, so Z0 must carry the type too.
    # Z0 draws a zero-channel noise tensor, so neither type exists; a word
    # claiming N1 there would name a setting with no effect.
    want(
        not (noise == 0 and levels["N"] != BASELINE["N"]),
        f"N{levels['N']} with Z0: no noise of either type is drawn, so the "
        f"token claims a setting that has no effect",
    )
    want_type = NOISE_TYPE_AT_ZERO if noise == 0 else NOISE_TYPE[levels["N"]]
    want(
        builder["noise_type"] == want_type,
        f"expected noise_type {want_type}, got {builder['noise_type']}"
        + (" (isotropic at zero channels dies in the MKL FFT)" if noise == 0 else ""),
    )

    init = training.get("parameter_init", {}).get("weights_path")
    if levels["I"] == "1":
        want(
            init == WARM_START_PLACEHOLDER,
            f"I1 but parameter_init.weights_path is {init!r}, not the "
            f"placeholder run-train.sh resolves",
        )
    else:
        want(init is None, f"I0 but parameter_init.weights_path is {init!r}")

    # -- the loss ------------------------------------------------------------
    loss = training["loss"]
    want(
        loss["type"] == OBJECTIVE[levels["D"]],
        f"D{levels['D']} but loss type is {loss['type']}",
    )
    if levels["D"] == "1":
        want(
            not loss.get("kwargs"),
            f"D1 but loss.kwargs is {loss.get('kwargs')}; MSE discards them, so "
            f"the file would claim settings the run does not have",
        )
        for pos in ("G", "Q", "Y"):
            want(
                levels[pos] == BASELINE[pos],
                f"D1 with {pos}{levels[pos]}: unreachable under MSE",
            )
    else:
        kwargs = loss.get("kwargs") or {}
        crps_w, energy_w = SPLIT[levels["G"]]
        want(
            kwargs.get("crps_weight") == crps_w
            and kwargs.get("energy_score_weight") == energy_w,
            f"G{levels['G']} wants ({crps_w}, {energy_w}), got "
            f"({kwargs.get('crps_weight')}, {kwargs.get('energy_score_weight')})",
        )
        # BLOCKER 1 -- what would have caught aug26's E25 and E26.
        want(
            not (energy_w > 0 and MEMBERS[levels["M"]] != 2),
            f"M{levels['M']} with energy_score_weight {energy_w}: "
            f"get_energy_score supports exactly two members and raises on the "
            f"first training batch otherwise",
        )
        # BLOCKER 2 -- the mode_weights shape bug.
        want(
            crps_w > 0,
            f"G{levels['G']} leaves the energy score as the only loss "
            f"component; its shape carries two spurious leading dimensions and "
            f"get_channel_losses raises on the first training batch",
        )
        fd = FDCRPS[levels["Q"]]
        if fd:
            want(
                kwargs.get("finite_difference_crps_weight") == FDCRPS_WEIGHT
                and kwargs.get("finite_difference_crps_levels") == fd,
                f"Q{levels['Q']} but kwargs are "
                f"{kwargs.get('finite_difference_crps_weight')} / "
                f"{kwargs.get('finite_difference_crps_levels')}",
            )
        else:
            want(
                "finite_difference_crps_weight" not in kwargs,
                "Q0 but finite_difference_crps_weight is set",
            )
        if levels["Y"] == BASELINE["Y"]:
            want(
                "almost_fair_crps_alpha" not in kwargs,
                "Y0 is the default, so it must not be written out",
            )
        else:
            want(
                kwargs.get("almost_fair_crps_alpha") == ALPHA[levels["Y"]],
                f"Y{levels['Y']} but almost_fair_crps_alpha is "
                f"{kwargs.get('almost_fair_crps_alpha')}",
            )
            # BLOCKER 4 -- get_crps hard-codes epsilon = (1-alpha)/2, which is
            # the AIFS almost-fair definition only at two members.  MEASURED
            # 0.89% out at M3, 1.16% at M4.
            want(
                MEMBERS[levels["M"]] == 2,
                f"Y{levels['Y']} at M{levels['M']}: epsilon is hard-coded to "
                f"(1-alpha)/2, so this is not almost-fair CRPS at "
                f"{MEMBERS[levels['M']]} members",
            )
        # BLOCKER 3 -- with no noise channels the members are bit-identical, so
        # a pure-CRPS objective at M>1 is the M1 objective at M times the cost.
        want(
            not (
                NOISE_DIM[levels["Z"]] == 0
                and energy_w == 0.0
                and MEMBERS[levels["M"]] != 1
            ),
            f"M{levels['M']} with Z0 and G{levels['G']}: the members are "
            f"bit-identical, so this is the M1 objective at "
            f"{MEMBERS[levels['M']]}x the cost",
        )

    # -- sizing --------------------------------------------------------------
    batch = d["train_loader"]["batch_size"]
    want(batch == BATCH, f"train batch_size is {batch}, not {BATCH}")
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
        # A weighted block selects the checkpoint, so its whole trajectory has
        # to stay out of the validation window -- not just its first timestep.
        if block.get("weight"):
            span = block["n_forward_steps"] / STEPS_PER_YEAR
            for t in block["loader"]["start_indices"]["times"]:
                s = int(str(t)[:4])
                want(
                    not (s < VALIDATION_WINDOW[1] and s + span > VALIDATION_WINDOW[0]),
                    f"weighted block {name!r} starts at {t} and runs {span:g} "
                    f"years, crossing the {VALIDATION_WINDOW[0]}-"
                    f"{VALIDATION_WINDOW[1]} validation window",
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
    return bad


def check_rf01_claim() -> list[str]:
    """RF01 is not generated; it is aug26's E01.  Verify that is still true.

    The template was copied from that config.  If either drifts, the reference
    this campaign differences five arms against silently stops being the run
    those three trained seeds belong to -- and nothing else would notice.
    """
    template = HERE / "config-train-atm.template.yaml"
    if not AUG26_BASELINE.exists():
        return [f"cannot verify the RF01 claim: {AUG26_BASELINE} is missing"]
    a = yaml.safe_load(template.read_text())
    b = yaml.safe_load(AUG26_BASELINE.read_text())
    # experiment_dir and the W&B project are submission properties, not model
    # ones; everything that defines the trained model has to agree.
    for cfg in (a, b):
        cfg.pop("experiment_dir", None)
        cfg.get("logging", {}).pop("project", None)
        cfg.get("logging", {}).pop("entity", None)
    if a != b:
        differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        return [
            f"the template no longer matches aug26's config-train-atm.yaml "
            f"(differs in {differing}), so RF01 is not E01 and the three "
            f"reference seeds do not belong to this campaign"
        ]
    return []


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dir", default=str(HERE / "runs"))
    args = p.parse_args(argv)

    paths = sorted(pathlib.Path(args.dir).glob("*.yaml"))
    if not paths:
        print(f"no configs in {args.dir}", file=sys.stderr)
        return 2
    bad = [c for path in paths for c in check(path)]
    bad += check_rf01_claim()
    for c in bad:
        print(c, file=sys.stderr)
    print(f"checked {len(paths)} configs + the RF01 claim, {len(bad)} complaints")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
