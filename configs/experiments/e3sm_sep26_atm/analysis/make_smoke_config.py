#!/usr/bin/env python3
"""Turn a generated run config into a smoke test: same objective, tiny data.

    ./make_smoke_config.py LG04 [-o <dir>] [--years 3] [--nodes 1]

A config that parses is not a config that runs. Both upstream blockers this
campaign found were found by running an arm, not by validating one -- they
raise on the *first training batch*, after config validation, after dataset
construction, after the model is built. So every new axis gets a real forward
and backward pass before it is trusted.

The point is to change the objective as little as possible while making the
run cheap, so build DOWN from `runs/<runid>.yaml` rather than sideways from a
neighbouring smoke config. Sideways is how you end up smoke-testing something
that is not the arm.

What is changed, and nothing else:

  * the training subset shrinks to `--years` years and validation to one,
  * inference is dropped entirely (a smoke test is about the training step,
    and the 5-year blocks are 7300 steps against 16 initial conditions),
  * one epoch, and the batch drops to fit `--nodes` nodes,
  * checkpointing off, so a rerun never trips over a stale directory.

The loss, the member count, the rollout, the noise conditioning and the model
are left exactly as the campaign generated them.
"""

import argparse
import copy
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
GPUS_PER_NODE = 4


def _shrink(subset: dict, years: int) -> dict:
    """Keep the subset's start and pull the stop in to `years` after it."""
    start = subset["start_time"]
    return {"start_time": start, "stop_time": f"{int(start[:4]) + years}{start[4:]}"}


def _year_glob(first: int, last: int) -> str:
    """A glob over the file-name years, e.g. 1940..1942 -> ``194[0-2]``.

    Narrowing `subset` alone does NOT make a smoke test cheap: the loader
    still opens every file the pattern matches and subsets afterwards, so
    setup stays at its full ~20 minutes. The file pattern has to come down
    with it. Falls back to a decade wildcard when the range straddles one.
    """
    if first // 10 == last // 10:
        return f"{first // 10}[{first % 10}-{last % 10}]"
    if first // 100 == last // 100:
        return f"{first // 100}[{(first % 100) // 10}-{(last % 100) // 10}]*"
    return "*"


def _narrow_pattern(dataset: dict, years: int) -> None:
    """Point `file_pattern` at just the years the shrunken subset keeps."""
    first = int(dataset["subset"]["start_time"][:4])
    # stop_time is exclusive of the following year's first step in practice,
    # but keep the boundary file so the last window is never short.
    last = int(dataset["subset"]["stop_time"][:4])
    pattern = dataset.get("file_pattern", "")
    if ".h0." in pattern:
        head, _, tail = pattern.partition(".h0.")
        dataset["file_pattern"] = f"{head}.h0.{_year_glob(first, last)}-{tail}"


def smoke(config: dict, years: int, nodes: int, out_dir: str) -> dict:
    c = copy.deepcopy(config)
    c["experiment_dir"] = out_dir
    c["max_epochs"] = 1
    c["log_train_every_n_batches"] = 5
    # A smoke test must be rerunnable; the trainer refuses to makedirs over an
    # existing checkpoint dir, which turns a second attempt into a red herring.
    c["save_checkpoint"] = False
    c.pop("save_per_epoch_diagnostics", None)

    ranks = nodes * GPUS_PER_NODE
    c["train_loader"]["batch_size"] = ranks
    c["validation"]["loader"]["batch_size"] = ranks

    for member in c["train_loader"]["dataset"]["concat"]:
        member["subset"] = _shrink(member["subset"], years)
        _narrow_pattern(member, years)
    val = c["validation"]["loader"]["dataset"]
    val["subset"] = _shrink(val["subset"], 1)
    _narrow_pattern(val, 1)

    # Inference blocks are 7300 forward steps against 16 initial conditions,
    # most of which fall outside a shrunken subset anyway. The training step is
    # what has never run.
    c["inference"] = []
    return c


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("exp", help="experiment id, e.g. LG04")
    p.add_argument("-o", "--out-dir", default=None)
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--nodes", type=int, default=1)
    args = p.parse_args()

    matches = sorted(RUNS.glob(f"{args.exp}.*.yaml"))
    if not matches:
        print(f"no generated run for {args.exp!r} in {RUNS}", file=sys.stderr)
        return 1
    src = matches[0]

    root = pathlib.Path(
        args.out_dir
        or f"/pscratch/sd/{__import__('os').environ['USER'][0]}"
        f"/{__import__('os').environ['USER']}/sep26-smoke/{args.exp.lower()}"
    )
    root.mkdir(parents=True, exist_ok=True)
    config = smoke(yaml.safe_load(src.read_text()), args.years, args.nodes, str(root))
    dest = root / "config.yaml"
    dest.write_text(yaml.safe_dump(config, sort_keys=False))

    loss = config["stepper_training"]["loss"]
    builder = config["stepper"]["step"]["config"]["builder"]["config"]
    print(f"{src.name}\n  -> {dest}")
    print(f"  loss        {loss}")
    print(f"  n_ensemble  {config['stepper_training']['n_ensemble']}")
    print(
        f"  rollout     {config['stepper_training']['n_forward_steps']} "
        f"(last-only={config['stepper_training'].get('optimize_last_step_only')})"
    )
    print(
        f"  noise       dim={builder['noise_embed_dim']} type={builder['noise_type']}"
    )
    segments = len(config["train_loader"]["dataset"]["concat"])
    batch = config["train_loader"]["batch_size"]
    print(
        f"  data        {args.years} yr x {segments} segments, "
        f"batch {batch} on {args.nodes} node(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
