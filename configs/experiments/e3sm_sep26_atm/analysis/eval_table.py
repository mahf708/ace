#!/usr/bin/env python
"""Read the scores pass into a comparison table.

The evaluator writes one ``ensemble_step_<lead>_diagnostics.nc`` per scored
lead, holding a scalar per variable per metric.  This collects them across
eval directories and answers the two questions the campaign actually asks:

  --seeds   how far apart are seeds of the same arm?  That spread is the
            noise floor: an arm difference smaller than it is not a result.
  --ladder  what does each noise override do, in units of that floor?

Usage:
    ./eval_table.py <eval-root> [--metric crps] [--vars Tat2m PS ...]
    ./eval_table.py <eval-root> --seeds
    ./eval_table.py <eval-root> --ladder
"""

import argparse
import os
import re
import statistics
import sys

import xarray as xr

# Directory names are <ARM>.<SEED>.eval-<pass>[-<noise>], written by
# make_eval_config.eval_id.
DIRNAME = re.compile(
    r"^(?P<arm>[A-Z]{2}\d{2})\.(?P<seed>S\d{2})\.eval-scores(-(?P<noise>\w+))?$"
)
FILENAME = re.compile(r"^ensemble_step_(?P<lead>\d+)_diagnostics\.nc$")

# 6-hourly steps to something a reader can hold in their head.
LEAD_NAMES = {1: "6h", 4: "1d", 20: "5d", 120: "30d", 360: "90d", 1460: "1y"}

DEFAULT_VARS = ["Tat2m", "PS", "surface_precipitation_rate", "FLUT", "U_6", "T_7"]
METRICS = ("crps", "ssr_bias", "ensemble_mean_rmse", "rank_bias", "rank_dispersion")


def lead_name(lead: int) -> str:
    return LEAD_NAMES.get(lead, f"{lead}st")


def collect(root: str) -> dict:
    """Read every eval dir under root into a nested mapping.

    Keyed (arm, seed, noise) -> lead -> metric -> variable -> value.
    """
    out: dict = {}
    for entry in sorted(os.listdir(root)):
        match = DIRNAME.match(entry)
        if match is None:
            continue
        noise = match.group("noise") or "keep"
        key = (match.group("arm"), match.group("seed"), noise)
        leads: dict = {}
        for filename in sorted(os.listdir(os.path.join(root, entry))):
            found = FILENAME.match(filename)
            if found is None:
                continue
            with xr.open_dataset(os.path.join(root, entry, filename)) as ds:
                values: dict = {}
                for name in ds.data_vars:
                    metric, _, var = str(name).partition("-")
                    if metric in METRICS:
                        values.setdefault(metric, {})[var] = float(ds[name].values)
            if values:
                leads[int(found.group("lead"))] = values
        if leads:
            out[key] = leads
    return out


def _cell(value: float | None, width: int = 10) -> str:
    if value is None:
        return " " * (width - 3) + "-- "
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:>{width}.2e}"
    return f"{value:>{width}.4g}"


def table(data: dict, metric: str, variables: list[str]) -> None:
    leads = sorted({lead for arm in data.values() for lead in arm})
    for var in variables:
        print(f"\n{metric}  --  {var}")
        header = f"{'run':<22}" + "".join(f"{lead_name(x):>10}" for x in leads)
        print(header)
        print("-" * len(header))
        for key in sorted(data):
            row = f"{'.'.join(key):<22}"
            for lead in leads:
                value = data[key].get(lead, {}).get(metric, {}).get(var)
                row += _cell(value)
            print(row)


def seeds(data: dict, variables: list[str]) -> None:
    """Seed-to-seed spread of the unmodified arm: the noise floor."""
    leads = sorted({lead for arm in data.values() for lead in arm})
    print("Seed spread at noise=keep -- mean +/- stdev over seeds, and the")
    print("coefficient of variation (stdev/mean), which is the fraction an arm")
    print("difference must exceed to mean anything.\n")
    for metric in METRICS:
        print(f"== {metric}")
        header = f"{'variable':<30}" + "".join(f"{lead_name(x):>18}" for x in leads)
        print(header)
        print("-" * len(header))
        for var in variables:
            row = f"{var:<30}"
            for lead in leads:
                values = [
                    data[key][lead][metric][var]
                    for key in data
                    if key[2] == "keep"
                    and lead in data[key]
                    and var in data[key][lead].get(metric, {})
                ]
                if len(values) < 2:
                    row += f"{'--':>18}"
                    continue
                mean = statistics.fmean(values)
                sd = statistics.stdev(values)
                cv = abs(sd / mean) if mean else float("nan")
                row += f"{mean:>10.4g}{cv:>8.1%}"
            print(row)
        print()


def ladder(data: dict, variables: list[str]) -> None:
    """Each noise override against keep, in units of the seed spread."""
    leads = sorted({lead for arm in data.values() for lead in arm})
    modes = sorted({key[2] for key in data} - {"keep"})
    if not modes:
        print("no noise-override runs found", file=sys.stderr)
        return
    print("Noise override minus keep, same seed, as a multiple of the")
    print("seed-to-seed stdev at keep. |z| < 1 is inside the noise floor.\n")
    for metric in METRICS:
        print(f"== {metric}")
        header = f"{'variable':<30}{'mode':<8}" + "".join(
            f"{lead_name(x):>10}" for x in leads
        )
        print(header)
        print("-" * len(header))
        for var in variables:
            for mode in modes:
                row = f"{var:<30}{mode:<8}"
                for lead in leads:
                    keeps = {
                        key[1]: data[key][lead][metric][var]
                        for key in data
                        if key[2] == "keep"
                        and lead in data[key]
                        and var in data[key][lead].get(metric, {})
                    }
                    pairs = [
                        (data[key][lead][metric][var], keeps[key[1]])
                        for key in data
                        if key[2] == mode
                        and key[1] in keeps
                        and lead in data[key]
                        and var in data[key][lead].get(metric, {})
                    ]
                    if not pairs or len(keeps) < 2:
                        row += _cell(None)
                        continue
                    sd = statistics.stdev(keeps.values())
                    delta = statistics.fmean(new - old for new, old in pairs)
                    row += _cell(delta / sd if sd else float("nan"))
                print(row)
            print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "root", nargs="+", help="eval roots holding <arm>.<seed>.eval-* dirs"
    )
    parser.add_argument("--metric", default="crps", choices=METRICS)
    parser.add_argument("--vars", nargs="+", default=DEFAULT_VARS)
    parser.add_argument("--seeds", action="store_true", help="the seed noise floor")
    parser.add_argument("--ladder", action="store_true", help="overrides vs keep")
    args = parser.parse_args(argv)

    # Several roots merge into one table so that, say, an epoch-matched set can
    # be read beside the default one. A later root wins a repeated key.
    data: dict = {}
    for root in args.root:
        data.update(collect(root))
    if not data:
        print(f"no scores-pass output under {' '.join(args.root)}", file=sys.stderr)
        return 1
    present = {
        var
        for key in data
        for lead in data[key]
        for metric in data[key][lead]
        for var in data[key][lead][metric]
    }
    variables = [v for v in args.vars if v in present]
    missing = [v for v in args.vars if v not in present]
    if missing:
        print(f"note: not in the output: {', '.join(missing)}", file=sys.stderr)
    if not variables:
        return 1

    runs = ", ".join(".".join(k) for k in sorted(data))
    print(f"{len(data)} runs under {' '.join(args.root)}\n  {runs}\n")
    if args.seeds:
        seeds(data, variables)
    elif args.ladder:
        ladder(data, variables)
    else:
        table(data, args.metric, variables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
