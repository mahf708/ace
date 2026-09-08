#!/usr/bin/env python3
"""The held-out 5-year time-mean error, per epoch, from diagnostics on disk.

The template carries TWO inference blocks on the same cadence.  `inference`
(ICs 1940-2027, weight 1.0) is what the trainer's `Inference error:` line
reports and what selects `best_ckpt.tar`.  `5yr_test` (ICs 2040-2047, weight
0.0) runs beside it, contributes nothing to selection, and is written to
`output/5yr_test/epoch_NNNN/` because the template sets
`save_per_epoch_diagnostics`.

That is a validation/test split nobody was using.  Selecting an epoch on
`inference` and reporting on `5yr_test` is ordinary model selection rather than
selection on the reported metric, which is what C2 could not otherwise afford.

The scalar reproduced here is `time_mean_norm/rmse/channel_mean`: the cos(lat)
area-weighted RMS of each channel's normalised time-mean bias map, averaged
over channels.  VERIFIED against the trainer's own number on
RF02.S02 -- epoch 18 gives 0.0361 and epoch 21 gives 0.0285 on the `inference`
block, matching the log to four decimals.

    ./heldout_error_trajectory.py $PSCRATCH/sep26/RF02*
    ./heldout_error_trajectory.py --block inference $PSCRATCH/sep26/LG01*
"""

import argparse
import pathlib
import re

import numpy as np
import xarray as xr

FNAME = "time_mean_norm_diagnostics.nc"


def error(path: pathlib.Path) -> float | None:
    """`time_mean_norm/rmse/channel_mean` from one epoch's diagnostics."""
    try:
        ds = xr.open_dataset(path)
    except (FileNotFoundError, OSError):
        return None
    # Area weight, normalised to mean 1 so the RMS keeps the metric's scale.
    w = np.cos(np.deg2rad(ds.lat))
    w = w / w.mean()
    per_channel = [
        float(np.sqrt(((ds[v] ** 2) * w).mean()))
        for v in ds.data_vars
        if v.startswith("bias_map-")
    ]
    return float(np.mean(per_channel)) if per_channel else None


def series(run: pathlib.Path, block: str) -> dict[int, float]:
    out = {}
    for d in sorted((run / "output" / block).glob("epoch_*")):
        e = error(d / FNAME)
        if e is not None:
            out[int(d.name.split("_")[1])] = e
    return out


def label(run: pathlib.Path) -> str:
    m = re.search(r"\.(S\d\d)$", run.name)
    return m.group(1) if m else run.name[:12]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=pathlib.Path)
    p.add_argument(
        "--block", default="5yr_test", help="5yr_test (default) or inference"
    )
    a = p.parse_args()

    data = {label(r): series(r, a.block) for r in a.runs if r.is_dir()}
    data = {k: v for k, v in data.items() if v}
    if not data:
        print(f"no {a.block} diagnostics found")
        return
    cols = sorted(data)
    epochs = sorted({e for v in data.values() for e in v})
    print(
        f"[{a.block}]  epoch" + "".join(f"{c:>9}" for c in cols) + "     mean   spread"
    )
    for e in epochs:
        vals = [data[c].get(e) for c in cols]
        got = [v for v in vals if v is not None]
        cells = "".join(f"{v:9.4f}" if v is not None else f"{'-':>9}" for v in vals)
        m = np.mean(got)
        # Range over mean: with three seeds the honest question is how far apart
        # the arms could look from seed alone, not a stdev on n=3.
        spread = (max(got) - min(got)) / m if len(got) > 1 else float("nan")
        print(f"{e:16d}{cells}{m:9.4f}{spread:8.0%}")


if __name__ == "__main__":
    main()
