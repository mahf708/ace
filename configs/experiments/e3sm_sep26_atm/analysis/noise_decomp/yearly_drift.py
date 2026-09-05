"""Year-by-year climate drift of individual trajectories, from the per-rank
autoregressive files (which are valid up to the last window written, so a
rollout that did not finish can still be read up to where it got).

Per variable, trajectory and rollout year: area-weighted RMS of the annual
time-mean bias map, and the area-mean per-gridpoint temporal std of the
prediction and of the target.

Usage: python yearly_drift.py <run_dir> [<run_dir> ...] [--vars Tat2m,PS]
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traj_stats import open_ranks  # noqa: E402

STEPS_PER_YEAR = 1460


def _valid_steps(pred, var: str) -> int:
    """Last fully written step + 1: the prediction is non-finite or zero
    beyond the last window a stalled run wrote.
    """
    probe = pred[var].isel(sample=0, lat=90, lon=180).values
    good = np.where(np.isfinite(probe) & (probe != 0))[0]
    return int(good.max()) + 1 if len(good) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--vars", default="Tat2m,PS,surface_precipitation_rate")
    a = ap.parse_args()
    variables = a.vars.split(",")
    for run in a.run_dirs:
        pred = open_ranks(run, "autoregressive_predictions")
        targ = open_ranks(run, "autoregressive_target")
        lat = pred["lat"].values
        w = np.cos(np.deg2rad(lat))
        w2d = np.broadcast_to(
            (w / w.sum())[:, None] / pred.sizes["lon"], (len(lat), pred.sizes["lon"])
        )
        n = pred.sizes["sample"]
        n_steps = _valid_steps(pred, variables[0])
        n_years = n_steps // STEPS_PER_YEAR
        name = os.path.basename(os.path.normpath(run))
        print(
            f"\n#### {name}: {n} trajectories, {n_steps} valid steps = {n_years} years"
        )
        for v in variables:
            print(f"== {v}   (mean ± std across trajectories)")
            print(
                f"{'year':>5} {'bias_rms':>18} {'tstd_pred':>18} {'tstd_targ':>18} "
                f"{'pred-targ %':>12}"
            )
            for y in range(n_years):
                sl = slice(y * STEPS_PER_YEAR, (y + 1) * STEPS_PER_YEAR)
                rows = []
                for s_ in range(n):
                    p = pred[v].isel(sample=s_, time=sl).values.astype(np.float32)
                    t = targ[v].isel(sample=s_, time=sl).values.astype(np.float32)
                    bias = p.mean(axis=0) - t.mean(axis=0)
                    rows.append(
                        (
                            float(np.sqrt((bias**2 * w2d).sum())),
                            float((p.std(axis=0) * w2d).sum()),
                            float((t.std(axis=0) * w2d).sum()),
                        )
                    )
                arr = np.array(rows)
                b, tp, tt = arr[:, 0], arr[:, 1], arr[:, 2]
                print(
                    f"{y + 1:>5} {b.mean():9.4g} ± {b.std():<6.2g} "
                    f"{tp.mean():9.4g} ± {tp.std():<6.2g} "
                    f"{tt.mean():9.4g} ± {tt.std():<6.2g} "
                    f"{100 * (tp.mean() / tt.mean() - 1):11.1f}%"
                )


if __name__ == "__main__":
    main()
