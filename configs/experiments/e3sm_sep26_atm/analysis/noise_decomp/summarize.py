"""Tabulate the evaluator diagnostics of several noise-decomposition runs.

Reads, per run directory:
    time_mean_diagnostics.nc      bias_map-<var>
        -> area-weighted RMSE and mean of the time-mean bias
    power_spectrum_diagnostics.nc <var>(source, wavenumber)
        -> mean log10 ratio prediction/target over the low and high thirds of l
    histogram_diagnostics.nc      <var>(source, bin)
        -> 0.1th and 99.9th percentile, prediction vs target
    mean_step_<n>_diagnostics.nc  weighted_rmse-<var>
        -> RMSE at lead n (6 h steps)

Usage: python summarize.py <run_dir> [<run_dir> ...] [--vars Tat2m,PS,...]
"""

import argparse
import glob
import os
import re

import numpy as np
import xarray as xr

DEFAULT_VARS = [
    "Tat2m",
    "PS",
    "surface_precipitation_rate",
    "FLUT",
    "U_6",
    "T_7",
    "STW_7",
]


def wrmse(field: xr.DataArray) -> float:
    w = np.cos(np.deg2rad(field["lat"]))
    return float(np.sqrt((field**2).weighted(w).mean(("lat", "lon"))))


def wmean(field: xr.DataArray) -> float:
    w = np.cos(np.deg2rad(field["lat"]))
    return float(field.weighted(w).mean(("lat", "lon")))


def quantile_from_hist(counts, edges, q):
    c = np.cumsum(counts) / counts.sum()
    i = np.searchsorted(c, q)
    i = min(i, len(edges) - 2)
    return float(edges[i + 1])


def summarize(run_dir, variables):
    out = {}
    tm = xr.open_dataset(os.path.join(run_dir, "time_mean_diagnostics.nc"))
    ps = xr.open_dataset(os.path.join(run_dir, "power_spectrum_diagnostics.nc"))
    hist_path = os.path.join(run_dir, "histogram_diagnostics.nc")
    hist = xr.open_dataset(hist_path) if os.path.exists(hist_path) else None
    steps = {}
    for f in glob.glob(os.path.join(run_dir, "mean_step_*_diagnostics.nc")):
        m = re.search(r"mean_step_(\d+)_diagnostics", f)
        if m:
            steps[int(m.group(1))] = xr.open_dataset(f)
    for v in variables:
        r = {}
        if f"bias_map-{v}" in tm:
            r["tm_bias_rmse"] = wrmse(tm[f"bias_map-{v}"])
            r["tm_bias_mean"] = wmean(tm[f"bias_map-{v}"])
        if v in ps:
            p = ps[v].sel(source="prediction").values
            t = ps[v].sel(source="target").values
            n = len(p)
            lo, hi = slice(1, n // 3), slice(2 * n // 3, n)
            r["spec_ratio_lo"] = float(np.mean(np.log10(p[lo] / t[lo])))
            r["spec_ratio_hi"] = float(np.mean(np.log10(p[hi] / t[hi])))
        if hist is not None and v in hist:
            cp = hist[v].sel(source="prediction").values
            ct = hist[v].sel(source="target").values
            ep = hist[f"{v}_bin_edges"].sel(source="prediction").values
            et = hist[f"{v}_bin_edges"].sel(source="target").values
            for q, name in ((0.999, "q999"), (0.001, "q001")):
                r[f"{name}_pred"] = quantile_from_hist(cp, ep, q)
                r[f"{name}_targ"] = quantile_from_hist(ct, et, q)
        for s, ds in sorted(steps.items()):
            if f"weighted_rmse-{v}" in ds:
                r[f"rmse_step{s}"] = float(ds[f"weighted_rmse-{v}"])
        out[v] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--vars", default=",".join(DEFAULT_VARS))
    a = ap.parse_args()
    variables = a.vars.split(",")
    results = {
        os.path.basename(os.path.normpath(d)): summarize(d, variables)
        for d in a.run_dirs
    }
    names = list(results)
    for v in variables:
        keys = sorted(
            {k for n in names for k in results[n][v]},
            key=lambda k: (not k.startswith("rmse"), k),
        )
        print(f"\n== {v}")
        print(f"{'metric':>18} " + " ".join(f"{n:>12}" for n in names))
        for k in keys:
            row = " ".join(
                f"{results[n][v][k]:12.4g}" if k in results[n][v] else f"{'-':>12}"
                for n in names
            )
            print(f"{k:>18} {row}")
