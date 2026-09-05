"""Per-trajectory statistics from an evaluator's autoregressive netCDF files.

Every statistic is computed WITHIN one trajectory (one sample) and then
summarised across samples, so it describes what a typical individual rollout
looks like rather than what an ensemble mean looks like. The target file gets
the same treatment, so each statistic has a truth value per initial
condition.

Per variable and sample:
    mean, std           area-weighted, pooled over the whole rollout
    skew, exkurt        pooled distribution (area-weighted)
    tstd                area-mean of the per-gridpoint temporal std
    ac1, ac4            area-mean lag-1 (6 h) and lag-4 (1 day) autocorrelation
                        of per-gridpoint anomalies from the trajectory's time mean
    q001..q999          area-weighted pooled quantiles
    wet_frac, wet_int   precipitation only: fraction of (t, x) above 1 mm/day,
                        and the mean rate given wet
    rmse_<lead>         area-weighted RMSE against the target at fixed leads

Usage: python traj_stats.py <run_dir> [--vars Tat2m,surface_precipitation_rate,PS]
Writes <run_dir>/traj_stats.npz and prints a table.
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

WET = 1.0 / 86400.0  # 1 mm/day in kg m-2 s-1
LEADS = {"6h": 1, "1d": 4, "5d": 20, "30d": 120, "180d": 720}


def open_ranks(run_dir: str, label: str) -> xr.Dataset:
    """Open the per-rank files a distributed evaluator writes, concatenated
    along ``sample``; fall back to the single-process file name.
    """
    import glob

    paths = sorted(glob.glob(os.path.join(run_dir, f"{label}_rank*.nc")))
    if not paths:
        paths = [os.path.join(run_dir, f"{label}.nc")]
    parts = [xr.open_dataset(p, engine="h5netcdf") for p in paths]
    if len(parts) == 1:
        return parts[0]
    # A run that was killed part-way -- a deadline, a requeue -- leaves the
    # rank that was mid-window with fewer timesteps than the others.  The
    # default outer join would pad that rank with NaN and every statistic
    # taken over the whole time axis silently becomes NaN, which reads as a
    # broken model rather than a truncated file.  Trim to the common length
    # and say so.
    lengths = {p.sizes["time"] for p in parts}
    if len(lengths) > 1:
        keep = min(lengths)
        print(
            f"note: {label} ranks have {sorted(lengths)} timesteps "
            f"(the run did not finish); truncating all to {keep}",
            file=sys.stderr,
        )
        parts = [p.isel(time=slice(0, keep)) for p in parts]
    return xr.concat(parts, dim="sample", join="exact")


def wquantiles(x, w, qs):
    x = x.ravel()
    # float64: a float32 cumsum of ~1e-5 weights over tens of millions of
    # elements loses the increments at the top of the array, i.e. exactly at
    # the upper quantiles.
    w = w.ravel().astype(np.float64)
    order = np.argsort(x)
    x = x[order]
    c = np.cumsum(w[order])
    c /= c[-1]
    return [float(x[min(np.searchsorted(c, q), len(x) - 1)]) for q in qs]


def stats_one(arr, w2d, is_precip):
    """arr: (time, lat, lon) float32; w2d: (lat, lon) weights summing to 1."""
    t = arr.shape[0]
    w3 = np.broadcast_to(w2d, arr.shape)
    mean = float((arr * w3).sum() / t)
    dev = arr - mean
    var = float((dev**2 * w3).sum() / t)
    std = np.sqrt(var)
    m3 = float((dev**3 * w3).sum() / t)
    m4 = float((dev**4 * w3).sum() / t)
    out = {
        "mean": mean,
        "std": std,
        "skew": m3 / std**3,
        "exkurt": m4 / var**2 - 3.0,
    }
    tmean = arr.mean(axis=0)
    anom = arr - tmean
    tvar = (anom**2).mean(axis=0)
    out["tstd"] = float((np.sqrt(tvar) * w2d).sum())
    for lag, name in ((1, "ac1"), (4, "ac4")):
        cov = (anom[lag:] * anom[:-lag]).mean(axis=0)
        ac = cov / np.maximum(tvar, 1e-30)
        out[name] = float((ac * w2d).sum())
    qs = [0.001, 0.01, 0.5, 0.99, 0.999]
    # subsample in time for the pooled quantiles (every 4th step) to bound memory
    sub = arr[::4]
    qv = wquantiles(sub, np.broadcast_to(w2d, sub.shape), qs)
    for q, v in zip(("q001", "q01", "q50", "q99", "q999"), qv):
        out[q] = v
    if is_precip:
        wet = arr > WET
        wet_frac = float((wet * w3).sum() / t)
        out["wet_frac"] = wet_frac
        out["wet_int"] = float((arr * wet * w3).sum() / t / max(wet_frac, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--vars", default="Tat2m,surface_precipitation_rate,PS")
    a = ap.parse_args()
    variables = a.vars.split(",")
    pred = open_ranks(a.run_dir, "autoregressive_predictions")
    targ = open_ranks(a.run_dir, "autoregressive_target")
    lat = pred["lat"].values
    w = np.cos(np.deg2rad(lat))
    w2d = np.broadcast_to(
        (w / w.sum())[:, None] / pred.sizes["lon"], (len(lat), pred.sizes["lon"])
    )
    n = pred.sizes["sample"]
    results = {}
    for v in variables:
        is_precip = "precipitation" in v
        per: dict[str, list[dict[str, float]]] = {"pred": [], "targ": []}
        for s in range(n):
            p = pred[v].isel(sample=s).values.astype(np.float32)
            y = targ[v].isel(sample=s).values.astype(np.float32)
            sp = stats_one(p, w2d, is_precip)
            st = stats_one(y, w2d, is_precip)
            for name, lead in LEADS.items():
                if lead - 1 < p.shape[0]:
                    d = p[lead - 1] - y[lead - 1]
                    sp[f"rmse_{name}"] = float(np.sqrt((d**2 * w2d).sum()))
            per["pred"].append(sp)
            per["targ"].append(st)
        results[v] = per
        keys = list(per["pred"][0])
        print(f"\n== {v}   ({n} trajectories; mean ± std across trajectories)")
        print(f"{'stat':>10} {'prediction':>24} {'target':>24}")
        for k in keys:
            pv = np.array([d[k] for d in per["pred"]])
            line = f"{k:>10} {pv.mean():12.5g} ± {pv.std():<9.3g}"
            if k in per["targ"][0]:
                tv = np.array([d[k] for d in per["targ"]])
                line += f" {tv.mean():12.5g} ± {tv.std():<9.3g}"
            print(line)
    np.savez(
        os.path.join(a.run_dir, "traj_stats.npz"),
        **{
            f"{v}/{src}/{k}": np.array([d[k] for d in results[v][src]])
            for v in results
            for src in ("pred", "targ")
            for k in results[v][src][0]
        },
    )


if __name__ == "__main__":
    main()
