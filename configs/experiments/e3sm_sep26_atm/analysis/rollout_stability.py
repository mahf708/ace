"""Tier 0.1 on the metric the decision rule uses: the held-out 5-year rollout,
not one-step validation.  Reads only what aug26 has already written.
"""

import glob
import os

import numpy as np
import xarray as xr

ROOT = os.path.expandvars("$PSCRATCH/aug26")
REF = "E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0"
ARMS = ["E07", "E08", "E10", "E15", "E09"]
HEAD = ["Tat2m", "PS", "surface_precipitation_rate", "FLUT"]


def series(run, block="5yr_test"):
    """{epoch: (step20 normalized channel-mean rmse, {var: time-mean bias rmse})}"""
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/{run}/output/{block}/epoch_*")):
        ep = int(d.rsplit("_", 1)[1])
        s20 = np.nan
        f = f"{d}/mean_step_20_norm_diagnostics.nc"
        if os.path.exists(f):
            with xr.open_dataset(f) as ds:
                if "weighted_rmse-channel_mean" in ds:
                    s20 = float(ds["weighted_rmse-channel_mean"].values)
        tm = {}
        f = f"{d}/time_mean_diagnostics.nc"
        if os.path.exists(f):
            with xr.open_dataset(f) as ds:
                w = np.cos(np.deg2rad(ds["lat"].values))[:, None]
                for v in HEAD:
                    k = f"bias_map-{v}"
                    if k in ds:
                        b = ds[k].values
                        tm[v] = float(
                            np.sqrt((w * b**2).sum() / (w * np.ones_like(b)).sum())
                        )
        out[ep] = (s20, tm)
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 1 else float("nan")


seeds = {s: series(f"{REF}.S{s}") for s in ("01", "02", "03")}
common = sorted(set.intersection(*[set(v) for v in seeds.values()]))
print("=== REF-S three-seed spread on the HELD-OUT rollout (5yr_test) ===")
print("step-20 normalized weighted_rmse channel mean")
print(
    f"{'epoch':>6} {'S01':>10} {'S02':>10} {'S03':>10} {'mean':>10} {'spread':>10} {'spread/mean':>12}"
)
seed_sp = {}
for ep in common:
    v = np.array([seeds[s][ep][0] for s in ("01", "02", "03")])
    sp = v.max() - v.min()
    seed_sp[ep] = (v.mean(), sp)
    print(
        f"{ep:>6} {v[0]:>10.5f} {v[1]:>10.5f} {v[2]:>10.5f} {v.mean():>10.5f} {sp:>10.5f} {sp/v.mean():>11.2%}"
    )

print("\ntime-mean bias RMSE, area weighted, per headline variable")
for v in HEAD:
    row = []
    for ep in common:
        x = [seeds[s][ep][1].get(v, np.nan) for s in ("01", "02", "03")]
        row.append((ep, np.nanmean(x), np.nanmax(x) - np.nanmin(x)))
    print(f"  {v:>28} " + "  ".join(f"ep{ep}: {m:.4g} +-{s:.3g}" for ep, m, s in row))

runs = {}
for e in ARMS:
    m = glob.glob(f"{ROOT}/{e}.aug26.atm.*")
    if m:
        runs[e] = series(os.path.basename(m[0]))
ac = sorted(set.intersection(*[set(v) for v in runs.values()]))
names = [n for n in ARMS if n in runs]
top = ac[-1]
print(
    f"\n=== five-arm family on the held-out rollout, vs deepest common epoch {top} ==="
)
print(
    f"{'epoch':>6} "
    + " ".join(f"{n:>9}" for n in names)
    + f" {'range':>9} {'rho':>6} {'rho w/o E09':>12}   order"
)
ref_all = [runs[n][top][0] for n in names]
sub = [n for n in names if n != "E09"]
ref_sub = [runs[n][top][0] for n in sub]
for ep in ac:
    v = [runs[n][ep][0] for n in names]
    vs = [runs[n][ep][0] for n in sub]
    order = ",".join(np.array(names)[np.argsort(v)])
    print(
        f"{ep:>6} "
        + " ".join(f"{x:>9.5f}" for x in v)
        + f" {max(v)-min(v):>9.5f} {spearman(v, ref_all):>6.3f} {spearman(vs, ref_sub):>12.3f}   {order}"
    )

print("\n=== discrimination on the held-out rollout: arm range vs seed spread ===")
print(
    f"{'epoch':>6} {'range(all 5)':>13} {'range(no E09)':>14} {'seed spread':>12} {'ratio(no E09)':>14}"
)
for ep in ac:
    if ep not in seed_sp:
        continue
    v = [runs[n][ep][0] for n in names]
    vs = [runs[n][ep][0] for n in sub]
    _, sp = seed_sp[ep]
    r = (max(vs) - min(vs)) / sp if sp else float("nan")
    print(
        f"{ep:>6} {max(v)-min(v):>13.5f} {max(vs)-min(vs):>14.5f} {sp:>12.5f} {r:>14.1f}"
    )
