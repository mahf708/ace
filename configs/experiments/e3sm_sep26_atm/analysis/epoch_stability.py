"""Tier 0.1: does the arm ranking (and the seed spread) at a low epoch match the
ranking at the highest epoch on disk?  Reads only files aug26 has already written."""
import glob, os, re, itertools
import numpy as np, xarray as xr

ROOT = os.path.expandvars("$PSCRATCH/aug26")
SEED_GROUP = "E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0"          # 3 seeds, A0_C0
ARM_GROUP = {                                                # 5 arms, all A3_C1
    "E07": "W1 flux upweight", "E08": "W2 dilution",
    "E09": "W4 zero STW_0", "E10": "X1 AMP", "E15": "W3 zero STW_1",
}

def val_series(run, var="weighted_rmse-channel_mean"):
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/{run}/output/val/epoch_*")):
        ep = int(d.rsplit("_", 1)[1])
        f = f"{d}/mean_norm_diagnostics.nc"
        if not os.path.exists(f):
            continue
        with xr.open_dataset(f) as ds:
            if var in ds:
                out[ep] = float(ds[var].values)
    return out

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    n = len(a)
    if n < 2:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])

# ---- the three-seed spread of the reference, as a function of epoch ----
seeds = {s: val_series(f"{SEED_GROUP}.S{s}") for s in ("01", "02", "03")}
common = sorted(set.intersection(*[set(v) for v in seeds.values()]))
print("=== REF-S three-seed spread, val weighted_rmse-channel_mean (normalized) ===")
print(f"{'epoch':>6} {'S01':>10} {'S02':>10} {'S03':>10} {'mean':>10} {'spread':>10} {'spread/mean':>12}")
spreads = {}
for ep in common:
    v = np.array([seeds[s][ep] for s in ("01", "02", "03")])
    sp = v.max() - v.min(); spreads[ep] = (v.mean(), sp)
    print(f"{ep:>6} {v[0]:>10.5f} {v[1]:>10.5f} {v[2]:>10.5f} {v.mean():>10.5f} {sp:>10.5f} {sp/v.mean():>11.2%}")

# ---- the five-arm family, all on one tuning set ----
runs = {}
for exp in ARM_GROUP:
    m = glob.glob(f"{ROOT}/{exp}.aug26.atm.*")
    if m:
        runs[exp] = val_series(os.path.basename(m[0]))
arm_common = sorted(set.intersection(*[set(v) for v in runs.values()]))
names = sorted(runs)
top = arm_common[-1]
print(f"\n=== five-arm family (A3_C1), ordering vs the deepest common epoch {top} ===")
print(f"{'epoch':>6} " + " ".join(f"{n:>9}" for n in names) + f" {'range':>9} {'rho vs top':>11} {'order':>28}")
ref = [runs[n][top] for n in names]
for ep in arm_common:
    v = [runs[n][ep] for n in names]
    order = ",".join(np.array(names)[np.argsort(v)])
    rng = max(v) - min(v)
    print(f"{ep:>6} " + " ".join(f"{x:>9.5f}" for x in v) + f" {rng:>9.5f} {spearman(v, ref):>11.3f}  {order:>28}")

print("\n=== discrimination: arm range against the reference seed spread ===")
print(f"{'epoch':>6} {'arm range':>11} {'seed spread':>12} {'ratio':>8}")
for ep in arm_common:
    if ep not in spreads:
        continue
    v = [runs[n][ep] for n in names]
    rng = max(v) - min(v); _, sp = spreads[ep]
    print(f"{ep:>6} {rng:>11.5f} {sp:>12.5f} {rng/sp if sp else float('nan'):>8.2f}")
