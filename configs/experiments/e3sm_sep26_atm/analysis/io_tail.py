"""Latency distribution of the training loader's own read, per filesystem.

Why this exists: on Perlmutter the *median* read is a poor guide. CFS through
DVS and Lustre scratch have medians within ~2.3x of each other and both stay
flat as concurrency rises, yet training loses 39-54% of wall clock on CFS and
none on Lustre. The damage is all in the tail, so measure the tail.

The probe replays one training sample -- every variable the stepper names, over
`time_buffer + n_timesteps` consecutive timesteps at a random offset -- and
scales the number of concurrent readers.

    srun -N 4 -n 64 --ntasks-per-node 16 \
        ./io_tail.py --backend cfs --reads 8 --config <run>/config.yaml > cfs.64
    ./io_tail.py --report cfs.64 scratch.64

MEASURED 2026-09-06 on four idle A100 nodes, 648 reads each:

    backend   median     p99      max   max/med
    cfs        4.619  46.052  529.746      115x
    scratch    1.970   3.046    3.978        2x

One caveat the numbers carry: this reopens the file per read, while the real
loader amortizes opens across many windows. The tail is real; the per-read
exceedance rate is an overestimate and must not be read as a stall rate.
"""

import argparse
import os
import random
import statistics as st
import sys
import time

DEFAULT_ROOTS = {
    "cfs": "/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run",
    "scratch": "/pscratch/sd/m/mahf708/v3.LR.historical_0101.aigo/run",
}
# The loader globs `*.eam.h0.*.nc`. A bare `*.nc` also matches the coupler
# history and restart files, which hold none of these variables and return
# almost instantly -- that dilutes the median and hides which reads were slow.
PATTERN = ".eam.h0."


def variables(config_path):
    import yaml

    cfg = yaml.safe_load(open(config_path))["stepper"]["step"]["config"]
    return sorted(set(cfg["in_names"]) | set(cfg["out_names"]))


def probe(root, names, reads, window, seed):
    import netCDF4 as nc

    rng = random.Random(seed)
    files = sorted(f for f in os.listdir(root) if PATTERN in f and f.endswith(".nc"))
    if not files:
        raise SystemExit(f"no {PATTERN}*.nc under {root}")
    out = []
    for _ in range(reads):
        path = os.path.join(root, rng.choice(files))
        t0 = time.time()
        d = nc.Dataset(path)
        n_time = len(d.dimensions["time"])
        start = rng.randrange(0, max(1, n_time - window))
        for v in names:
            if v in d.variables:
                d.variables[v][start : start + window]
        d.close()
        out.append(time.time() - t0)
    return out


def report(paths):
    print(
        f"{'backend':10s} {'readers':>7s} {'n':>5s} {'median':>8s} {'p95':>8s} "
        f"{'p99':>8s} {'max':>9s} {'max/med':>8s}"
    )
    for p in paths:
        lat, backend, ranks = [], "?", set()
        for line in open(p):
            f = line.split()
            if len(f) > 3 and f[0] == "RANK":
                ranks.add(f[1])
                backend = f[2]
                lat += [float(x) for x in f[3:]]
        if not lat:
            continue
        lat.sort()
        med = st.median(lat)
        print(
            f"{backend:10s} {len(ranks):7d} {len(lat):5d} {med:8.3f} "
            f"{lat[int(0.95 * len(lat))]:8.3f} {lat[int(0.99 * len(lat))]:8.3f} "
            f"{max(lat):9.3f} {max(lat)/med:7.0f}x"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=sorted(DEFAULT_ROOTS))
    p.add_argument("--root", help="override the path for --backend")
    p.add_argument("--config", help="run config.yaml to take the variable list from")
    p.add_argument("--reads", type=int, default=8)
    p.add_argument(
        "--window",
        type=int,
        default=12,
        help="timesteps per read: time_buffer + n_timesteps (default 12)",
    )
    p.add_argument(
        "--report",
        nargs="+",
        metavar="FILE",
        help="summarize saved probe output instead of measuring",
    )
    a = p.parse_args()

    if a.report:
        report(a.report)
        return
    if not a.backend or not a.config:
        p.error("--backend and --config are required unless --report is given")

    rank = int(os.environ.get("SLURM_PROCID", 0))
    names = variables(a.config)
    lat = probe(
        a.root or DEFAULT_ROOTS[a.backend], names, a.reads, a.window, 1234 + rank
    )
    print(f"RANK {rank} {a.backend} " + " ".join(f"{x:.4f}" for x in lat), flush=True)
    print(
        f"rank {rank}: {len(names)} variables x {a.window} timesteps, "
        f"{a.reads} reads",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
