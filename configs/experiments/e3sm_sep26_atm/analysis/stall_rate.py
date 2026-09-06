"""Stall frequency and lost wall clock, from the trainer's own `Step N:` lines.

`steprate.py` answers "how fast is a step"; this answers "how much of the run
was spent not stepping", which is the question a filesystem change is judged on.
The two are different measurements and the median cannot substitute for either:
on RF02 the median interval was 69-71 s on CFS and on Lustre alike, while CFS
lost 39-54% of wall clock to stalls and Lustre lost none.

A stall is an interval above `--threshold` times the median (default 5x). The
epoch boundary is excluded rather than counted: at each multiple of
`--epoch-batches` the trainer runs validation and writes three checkpoints,
which costs 330-900 s on every seed on every filesystem, and counting it turns a
clean run into one that appears to stall once per epoch.

    ./stall_rate.py $CAMPAIGN_ROOT/RF02*/out.log
    ./stall_rate.py --since '2026-09-06 09:38:38' .../S03/out.log   # after a restart
"""

import argparse
import datetime as dt
import os
import re
import statistics as st

STEP = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d+) .*?Step (\d+):")
# A resumed segment continues the batch counter, so a restart does not show up as
# a counter reset -- the gap across it would otherwise be the largest "stall" in
# the run. The trainer prints this on every start, including a resume.
RESTART = re.compile(r"Initializing training data loader")
EPOCH_BATCHES = 8217  # sep26 atm: 1 epoch at batch_size 16 over the training subset


def read_points(path, since=None, until=None):
    """Return (timestamp, batch, segment) per `Step N:` line.

    `segment` increments at each trainer start, so intervals are never taken
    across a restart.
    """
    pts, segment = [], 0
    for line in open(path, errors="ignore"):
        if RESTART.search(line):
            segment += 1
            continue
        m = STEP.match(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        ts = t.timestamp() + int(m.group(2)) / 1000
        if (since and ts < since) or (until and ts >= until):
            continue
        pts.append((ts, int(m.group(3)), segment))
    return pts


def restart_gap(pts):
    """Seconds spent between segments -- requeue, resubmit, or a crash and resume."""
    return sum(t1 - t0 for (t0, _, s0), (t1, _, s1) in zip(pts, pts[1:]) if s1 != s0)


def stalls(pts, epoch_batches, threshold):
    """Return (intervals, boundary_intervals) in seconds per 100 batches.

    A restart resets the step counter, so a non-increasing batch number ends one
    segment and begins the next rather than producing a negative interval.
    """
    gaps, boundary = [], []
    for (t0, b0, s0), (t1, b1, s1) in zip(pts, pts[1:]):
        if b1 <= b0 or s1 != s0:
            continue
        per100 = (t1 - t0) / ((b1 - b0) / 100)
        if epoch_batches and (b0 // epoch_batches) != (b1 // epoch_batches):
            boundary.append(per100)
        else:
            gaps.append(per100)
    if not gaps:
        return None
    med = st.median(gaps)
    return gaps, boundary, med, [g for g in gaps if g > threshold * med]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("logs", nargs="+", help="out.log files")
    p.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="multiple of the median that counts as a stall (default 5)",
    )
    p.add_argument(
        "--epoch-batches",
        type=int,
        default=EPOCH_BATCHES,
        help="batches per epoch; 0 to count boundaries as ordinary intervals",
    )
    p.add_argument("--since", help="ignore lines before 'YYYY-MM-DD HH:MM:SS' (local)")
    p.add_argument("--until", help="ignore lines from 'YYYY-MM-DD HH:MM:SS' onward")
    a = p.parse_args()

    def stamp(s):
        return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp() if s else None

    since, until = stamp(a.since), stamp(a.until)
    total_stalls = total_min = 0.0
    print(
        f"{'run':28s} {'span':>7s} {'n':>4s} {'med':>7s} {'max':>9s} "
        f"{'stalls':>6s} {'lost':>16s}  epoch boundary"
    )
    for path in a.logs:
        pts = read_points(path, since, until)
        if len(pts) < 3:
            print(
                f"{os.path.basename(os.path.dirname(path))[-28:]:28s} "
                f"too few `Step` lines ({len(pts)})"
            )
            continue
        got = stalls(pts, a.epoch_batches, a.threshold)
        if not got:
            continue
        gaps, boundary, med, bad = got
        span = pts[-1][0] - pts[0][0] - sum(boundary) - restart_gap(pts)
        lost = sum(g - med for g in bad)
        # run ids differ only in their seed suffix, so elide the head, not the tail
        name = os.path.basename(os.path.dirname(path))
        if len(name) > 28:
            name = "..." + name[-25:]
        print(
            f"{name:28s} {span/60:6.0f}m {len(gaps):4d} {med:6.1f}s "
            f"{max(gaps):8.1f}s {len(bad):6d} {lost/60:8.1f}m ({100*lost/span:4.1f}%)  "
            + " ".join(f"{b:.0f}s" for b in boundary)
        )
        total_stalls += len(bad)
        total_min += span / 60
    if len(a.logs) > 1 and total_stalls:
        print(
            f"\npooled: {total_stalls:.0f} stalls / {total_min:.0f} min "
            f"= one per {total_min/total_stalls:.1f} min"
        )


if __name__ == "__main__":
    main()
