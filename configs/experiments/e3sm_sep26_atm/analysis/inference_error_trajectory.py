"""The trainer's own `Inference error:` line, per epoch and across seeds.

C2 decided the scoring epoch from an *offline* sweep of two seeds at 1 d and
1 y.  This reads a third, independent witness that every run writes for free:
the in-training 5-year rollout the trainer runs on a cadence anyway.  It agrees
on the shape and puts the knee earlier, which is the direction a longer rollout
should move it.

Validation loss falls monotonically for all 30 epochs while this climbs 6-10x
after epoch ~12, so `best_ckpt.tar` -- selected on validation loss -- is close
to the worst checkpoint on disk for climate.  That is the whole reason
`SCORING_EPOCH` exists; this is the curve behind it.

    ./inference_error_trajectory.py $PSCRATCH/aug26/E01.aug26.atm.*/out.log
    ./inference_error_trajectory.py --start 2 --step 3 joblogs/RF02*.out
"""

import argparse
import re
import statistics as st

# Emitted by fme/core/generics/trainer.py on every inference epoch, unlike
# "Epoch inference error (...) is lower than ...", which only fires on an
# improvement and so silently hides the degradation this tool exists to show.
ERROR = re.compile(r"Inference error: ([0-9.]+(?:[eE][-+]?\d+)?)")


def series(paths, start, step):
    """{label: {epoch: error}}, one entry per run, in cadence order.

    A run that restarts appends to the same log, so reading in file order and
    numbering by cadence recovers the epoch even though the line itself does
    not carry one.
    """
    out = {}
    for path in paths:
        errs = [
            float(m.group(1))
            for m in map(ERROR.search, open(path, errors="ignore"))
            if m
        ]
        # `epochs: {start: 2, step: 3}` is 0-indexed; the checkpoint and the
        # output/inference/epoch_NNNN directory both count complete epochs.
        out[label(path)] = {start + 1 + step * i: e for i, e in enumerate(errs)}
    return out


def label(path):
    """Seed suffix when the runs differ only by seed, else the whole run id."""
    for part in reversed(path.split("/")):
        if m := re.search(r"\.(S\d\d)\b", part):
            return m.group(1)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("logs", nargs="+")
    p.add_argument("--start", type=int, default=2, help="inference.epochs.start")
    p.add_argument("--step", type=int, default=3, help="inference.epochs.step")
    args = p.parse_args()

    runs = series(args.logs, args.start, args.step)
    epochs = sorted({e for s in runs.values() for e in s})
    names = sorted(runs)

    print("epoch  " + "".join(f"{n:>9}" for n in names) + "     mean   spread")
    for e in epochs:
        vals = [runs[n].get(e) for n in names]
        got = [v for v in vals if v is not None]
        cells = "".join(f"{v:9.4f}" if v is not None else f"{'-':>9}" for v in vals)
        # Range over mean, not stdev: with three seeds the spread that matters
        # is how far apart the arms could look from seed alone.
        spread = (max(got) - min(got)) / st.mean(got) if len(got) > 1 else float("nan")
        print(f"{e:5d}  {cells}  {st.mean(got):7.4f}  {spread:6.0%}")


if __name__ == "__main__":
    main()
