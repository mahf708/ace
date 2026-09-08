#!/usr/bin/env python
"""Average several per-epoch EMA checkpoints into one, to widen the EMA window
after the fact.

The template sets `ema.decay: 0.999`, which at one update per batch and 8,217
batches per epoch is a window of 1,000 updates -- **0.12 of an epoch**, and ten
times shorter than fme's own default of 0.9999. So "epoch N averaged weights"
is the average over roughly the last eighth of epoch N, not across epochs.

That entangles two axes the campaign has only explored one of. RF01's 8x
degradation after epoch 9 is either the optimum being passed or an EMA window
too short to suppress late-training iterate noise, and the difference decides
whether C2's scoring rule is a finding or a workaround. C2 already measured
that raw and averaged weights differ by 2.4x at epoch 22 (1.750 K against
4.130 K, 90-day RMSE), so a 1,000-update window is doing a great deal of work.

`ema_ckpt_NNNN.tar` is saved every epoch and already holds averaged weights in
`stepper.step.module` with no `ema_params` beside them -- the same shape as
`best_ckpt.tar`. Averaging k of them is therefore plain SWA over a k-epoch
window, and needs no retraining.

Entries that are not floating-point tensors -- `None` placeholders, counters,
integer buffers -- are taken from the last epoch in the window rather than
averaged. The accumulator is float32 and updated as a
running mean, so peak memory is two checkpoints rather than k.

    ./swa_checkpoint.py <run-dir> 15 21 <out-dir>
    ./swa_checkpoint.py <run-dir> 21 21 <out-dir>   # round-trip: one epoch
"""

import argparse
import os
import pathlib

import torch


def swa(ckpt_dir: pathlib.Path, first: int, last: int) -> dict:
    """Running mean of `stepper.step.module` over ema_ckpt_{first..last}."""
    paths = [ckpt_dir / f"ema_ckpt_{e:04d}.tar" for e in range(first, last + 1)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing checkpoints: {', '.join(missing)}")

    out: dict | None = None
    acc: dict[str, torch.Tensor] = {}
    for n, path in enumerate(paths, start=1):
        c = torch.load(path, map_location="cpu", weights_only=False)
        if c.get("ema", {}).get("ema_params"):
            raise SystemExit(
                f"{path.name} carries ema_params, so it holds RAW weights; "
                "this tool averages already-folded ema_ckpt files"
            )
        module = c["stepper"]["step"]["module"]
        for k, v in module.items():
            if not torch.is_tensor(v) or not torch.is_floating_point(v):
                # None entries, counters and integer buffers: take the last
                acc[k] = v
            elif n == 1:
                acc[k] = v.clone()
            else:
                acc[k].add_((v - acc[k]) / n)
        out = c  # keep the last epoch's scaffolding
        del c, module
    assert out is not None
    out["stepper"]["step"]["module"] = acc
    out["swa_window"] = [first, last]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=pathlib.Path, help="run directory")
    p.add_argument("first", type=int)
    p.add_argument("last", type=int)
    p.add_argument("out", type=pathlib.Path, help="directory to write")
    a = p.parse_args()
    if a.last < a.first:
        raise SystemExit("last epoch is before first")

    merged = swa(a.run / "training_checkpoints", a.first, a.last)
    target = a.out / "training_checkpoints" / "best_ckpt.tar"
    os.makedirs(target.parent, exist_ok=True)
    torch.save(merged, target)
    n = a.last - a.first + 1
    print(f"averaged {n} epoch(s) {a.first}-{a.last} -> {target}")


if __name__ == "__main__":
    main()
