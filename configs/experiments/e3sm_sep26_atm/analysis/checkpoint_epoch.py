#!/usr/bin/env python
"""Say which epoch a checkpoint is, and what it was selected on.

`best_ckpt.tar` is rewritten whenever validation loss improves, so two
evaluations of one arm can be two epochs, and two seeds evaluated on the same
day are almost certainly at different ones. That turns a seed spread into a
mixture of seed and epoch. `make_eval_config.py` records size and mtime in
eval.env, which detects a change; this says what changed.

    ./checkpoint_epoch.py <ckpt.tar> [<ckpt.tar> ...]
    ./checkpoint_epoch.py $PSCRATCH/sep26-pin/RF01.S0*/training_checkpoints/*.tar
"""

import argparse
import sys

import torch


def read(path: str) -> dict:
    # weights_only=False: these are training checkpoints holding optimizer and
    # scheduler state, not just tensors, and they are ours.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "batches": checkpoint.get("num_batches_seen"),
        "val_loss": checkpoint.get("best_validation_loss"),
        "inference_error": checkpoint.get("best_inference_error"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    print(f"{'checkpoint':<46}{'epoch':>7}{'batches':>10}{'val_loss':>12}")
    for path in args.paths:
        try:
            info = read(path)
        except Exception as err:  # a partly-written checkpoint is the common case
            print(f"{path:<46}  unreadable: {err}", file=sys.stderr)
            continue
        label = "/".join(path.split("/")[-3:])
        val = info["val_loss"]
        print(
            f"{label:<46}{str(info['epoch']):>7}{str(info['batches']):>10}"
            f"{'' if val is None else f'{val:>12.6f}'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
