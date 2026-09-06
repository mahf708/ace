#!/usr/bin/env python
"""Say which epoch a checkpoint is, what it was selected on, and which weights
it holds.

Two traps, both of which have already cost a result here.

`best_ckpt.tar` is rewritten whenever validation loss improves, so two
evaluations of one arm can be two epochs, and two seeds evaluated on the same
day are almost certainly at different ones. That turns a seed spread into a
mixture of seed and epoch.

Worse, the two kinds of checkpoint do not hold the same weights at the same
epoch. MEASURED on RF01.S01 at epoch 22: all 135 tensors of `ckpt_0022.tar`'s
`ema` are bit-identical to `best_ckpt.tar`'s `stepper`, and differ from
`ckpt_0022.tar`'s own `stepper` by up to 1.2e-2. So **`best_ckpt.tar` holds the
averaged weights and `ckpt_NNNN.tar` holds the raw ones**, the evaluator loads
`stepper`, and scoring one against the other compares two models. The `weights`
column below says which is which; check it before quoting any comparison that
spans checkpoint files.

    ./checkpoint_epoch.py <ckpt.tar> [<ckpt.tar> ...]
    ./checkpoint_epoch.py $PSCRATCH/sep26-pin/RF01.S0*/training_checkpoints/*.tar
"""

import argparse
import sys

import torch


def weights_kind(checkpoint: dict) -> str:
    """Whether the weights the evaluator will load are averaged or raw.

    The evaluator loads ``stepper``. A periodic training checkpoint keeps raw
    weights there and the running average beside them under ``ema.ema_params``;
    a best/EMA snapshot has already folded the average into ``stepper`` and
    stores no ``ema_params`` at all. So the presence of that key is the
    discriminator, and it is one dictionary lookup rather than a tensor sweep.

    Verified by name-matched equality on RF01.S01 at epoch 22, where both files
    exist: all 135 of ``ckpt_0022.tar``'s ``ema_params`` are bit-identical to
    ``best_ckpt.tar``'s ``stepper``, and differ from ``ckpt_0022.tar``'s own
    ``stepper`` by up to 1.2e-2.
    """
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict):
        return "?"
    return "raw" if ema.get("ema_params") else "ema"


def read(path: str) -> dict:
    # weights_only=False: these are training checkpoints holding optimizer and
    # scheduler state, not just tensors, and they are ours.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "epoch": checkpoint.get("epoch"),
        "batches": checkpoint.get("num_batches_seen"),
        "val_loss": checkpoint.get("best_validation_loss"),
        "inference_error": checkpoint.get("best_inference_error"),
        "weights": weights_kind(checkpoint),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    print(
        f"{'checkpoint':<46}{'epoch':>7}{'batches':>10}{'val_loss':>12}{'weights':>9}"
    )
    seen: set[str] = set()
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
            f"{'' if val is None else f'{val:>12.6f}'}{info['weights']:>9}"
        )
        seen.add(info["weights"])
    if len(seen) > 1:
        print(
            f"\nWARNING: this set mixes weight kinds {sorted(seen)}. Comparing "
            "across them compares two models, not two runs.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
