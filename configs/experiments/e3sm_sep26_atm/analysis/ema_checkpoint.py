#!/usr/bin/env python
"""Fold a periodic checkpoint's running average into the weights the evaluator
will load, so an arbitrary epoch can be scored the way `best_ckpt.tar` is.

`best_ckpt.tar` holds averaged weights in `stepper` and no `ema_params`.
`ckpt_NNNN.tar` holds raw weights in `stepper` and the average beside them, so
scoring it evaluates a different model -- by up to 1.2e-2 per tensor on RF01,
which is enough to reorder seeds. Comparisons that span the two kinds are
therefore meaningless, and that is not obvious from either file.

This writes a third thing: `ckpt_NNNN.tar` with its own average moved into
`stepper` and `ema_params` dropped, which is bit-for-bit what the trainer would
have saved as `best_ckpt.tar` had that epoch been the best one. It makes the
comparison the campaign actually needs possible -- **averaged weights, several
seeds, one epoch** -- which no pair of files on disk supports.

The rename table is the checkpoint's own `ema.module_name_to_ema_name`, so
nothing here guesses at name mangling.

    ./ema_checkpoint.py <ckpt_NNNN.tar> <out.tar>
    ./ema_checkpoint.py <ckpt_NNNN.tar> <out.tar> --verify <best_ckpt.tar>
"""

import argparse
import sys

import torch


def fold_ema(checkpoint: dict) -> dict:
    """Move `ema.ema_params` into the module weights, in place."""
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict) or not ema.get("ema_params"):
        raise SystemExit(
            "no ema_params in this checkpoint: it already holds averaged "
            "weights, so there is nothing to fold"
        )
    params = ema["ema_params"]
    module = checkpoint["stepper"]["step"]["module"]
    renames = ema["module_name_to_ema_name"]
    missing = []
    for module_name, ema_name in renames.items():
        # the leading index is the ModuleList position, which the module's own
        # state dict does not carry
        key = module_name.split(".", 1)[1]
        if key not in module or ema_name not in params:
            missing.append(module_name)
            continue
        module[key] = params[ema_name]
    if missing:
        raise SystemExit(f"{len(missing)} names did not match, e.g. {missing[:3]}")
    # a folded checkpoint carries no separate average, which is what makes
    # `weights_kind` report it as averaged
    del ema["ema_params"]
    return checkpoint


def verify(folded: dict, reference_path: str) -> int:
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    ours = folded["stepper"]["step"]["module"]
    theirs = reference["stepper"]["step"]["module"]
    # a module state dict can carry non-tensor entries; compare the weights
    shared = [
        k
        for k in ours
        if k in theirs and torch.is_tensor(ours[k]) and torch.is_tensor(theirs[k])
    ]
    bad = [k for k in shared if not torch.equal(ours[k], theirs[k])]
    print(f"verify: {len(shared) - len(bad)}/{len(shared)} tensors identical")
    if bad:
        worst = max((ours[k] - theirs[k]).abs().max().item() for k in bad)
        print(f"  {len(bad)} differ, max|diff| = {worst:.3e}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", help="a ckpt_NNNN.tar carrying ema_params")
    parser.add_argument("out", help="where to write the folded checkpoint")
    parser.add_argument(
        "--verify",
        metavar="BEST_CKPT",
        help="assert the result matches this averaged checkpoint tensor for "
        "tensor; use the epoch where both exist to prove the fold",
    )
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    folded = fold_ema(checkpoint)
    status = verify(folded, args.verify) if args.verify else 0
    if status == 0:
        torch.save(folded, args.out)
        print(f"wrote {args.out} (epoch {folded.get('epoch')}, averaged weights)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
