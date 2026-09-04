"""How strongly does the learned noise pathway modulate the network?

Noise z ~ N(0,1) with `noise_embed_dim` channels enters each conditional
layer norm through 1x1 convs W_scale_2d / W_bias_2d, both zero-initialised
(and W_scale.bias = 1, so step 0 is exactly the identity). The 1-sigma
modulation of an output channel is the L2 norm of its row of weights.
"""

import glob
import os
import re
import sys

import torch

CKPT_DIR = sys.argv[1]


def amplitude(sd):
    out = {}
    for kind in ("W_scale_2d", "W_bias_2d"):
        rows = []
        for k, v in sd.items():
            if kind in k and k.endswith("weight") and torch.is_tensor(v):
                w = v.float().flatten(1)  # (out_channels, noise_dim)
                rows.append(w.norm(dim=1))
        if rows:
            allrows = torch.cat(rows)
            out[kind] = (float(allrows.mean()), float(allrows.max()), len(rows))
    return out


paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.tar")))
print(
    f"{'epoch':>6} {'scale 1sd (mean)':>17} {'scale 1sd (max ch)':>19} {'bias 1sd (mean)':>16} {'n layers':>9}"
)
for p in paths:
    obj = torch.load(p, map_location="cpu", weights_only=False)
    ep = obj.get("epoch", re.search(r"ckpt_(\d+)", p).group(1))
    a = amplitude(obj["stepper"]["step"]["module"])
    s, b = a.get("W_scale_2d"), a.get("W_bias_2d")
    print(
        f"{str(ep):>6} {s[0]:>17.4f} {s[1]:>19.4f} {b[0]:>16.4f} {s[2]:>9}"
        if s and b
        else f"{str(ep):>6}  (no conditioning tensors)"
    )
    del obj
