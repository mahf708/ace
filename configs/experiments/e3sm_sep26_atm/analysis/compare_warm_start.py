"""Did the warm start actually transfer weights, or silently no-op?

CU01 loads a Z0 (deterministic, no noise convs) checkpoint into a Z1
architecture that has 8 tensors the checkpoint does not contain. It trains
either way -- a silent no-op looks identical from the log -- so compare the
child's weights against the parent's on the tensors they share.

A loaded child that has taken a few steps sits within ~1e-2 relative of the
parent. A child that ignored the checkpoint is an independent random draw and
sits at O(1).
"""

import sys

import torch


def module_of(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj["stepper"]["step"]["module"], obj.get("epoch")


parent, child = sys.argv[1], sys.argv[2]
p, _ = module_of(parent)
c, _ = module_of(child)

shared = [
    k
    for k in p
    if k in c
    and torch.is_tensor(p[k])
    and torch.is_floating_point(p[k])
    and p[k].shape == c[k].shape
    and p[k].numel() > 1
]
only_child = [k for k in c if k not in p]
print(
    f"parent tensors {len(p)} | child tensors {len(c)} | shared {len(shared)} "
    f"| only in child {len(only_child)}"
)

rel = []
for k in shared:
    d = (c[k].float() - p[k].float()).norm()
    n = p[k].float().norm().clamp(min=1e-12)
    rel.append(float(d / n))
rel.sort()
med = rel[len(rel) // 2]
print(
    f"relative difference on shared tensors: median {med:.3e}  "
    f"min {rel[0]:.3e}  max {rel[-1]:.3e}"
)

# A random re-init of the same architecture, as the scale for "not loaded".
torch.manual_seed(1234)
control = [
    float(
        (torch.randn_like(p[k].float()) * p[k].float().std() - p[k].float()).norm()
        / p[k].float().norm().clamp(min=1e-12)
    )
    for k in shared[: min(20, len(shared))]
]
control.sort()
print(f"an independent draw would sit near: {control[len(control)//2]:.3e}")
print()
print("VERDICT:", "WEIGHTS LOADED" if med < 0.1 else "NOT LOADED (silent no-op)")

if only_child:
    zeroed = sum(
        1
        for k in only_child
        if torch.is_tensor(c[k])
        and torch.is_floating_point(c[k])
        and float(c[k].abs().max()) == 0.0
    )
    print(
        f"\ntensors only in the child (the noise pathway): {len(only_child)}, "
        f"of which still exactly zero: {zeroed}"
    )
