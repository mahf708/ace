"""Is `S01` at Z0 the same shared-core initialization as `S01` at Z1/Z2?

The campaign differences arms against each other at a matched seed label. That
is only a paired comparison if the parameters the arms share are initialized
identically. Adding the noise-conditioning modules may advance the RNG before
later shared layers are built, which would break the pairing.
"""

import hashlib

import torch
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.rand import set_seed

IMG = (90, 180)
SEED = 0


def h(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:12]


def build(noise_embed_dim: int) -> dict[str, str]:
    set_seed(SEED)
    builder = NoiseConditionedSFNOBuilder(
        embed_dim=32,
        noise_embed_dim=noise_embed_dim,
        num_layers=2,
        noise_type="gaussian",
        pos_embed=False,
    )
    model = builder.build(
        n_in_channels=4,
        n_out_channels=4,
        dataset_info=DatasetInfo(img_shape=IMG),
    )
    return {k: h(v) for k, v in model.state_dict().items()}


def compare(a_dim: int, b_dim: int) -> None:
    a, b = build(a_dim), build(b_dim)
    shared = sorted(set(a) & set(b))
    same = [k for k in shared if a[k] == b[k]]
    diff = [k for k in shared if a[k] != b[k]]
    only_b = sorted(set(b) - set(a))
    print(f"\n--- Z(noise_embed_dim={a_dim}) vs Z(noise_embed_dim={b_dim}) ---")
    print(
        f"shared tensors: {len(shared)}   identical: {len(same)}   DIFFERENT: {len(diff)}"
    )
    print(f"only in {b_dim}: {len(only_b)}")
    if diff:
        print("  first differing shared tensors:")
        for k in diff[:8]:
            print(f"    {k}")
    print(f"PAIRED: {len(diff) == 0}")


def main():
    # reproducibility control: same dim twice must be identical
    a, b = build(32), build(32)
    print(f"control (32 vs 32, same seed) identical: {a == b}")
    compare(0, 32)
    compare(32, 64)


if __name__ == "__main__":
    with Distributed.context(handle_signals=False):
        main()
