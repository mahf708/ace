"""Candidate fix for rank-identical noise: offset only the CUDA seed by rank.

Must (a) decorrelate the noise across ranks and (b) leave model
initialization bit-identical across ranks, since init runs on the CPU RNG.
"""

import hashlib
import os

import torch
import torch.distributed as td
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.rand import randn, set_seed

IMG = (90, 180)


def h(t):
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:12]


def main():
    Distributed.get_instance()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    torch.cuda.set_device(device)

    out = {}
    for label in ("current", "rank_offset_cuda_seed"):
        set_seed(0)
        if label == "rank_offset_cuda_seed":
            # the one-line candidate: CUDA stream only, CPU stream untouched
            torch.cuda.manual_seed_all(0 + 4 + 7919 * rank)

        model = (
            NoiseConditionedSFNOBuilder(
                embed_dim=32,
                noise_embed_dim=8,
                num_layers=2,
                noise_type="gaussian",
                pos_embed=False,
            )
            .build(4, 4, DatasetInfo(img_shape=IMG))
            .to(device)
        )

        init_hash = hashlib.sha256(
            "".join(h(v) for _, v in sorted(model.state_dict().items())).encode()
        ).hexdigest()[:12]
        noise_hash = h(randn(torch.Size([1, 8, *IMG]), device=device))
        out[label] = {"init": init_hash, "noise": noise_hash}

    g = [None] * world
    td.all_gather_object(g, {"rank": rank, **out})
    if rank == 0:
        print(
            f"{'variant':<24} {'init identical across ranks':<30} {'noise identical across ranks'}"
        )
        for label in ("current", "rank_offset_cuda_seed"):
            inits = [x[label]["init"] for x in g]
            noises = [x[label]["noise"] for x in g]
            print(
                f"{label:<24} {str(len(set(inits)) == 1):<30} {len(set(noises)) == 1}"
            )
        print("\nwanted: init identical=True (DDP consistency), noise identical=False")
    td.barrier()


if __name__ == "__main__":
    with Distributed.context(handle_signals=False):
        main()
