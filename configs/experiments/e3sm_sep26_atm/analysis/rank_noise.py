"""Does every data-parallel rank draw the SAME conditioning noise?

Runs the real NoiseConditionedModel.forward noise path (gaussian and
isotropic) on N ranks after the exact seeding train.py performs, and
compares per-sample noise hashes within and across ranks.
"""

import hashlib
import json
import os

import torch
import torch.distributed as td
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.distributed import Distributed
from fme.core.rand import set_seed

IMG = (90, 180)
EMBED = 4
LOCAL_BATCH = 2  # (batch x ensemble) folded, as the stepper passes it


class Passthrough(torch.nn.Module):
    """Returns the conditioning noise itself, so we can hash exactly what the
    SFNO body would have received.
    """

    def forward(self, x, context):
        return context.noise


def h(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()[:16]


def sample_hashes(model, device) -> list[str]:
    x = torch.zeros(LOCAL_BATCH, 3, *IMG, device=device)
    noise = model(x)
    return [h(noise[i]) for i in range(noise.shape[0])]


def main():
    dist = Distributed.get_instance()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    torch.cuda.set_device(device)

    # Exactly what fme/ace/train/train.py:100 does, with the same seed on
    # every rank because TrainConfig.seed is a single scalar in the yaml.
    set_seed(0)

    results = {}
    for noise_type in ("gaussian", "isotropic"):
        if noise_type == "isotropic":
            isht = dist.get_isht(IMG[0], IMG[1], grid="legendre-gauss").to(device)
            kw = dict(inverse_sht=isht, lmax=isht.lmax, mmax=isht.mmax)
        else:
            kw = dict(inverse_sht=None, lmax=0, mmax=0)
        model = NoiseConditionedModel(
            Passthrough(), img_shape=IMG, embed_dim_noise=EMBED, **kw
        ).to(device)
        results[noise_type] = {
            "draw1": sample_hashes(model, device),
            "draw2": sample_hashes(model, device),
        }

    gathered = [None] * world
    td.all_gather_object(gathered, {"rank": rank, "results": results})
    if rank == 0:
        print(json.dumps(gathered, indent=1))
        print("=" * 60)
        for nt in ("gaussian", "isotropic"):
            r0 = gathered[0]["results"][nt]["draw1"]
            within_unique = len(set(r0)) == len(r0)
            cross = [g["results"][nt]["draw1"] for g in gathered]
            all_ranks_identical = all(c == cross[0] for c in cross)
            advances = (
                gathered[0]["results"][nt]["draw2"]
                != gathered[0]["results"][nt]["draw1"]
            )
            n_unique_global = len({x for c in cross for x in c})
            print(
                f"{nt:>10}: distinct within rank={within_unique} "
                f"identical across ranks={all_ranks_identical} "
                f"draw advances={advances} "
                f"unique fields globally={n_unique_global}/{world * LOCAL_BATCH}"
            )
    td.barrier()


if __name__ == "__main__":
    with Distributed.context(handle_signals=False):
        main()
