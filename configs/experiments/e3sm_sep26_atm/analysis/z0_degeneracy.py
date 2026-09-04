"""With no noise pathway (Z0), are the two 'ensemble members' distinct at all?

The review proposes two M2/Z0 controls. If a Z0 model is deterministic, both
members are the same field, the CRPS pairwise term is identically zero, and
the arm is either degenerate with its M1 twin or merely wasteful.
"""

import torch
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.ensemble import get_crps, get_energy_score
from fme.core.rand import set_seed

IMG = (90, 180)


def main():
    device = torch.device("cpu")
    for zdim in (0, 8):
        set_seed(0)
        model = (
            NoiseConditionedSFNOBuilder(
                embed_dim=32,
                noise_embed_dim=zdim,
                num_layers=2,
                noise_type="gaussian",
                pos_embed=False,
            )
            .build(4, 4, DatasetInfo(img_shape=IMG))
            .to(device)
            .eval()
        )
        # one sample broadcast to 2 ensemble members, exactly as the stepper does
        x = torch.randn(1, 4, *IMG, device=device).repeat(2, 1, 1, 1)
        with torch.no_grad():
            out = model(x)
        # are the noise->scale/bias convs zero-initialised?
        nz = [
            (k, float(v.abs().max()))
            for k, v in model.state_dict().items()
            if "W_bias" in k or "noise" in k.lower()
        ]
        if nz:
            print(
                f"      noise-conditioning tensors: {len(nz)}, "
                f"max|w| = {max(v for _, v in nz):.3e}"
            )
        m0, m1 = out[0], out[1]
        maxdiff = (m0 - m1).abs().max().item()
        gen = out.unsqueeze(0)  # (B=1, E=2, C, lat, lon)
        tgt = torch.randn(1, 1, 4, *IMG, device=device)
        crps = get_crps(gen, tgt).mean().item()
        mae = (gen - tgt).abs().mean().item()
        print(
            f"  Z(noise_embed_dim={zdim}): max|member0-member1| = {maxdiff:.3e}   "
            f"members identical = {maxdiff == 0.0}"
        )
        print(
            f"      CRPS(M=2) = {crps:.6f}   MAE = {mae:.6f}   equal = {abs(crps - mae) < 1e-7}"
        )
        gh = torch.fft.rfft2(gen)  # stand-in complex spectral coeffs
        yh = torch.fft.rfft2(tgt)
        es = get_energy_score(gh, yh)
        internal = 0.5 * (gh[:, 0] - gh[:, 1]).abs()
        print(
            f"      energy-score dispersion term max = {internal.max().item():.3e} "
            f"(zero => score collapses to a spectral L1 distance), ES mean={es.mean().item():.4f}"
        )


if __name__ == "__main__":
    with Distributed.context(handle_signals=False):
        main()
