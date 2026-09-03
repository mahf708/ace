"""Verify the one-line fix for the EnergyScoreLoss mode_weights shape bug,
without touching the tree (jobs are reading it).

Current:  mode_weights shaped (*([1] * (x_hat.ndim - 1)), n_l, n_m)
          -> x_hat.ndim + 1 dims, but `es` has x_hat.ndim - 1 dims,
             so the product gains TWO spurious leading singleton dims.
Fix:      build it against `es`, i.e. (*([1] * (es.ndim - 2)), n_l, n_m).
"""
import torch
from fme.core.ensemble import get_energy_score
from fme.core.loss import CRPSLoss, EnsembleLoss, LossOutput, StandardLoss

B, E, C, LAT, LON = 2, 2, 5, 16, 32
names = [f"ch{i}" for i in range(C)]
torch.manual_seed(0)
gen, tgt = torch.randn(B, E, C, LAT, LON), torch.randn(B, 1, C, LAT, LON)
def sht(x): return torch.fft.rfft2(x)[..., : LAT // 2, : LON // 4 + 1]


def fixed_energy_forward(self, x, y):
    x_hat, y_hat = self.sht(x), self.sht(y)
    n_l, n_m = x_hat.shape[-2], x_hat.shape[-1]
    if self.scaling is None:
        self.scaling = 2 * (n_l * n_m) ** 0.5
        self.n_spectral = n_l * n_m
    es = get_energy_score(x_hat, y_hat)
    # THE FIX: size the broadcast against `es`, which has already lost the
    # ensemble dim, rather than against x_hat, which has not.
    mode_weights = 2 * torch.ones((*([1] * (es.ndim - 2)), n_l, n_m), device=es.device)
    mode_weights[..., 0] = 1
    es = es * mode_weights
    if self._whitening is not None:
        es = es * self._whitening.factor(y_hat)
    return [StandardLoss(es * (self.n_spectral / self.scaling))]


def run(patched: bool, cw: float, ew: float):
    import fme.core.loss as L
    orig = L.EnergyScoreLoss.forward
    if patched:
        L.EnergyScoreLoss.forward = fixed_energy_forward
    try:
        loss = EnsembleLoss(crps_weight=cw, energy_score_weight=ew, sht=sht)
        out = LossOutput(loss(gen, tgt), channel_names=names)
        total = float(out.total())
        try:
            pc = torch.stack([out.get_channel_losses()[n].loss for n in names])
            ch = pc
        except Exception as e:
            ch = f"{type(e).__name__}"
        return total, ch
    finally:
        L.EnergyScoreLoss.forward = orig


crps_only = run(False, 1.0, 0.0)
for label, (cw, ew) in {"E01 0.9/0.1": (0.9, 0.1), "pure energy 0.0/1.0": (0.0, 1.0)}.items():
    before_t, before_c = run(False, cw, ew)
    after_t, after_c = run(True, cw, ew)
    print(f"\n{label}")
    print(f"  total   before={before_t!r:>22}  after={after_t!r:>22}  "
          f"unchanged={isinstance(before_t, float) and abs(before_t - after_t) < 1e-4}")
    if isinstance(before_c, str):
        print(f"  per-ch  before={before_c}  after={'ok, %d channels' % len(after_c)}")
    else:
        e_before = before_c - cw * crps_only[1]
        e_after = after_c - cw * crps_only[1]
        print(f"  energy part varies across channels: "
              f"before={bool((e_before.std() > 1e-6).item())}  "
              f"after={bool((e_after.std() > 1e-6).item())}")
