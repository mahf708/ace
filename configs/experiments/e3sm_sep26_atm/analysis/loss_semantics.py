"""Numerical checks on what the ensemble losses actually score."""

import torch
from fme.core.ensemble import get_crps, get_energy_score

torch.manual_seed(0)


def hdr(s):
    print(f"\n{'=' * 68}\n{s}\n{'=' * 68}")


# ---------------------------------------------------------------- almost-fair
hdr("1. almost-fair CRPS epsilon: code uses (1-a)/2, AIFS defines (1-a)/M")


def analytic_acrps(gen, target, alpha):
    """ACRPS = a*fairCRPS + (1-a)*CRPS, expanded (AIFS-CRPS, arXiv:2412.15832)."""
    m = gen.shape[1]
    target_term = (gen - target).abs().mean(dim=1)
    s = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                s = s + (gen[:, i] - gen[:, j]).abs()
    fair = target_term - s / (2 * m * (m - 1))
    ordinary = target_term - s / (2 * m * m)
    return alpha * fair + (1 - alpha) * ordinary


for m in (2, 3, 4):
    gen = torch.randn(64, m, 8)
    tgt = torch.randn(64, 1, 8)
    for alpha in (1.0, 0.95):
        code = get_crps(gen, tgt, alpha=alpha).mean().item()
        want = analytic_acrps(gen, tgt, alpha).mean().item()
        flag = "OK " if abs(code - want) < 1e-6 else "MISMATCH"
        print(
            f"  M={m} alpha={alpha:<5}  code={code:.8f}  analytic={want:.8f}  "
            f"rel.err={abs(code - want) / abs(want):.2e}  {flag}"
        )

# ------------------------------------------------------------- energy score
hdr("2. Energy score: does it couple spectral modes / channels?")

B, E, C, L, M = 4, 2, 3, 6, 5
a = torch.randn(B, 1, C, L, M) + 1j * torch.randn(B, 1, C, L, M)
b = torch.randn(B, 1, C, L, M) + 1j * torch.randn(B, 1, C, L, M)
y = torch.randn(B, 1, C, L, M) + 1j * torch.randn(B, 1, C, L, M)

gen_ab = torch.cat([a, b], dim=1)  # member0=a, member1=b everywhere

# Same per-coefficient marginals {a,b}, different cross-coefficient pairing:
# swap which member holds which value on a checkerboard of (channel, mode).
mask = torch.zeros(1, 1, C, L, M, dtype=torch.bool)
mask[..., ::2, :] = True
mask[:, :, ::2, :, :] ^= True
m0 = torch.where(mask, b, a)
m1 = torch.where(mask, a, b)
gen_swapped = torch.cat([m0, m1], dim=1)

s_ab = get_energy_score(gen_ab, y)
s_sw = get_energy_score(gen_swapped, y)
print(f"  score(original)  sum = {s_ab.sum().item():.10f}")
print(f"  score(swapped)   sum = {s_sw.sum().item():.10f}")
print(f"  identical: {torch.allclose(s_ab.sum(), s_sw.sum(), atol=1e-6)}")
print(
    f"  per-coefficient tensors elementwise equal: {torch.allclose(s_ab, s_sw, atol=1e-6)}"
)


def joint_energy_score(gen, target):
    """The textbook multivariate ES: L2 norm over ALL non-batch dims."""
    d = (gen - target).flatten(2)
    term1 = d.abs().pow(2).sum(-1).sqrt().mean(dim=1)
    p = (gen[:, 0] - gen[:, 1]).flatten(1)
    term2 = 0.5 * p.abs().pow(2).sum(-1).sqrt()
    return (term1 - term2).mean()


j_ab = joint_energy_score(gen_ab, y).item()
j_sw = joint_energy_score(gen_swapped, y).item()
print(f"\n  For contrast, a TRUE joint energy score (L2 over all modes/channels):")
print(f"    joint(original) = {j_ab:.10f}")
print(f"    joint(swapped)  = {j_sw:.10f}")
print(f"    identical: {abs(j_ab - j_sw) < 1e-6}   (differs by {abs(j_ab - j_sw):.4f})")

# ------------------------------------------------------- CRPS at one member
hdr("3. At M=1 the CRPS pairwise term vanishes -> CRPS == MAE")
gen1 = torch.randn(32, 1, 10)
tgt1 = torch.randn(32, 1, 10)
c = get_crps(gen1, tgt1, alpha=1.0)
mae = (gen1 - tgt1).abs().mean(dim=1)
print(
    f"  max|CRPS - MAE| = {(c - mae).abs().max().item():.3e}  -> {torch.allclose(c, mae)}"
)
