# sep26v3 atmosphere ablation pilot — campaign overview

21 runs (7 configurations × 3 seeds) training `fme.ace` on the full continuous
E3SMv3 historical record (1940-1970 + 1980-1990). All runs: 30 epochs, batch
16, 4 nodes, seeds S01/S47/S82 for an error bar. Full generator inputs live in
`runs/MANIFEST.tsv`; this table is the config-level summary of what each run
actually changes and why.

Two axes hold the two baselines apart from each other; every other run then
moves exactly **one** axis off one of the two baselines, so each row's effect
is isolable against its named parent:

* **BL01** (stochastic pole): 2-member ensemble, `EnsembleLoss` (CRPS 0.9 +
  spectral energy score 0.1), sampled rollout (mostly 1 step, occasionally out
  to 20), 32-dim isotropic noise embedding.
* **BL02** (deterministic pole): 1 member, plain `MSE`, fixed 2-step rollout
  (both steps scored), no noise (`noise_embed_dim: 0`).

| Run (word) | Study / goal | vs. | `n_ensemble` | Loss | Rollout (`n_forward_steps`) | `optimize_last_step_only` | Noise (`noise_embed_dim`/`noise_type`) | Seeds | Priority | `rel` / hours |
|---|---|---|---|---|---|---|---|---|---|---|
| **BL01** `D0_G0_I0_M2_N0_Q0_R4_Y0_Z1` | Baseline, stochastic pole. | — | 2 | `EnsembleLoss`: crps 0.9 / energy 0.1 | sampled: {1:.6, 2:.2, 4:.1, 12:.05, 20:.05} | `true` | 32 / isotropic | 01/47/82 | 1 | 1.52 / 114h |
| **BL02** `D1_G0_I0_M1_N0_Q0_R2_Y0_Z0` | Baseline, deterministic pole. | — | 1 | `MSE` | fixed: 2 | `false` | 0 / gaussian | 01/47/82 | 1 | 0.90 / 74h |
| **LG02** `D0_G1_I0_M1_N0_Q0_R4_Y0_Z1` | Loss geometry: pure CRPS (MAE-like) at 1 member — noise is wired in but the M1 objective can't reward it, isolating whether the loss *shape* alone (vs. ensemble size) drives BL01's behavior. | BL01 | 1 | `EnsembleLoss`: crps 1.0 / energy 0.0 | sampled: {1:.6, 2:.2, 4:.1, 12:.05, 20:.05} | `true` | 32 / isotropic | 01/47/82 | 2 | 0.72 / 63h |
| **RO01** `D0_G0_I0_M2_N0_Q0_R2_Y0_Z1` | Rollout: BL01's stochastic model/objective at a **fixed 2-step** rollout instead of the sampled schedule — mimics ACE2-style training. Reads the sampled-vs-fixed choice at a matched depth. | BL01 | 2 | `EnsembleLoss`: crps 0.9 / energy 0.1 | fixed: 2 | `false` | 32 / isotropic | 01/47/82 | 2 | 1.89 / 137h |
| **RO02** `D1_G0_I0_M1_N0_Q0_R4_Y0_Z0` | Rollout: BL02's deterministic objective extended to BL01's **sampled 20-step** rollout instead of BL02's fixed 2 steps — the same rollout-depth axis RO01 reads on the stochastic pole, read here on the deterministic one. | BL02 | 1 | `MSE` | sampled: {1:.6, 2:.2, 4:.1, 12:.05, 20:.05} | `true` | 0 / gaussian | 01/47/82 | 2 | 0.72 / 63h |
| **EN02** `D0_G0_I0_M3_N0_Q0_R4_Y0_Z1` | Ensemble size: 3 members at BL01's full objective and rollout, isolating member count alone. | BL01 | 3 | `EnsembleLoss`: crps 0.9 / energy 0.1 | sampled: {1:.6, 2:.2, 4:.1, 12:.05, 20:.05} | `true` | 32 / isotropic | 01/47/82 | 3 | 2.18 / 156h |
| **NC01** `D0_G0_I0_M2_N0_Q0_R4_Y0_Z2` | Noise conditioning: 64-dim noise embedding against BL01's 32-dim default, isolating noise width alone. | BL01 | 2 | `EnsembleLoss`: crps 0.9 / energy 0.1 | sampled: {1:.6, 2:.2, 4:.1, 12:.05, 20:.05} | `true` | 64 / isotropic | 01/47/82 | 3 | 1.52 / 114h |

Reading the table by column gives the six single-axis contrasts the sweep is
built to support:

* **loss geometry** — LG02 − BL01 (energy-score weight → 0, ensemble → 1)
* **rollout schedule, stochastic pole** — RO01 − BL01 (sampled → fixed 2-step)
* **rollout schedule, deterministic pole** — RO02 − BL02 (fixed 2-step →
  sampled 20-step)
* **ensemble size** — EN02 − BL01 (2 → 3 members)
* **noise width** — NC01 − BL01 (32 → 64-dim embedding)
* **objective + rollout family, pole-to-pole** — BL01 vs. BL02 itself (not a
  single-axis contrast; it's the two-pole reference the other five are read
  against)

Every other field in `config-train-atm.template.yaml` (data paths,
normalization stats, architecture, epochs, batch size, logging/W&B config) is
identical across all 21 runs — confirmed by diffing each run's `.S01.yaml`
against BL01's.
