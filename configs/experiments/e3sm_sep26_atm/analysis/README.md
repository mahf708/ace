# analysis/

Each script answers one question and prints the number it answers it with.
Anything asserted in `make_campaign.py`'s comments as MEASURED was measured
by one of these.

## Ran 2026-09-03, written up in `PLAN.md` §11

| script | question | answer |
|---|---|---|
| `rank_noise.py` | do data-parallel ranks draw different conditioning noise? | **no** — byte-identical on every rank |
| `rank_noise_fix.py` | does offsetting only the CUDA seed by rank fix it? | **yes**, and model init stays identical across ranks |
| `seed_pairing.py` | is `S01` at `Z0` the same shared-core init as `S01` at `Z1`? | **no** — 5 of 22 shared tensors survive |
| `z0_degeneracy.py` | at `Z0`, are the two "members" distinct? | **no** — bit-identical; CRPS ≡ MAE exactly |
| `noise_amplitude.py` | has aug26 E01 learned to use its noise? | **yes** — 0 → 5.0% 1σ scale modulation, saturating ~epoch 11 |
| `loss_semantics.py` | what do the ensemble losses actually score? | energy score is per-coefficient marginal; almost-fair CRPS is exact only at `M2` |

## Running them

`rank_noise.py` and `rank_noise_fix.py` need ≥2 GPUs under `torchrun`:

```bash
FME_DISTRIBUTED_BACKEND=torch \
  .venv/bin/torchrun --nproc-per-node 2 analysis/rank_noise.py
```

`z0_degeneracy.py` must run on **CPU** — GPU kernel nondeterminism is ~2e-4,
which is the same size as the effect being measured:

```bash
FME_FORCE_CPU=1 .venv/bin/python analysis/z0_degeneracy.py
```

`noise_amplitude.py` takes a `training_checkpoints/` directory. The others
are single-process and need no arguments.

## Earlier

`card-sweep.sh`, `steprate.py` — the 40 GB vs 80 GB memory and step-rate
sweep behind the cost model. `epoch_stability.py`, `rollout_stability.py` —
the Tier 0 reads on how early an arm can be judged.
`verify_mode_weights_fix.py` — the one-line upstream fix for `G2`.
