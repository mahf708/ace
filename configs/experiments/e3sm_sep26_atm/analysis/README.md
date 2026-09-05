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

## Smoke-testing a new arm

`make_smoke_config.py` turns a generated run config into a cheap one that
still trains the same objective. Build DOWN from `runs/<runid>.yaml` -- never
sideways from a neighbouring smoke config, which is how you end up
smoke-testing something that is not the arm.

```bash
./analysis/make_smoke_config.py LG04 --years 3 --nodes 1
./analysis/run_smoke.sh $CAMPAIGN_SMOKE/lg04
```

**Use `run_smoke.sh`.** Every smoke failure in this campaign so far has been
the launcher rather than the arm, in three different ways, and each one looked
like a result. Documenting them was not enough -- the second trap was hit again
*after* being written down -- so the script does the right thing instead:
finds a node whose GPUs are actually idle, launches through `torchrun` with
absolute paths, defaults to a 45-minute deadline, and moves an existing passing
log aside rather than deleting it. It refuses when no node is free.

Two traps, both paid for once already:

* **Launch with `torchrun`, not bare `srun -n4`.** FME picks its device from
  `LOCAL_RANK`, which plain srun does not set, so all four ranks land on
  `cuda:0` and OOM. The OOM looks like the arm needing more memory; it is not.
  The giveaway is several processes listed against device 0.
* **Pin the node with `-w`, and check it is genuinely idle.** Two overlapping
  steps that each ask for all four GPUs on one node collide, and the loser dies
  with no traceback -- it just stops logging. A different node *name* is not
  enough: a previous smoke run may still hold it.
* **Give it at least 30 minutes.** Dataset setup alone is 10-15 min on the
  3-year subset. A shorter deadline kills the run before its first batch, which
  is indistinguishable from a hang.
* **Narrowing `subset` alone does not make it cheap.** The loader still opens
  every file the pattern matches and subsets afterwards, so setup stays at its
  full length. `make_smoke_config.py` narrows `file_pattern` to the same years
  -- 120 files instead of ~1,200.

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

To smoke a warm-start arm, make its parent produce a checkpoint first --
`--checkpoint-every` writes one after N batches rather than a whole epoch:

```bash
./analysis/make_smoke_config.py RF02 --checkpoint-every 5 -o $SMOKE/rf02-parent
# ...run it, then:
./analysis/make_smoke_config.py CU01 --warm-start $SMOKE/rf02-parent/training_checkpoints/ckpt.tar
```

`make_smoke_config.py` refuses an `I1` arm without `--warm-start` rather than
writing a config with the placeholder still in it.

`noise_amplitude.py` takes a `training_checkpoints/` directory. The others
are single-process and need no arguments.

## Earlier

`card-sweep.sh`, `steprate.py` — the 40 GB vs 80 GB memory and step-rate
sweep behind the cost model. `epoch_stability.py`, `rollout_stability.py` —
the Tier 0 reads on how early an arm can be judged.
`verify_mode_weights_fix.py` — the one-line upstream fix for `G2`.

## noise_decomp/ — what the conditioning noise does at inference

Ran 2026-09-04 on RF01's three seeds; written up in `noise_decomp/REVIEW.md`,
tables in `noise_decomp/results/`. Needs the `stepper_override.noise` knob on
this branch.

| script | question | answer |
|---|---|---|
| `one_step_drift.py` | is the noise-off backbone the model's mean? | **no** — averaging over the noise cuts one-step error 14–16%, and the drift is aligned with the error in 128 of 128 states |
| `make_eval.py`, `run_eval.sh` | the rollout ladder: off / fresh / fixed / half / double / mean / ensemble | 23 rollouts on held-out 2040s ICs |
| `summarize.py` | time-mean bias, spectra and pooled tails per mode | noise off inflates the 1-year time-mean bias 4–8× and drops 0.15–0.7 dex of small-scale power |
| `traj_stats.py` | statistics **inside** one trajectory, prediction and target alike | fresh noise puts variance, lag-1-day persistence and the 0.1/99.9% quantiles within 1–2 target σ |
| `yearly_drift.py` | does the first-year result hold to three years? | the variance and tail result does; the mean-state result does not |

The generalisable finding for the campaign: **`noise off` at inference is not a
deterministic control.** It is the stochastic model with its mean pathway
removed, and it is a worse operator than the model's own conditional mean. The
deterministic control is RF02.
