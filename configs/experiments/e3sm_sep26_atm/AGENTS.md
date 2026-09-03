# Working log — sep26 atmosphere ablation

Chronological record of what was done, what was verified rather than inferred,
and what is still open. `README.md` is the reference a colleague reads;
`PLAN.md` is the design argument; this file is the history, kept so decisions do
not have to be rediscovered.

## Guidance for agents working in this directory

* **`config-train-atm.template.yaml` is a template, not a run.** Unlike aug26's
  `config-train-atm.yaml` — which *is* E01 — nothing here is both. Edit the
  template and regenerate; the baseline arm is `runs/sep26.atm.base.s01.yaml`
  like any other.
* **`runs/` is entirely generated.** Do not hand-edit it. Regenerate with
  `sbatch-scripts/generate-campaign.sh`, which also runs the checker.
* **Regeneration must be a no-op** against a committed `runs/`: no username, no
  scratch path, no timestamp. `run-train.sh` refuses a dirty worktree, so if
  regenerating dirtied the tree only the generator's author could launch.
* **`check_campaign.py` duplicates `make_campaign.py`'s level tables on
  purpose.** Keep duplicating them. A checker that imports the generator's
  mapping can only prove the generator is self-consistent.
* **A config that parses is not a config that runs.** Two arms in this campaign
  passed `fme.ace.validate_config` and raise on their first training batch. Any
  new axis gets a real forward *and backward* pass before it is queued.
* **Do not put filesystem paths that are inputs under a personal `$PSCRATCH`.**
  Use the shared CFS location. `check_campaign.py` fails a generated file
  containing `/pscratch/`.
* Judge a run by `REAL_EXIT=0` and `DONE ---- rank 0`, never by the log tail —
  successful runs print alarming teardown tracebacks.
* **Never `git checkout` a tracked file here.**
* The aug26 campaign's E01 and E21 are this campaign's references. They are
  live runs in someone else's directory; nothing here may modify them.

## 2026-09-03 — the campaign opened, and two arms that cannot train

Started from a plan that would have added E29–E39 to aug26. Moved it to its own
campaign instead: aug26's run id is a positional word, so widening it renames 35
live W&B runs and scratch directories, and the plan had already worked around
that with a third factor word and a fourth, parsed by first letter, spending 20
of 26 letters. Greenfield, that is all avoidable — see `PLAN.md` §1 and §3.

### What was measured rather than estimated

**40 GB cards work, and the campaign is sized for them.** Measured on
`nid001185` (A100-SXM4-40GB) against `nid008649` (80GB), the two sweeps run
concurrently so the ratio is clean under shared CFS load:

| arm | peak MiB | 40 GB s/batch | 80 GB s/batch | ratio |
|---|---|---|---|---|
| 1 member, 1 step | 15,383 | 0.433 | 0.390 | 1.110 |
| 2 members, 1 step | 21,155 | 0.903 | 0.826 | 1.094 |
| 3 members, 1 step | 24,651 | 1.304 | 1.177 | 1.108 |
| 2 members, 2 steps both scored | 21,921 | 1.713 | 1.551 | 1.105 |

9.4–11% slower, stable across every variant, for 5.5x the node pool (1408
`hbm40g` against 256 `hbm80g` in `gpu_ss11`) and no reservation. Peak memory
agrees between the cards to within 4 MiB, which is the expected result and a
check that nothing about the probe was card-specific.

**The memory model was wrong about rollouts and has been corrected.** A
both-scored two-step rollout was predicted at 28.7 GB and measures 21.9 —
barely above the one-step number — because `use_gradient_accumulation: true`
runs each scored step's backward before the next forward, so the two steps'
activations are never held at once. Memory scales with **members only**; rollout
length costs time. The sampled-rollout arms are therefore free on memory however
long they get, and the worst arm in the campaign is the three-member one at 40%
headroom rather than the rollout one.

**The `rel` cost model is confirmed to 4–6%:** measured 0.480 against 0.50 for
one member, 1.444 against 1.50 for three, 1.897 against 2.00 for the both-scored
rollout.

**The step-time measurement nearly went wrong.** The first pass gave a 1.72x
card ratio — impossible for two parts differing only in memory bandwidth, which
is what flagged it. One 3.73 s/batch interval in the 40 GB run against a steady
0.87–0.93 elsewhere: the `time_buffer` window refill that aug26 documents as
bimodal rather than noisy. Over a 70-batch probe exactly one such stall lands in
one run and not the other. `analysis/steprate.py` reports the median of the
per-window rates with the max beside it, so the stall stays visible.

### Two arms cannot train, and both passed `validate_config`

**1. `mem-1` and `mem-3` under any energy-score weight.** `get_energy_score`
(`fme/core/ensemble.py:80`) opens with `if gen.shape[1] != 2: raise
NotImplementedError`; `EnergyScoreLoss.forward` calls it unconditionally and
`EnsembleLoss.forward` calls that whenever `energy_score_weight > 0`. Verified
twice: by direct call on CPU at 1, 2 and 3 members, and then by running E01's
config with `n_ensemble: 1` and with `n_ensemble: 3` on 4 GPUs on both card
types — four runs, zero training steps between them. It dies *after* config
validation, *after* dataset construction, *after* the model is built and its
456,223,488 parameters are logged, on the first batch.

**This is aug26's E25 and E26**, queued at P6 for ~1,150 node-hours.
`check_campaign.py` there passes them because it verifies a config agrees with
its id and nothing more; the test suite misses them because every
`EnergyScoreLoss` test uses two members and the one three-member test goes
through `MSELoss`.

Workaround in use here: the member sweep runs on `crps-pure`
(`energy_score_weight: 0.0`), which makes `EnsembleLoss.forward` skip the energy
score entirely. Confirmed by smoke test at 1 and 3 members. The cost is that
those arms are two factors from REF-S rather than one.

**2. `crps-energy`, the pure spectral-space objective.** `EnergyScoreLoss`
builds `mode_weights` with `x_hat.ndim - 1` leading singleton dims, but
`get_energy_score` has already consumed the ensemble dim, so the energy
component comes out shaped `(1, 1, B, C, n_l, n_m)` instead of
`(B, C, n_l, n_m)`. Measured directly:

    E01-like 0.9/0.1     shapes=[(2, 5, 16, 32), (1, 1, 2, 5, 8, 9)]
    pure CRPS 1.0/0.0    shapes=[(2, 5, 16, 32)]
    pure energy 0.0/1.0  shapes=[(1, 1, 2, 5, 8, 9)]

With a CRPS component present the correctly-shaped one carries the channel
breakdown and nothing fails. At `crps_weight: 0` the energy score is alone and
`single_module.py:1757` raises `Per-channel loss has 1 elements but 50 channel
names were provided` on the first batch. Found by smoke-testing the generated
config; `validate_config` accepts it.

**Consequence for aug26, which is running now.** E01 is *not* broken: its total
is exactly `0.9 × 0.566 + 0.1 × 114.27 = 11.936`, checked, so the optimization
target and the gradient are right. But the energy term's contribution to the
*per-channel* breakdown is a **constant across channels** — measured as
`per_channel - 0.9 × crps_only_per_channel` having zero variance. Per-channel
loss plots for any `D0` run therefore show `0.9 × CRPS_channel + constant`, and
the spectral term contributes nothing channel-specific to them. Worth knowing
before that plot goes in a talk.

Both are upstream `ai2cm/ace` fixes, each its own PR. The member generalization
has a trap worth stating: the current two-member code pulls the 0.5 out of the
pairwise term because a two-member mean over one pair makes it cancel, so a
naive generalization changes the M2 number and silently invalidates REF-S.

### Tier 0: a free measurement that reversed a recommendation

The draft plan proposed running at 12–15 epochs on the argument that arm
ranking stabilises early. Read off aug26's existing per-epoch diagnostics, no
GPU time (`analysis/epoch_stability.py`, `analysis/rollout_stability.py`):

* On **one-step validation** the case is strong — REF-S's three-seed spread is
  under 1.5% at every epoch and the four-arm ordering is identical throughout
  (Spearman ρ = 1.000 against the deepest epoch).
* On the **held-out five-year rollout**, which is what the decision rule uses,
  the same arms reorder (ρ = 0.800 at epochs 3 and 6, 1.000 only at 9) and the
  seed spread does not narrow monotonically: 3.50%, 3.26%, 1.00%, 3.18%, 1.65%
  at epochs 3, 6, 9, 12, 15. Discrimination is **1.8–2.7x** the seed spread at
  epochs 3–6.

So the metric that says "run short" is the wrong metric. Three consequences:
keep 30 epochs; the decision rule must pool its spread over several scored
epochs rather than reading it at one, or it will call the same arm significant
at epoch 9 and not at 12; and buy seeds rather than arms.

E09 is excluded from that arm family — it zeroes `STW_0` from the loss, so its
rollout score is 1.26–1.58 against ~0.6 for the rest, and it would dominate any
range statistic.

### Smoke tests

Every runnable arm, from its generated config, real data, real forward and
backward on 4 GPUs: `crps-pure_mem-3`, `fdcrps-1`, `roll-c2`, `mem-1_obj-mse`
and `crps-pure_mem-1_noise-0` all reach logged training steps cleanly. The last
of those is the one that exercises `noise-0` forcing `noise_type: gaussian` —
isotropic at zero channels dies in the MKL FFT, reproduced in aug26.
`crps-energy` is the only failure, per above.

### Two operational notes, learned the hard way

* **`/tmp` is node-local on Perlmutter.** Three batch submissions died with exit
  127 nine seconds in because the script and config were staged there and the
  job landed on a different node. Stage to `$PSCRATCH`.
* **`log_train_every_n_batches` is a `TrainConfig` field, not a
  `LoggingConfig` one.** `--override logging.log_train_every_n_batches=N` is
  rejected by dacite at config parse, before any GPU work — which in a sweep
  harness that only greps for step lines reads as a silent no-op.
* **`uvx` hits the same flock/errno 524 problem as `pre-commit`.** Set
  `UV_TOOL_DIR` and `UV_CACHE_DIR` to node-local storage first.

## Open

* **REF-D (aug26 E21) does not exist yet.** Half the run list differences
  against it. This is the schedule's first question, not a detail.
* **aug26's E25 and E26 are still queued** and will crash at step 1.
* **`crps-energy` (L01) is commented out of `RUNLIST`**, not deleted. It is the
  sharpest test in the campaign for a model sold on spatial structure, and it
  fails for a reason unrelated to the science. Restore it when the upstream
  shape fix lands.
* The science design goals are still to come; the run list in `make_campaign.py`
  is a proposal, and `PLAN.md` §10 lists what is still the user's call.
