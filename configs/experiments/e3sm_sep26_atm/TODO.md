# TODO — sep26

State as of 2026-09-03. The campaign is **built, checked, validated, staged and
smoke-tested. Nothing is queued.** `PLAN.md` is the design argument, `README.md`
the reference, `AGENTS.md` the history; this file is the work that is left.

Ordered by whether it blocks someone else, not by size.

---

## A. Urgent, and not sep26's problem

### A1. aug26's E25 and E26 will crash at step 1 — drop or fix them

Both are queued at P6 for **~1,150 node-hours** and cannot train:
`get_energy_score` (`fme/core/ensemble.py:80`) supports exactly two members, and
E25 is `M1` while E26 is `M3`. Reproduced on a GPU node on both card types —
four runs, zero training steps between them. It dies *after* config validation,
dataset construction and model build, on the first batch.

**Action:** remove them from the aug26 submission, or land §B1 first. Whoever
owns the aug26 queue needs to know before the next window opens.

### A2. REF-D does not exist yet

aug26's E21 (the deterministic pole) is queued but has never run. **Five of the
nine sep26 arms difference against it**, so the campaign cannot be *read* until
it exists, even though it can be *run* without it. Confirm it is in the next
aug26 window before releasing sep26's P9 tier.

### A3. Per-channel loss plots for any `D0` run are misleading

Not a crash, so it is easy to miss. E01's *total* loss is exactly
`0.9 × CRPS + 0.1 × energy` — verified, gradients are fine — but the energy
term's contribution to the **per-channel** breakdown is a constant across all 50
channels. Per-channel plots therefore show `0.9 × CRPS_channel + constant`:
channel *ranking* survives, magnitudes and attribution do not. Fixed by §B1.
Until then, do not present per-channel loss comparisons from a `D0` run.

---

## B. Upstream `ai2cm/ace` PRs — each its own branch and review

### B1. `EnergyScoreLoss` mode_weights shape — one line, verified

```
(*([1] * (x_hat.ndim - 1)), n_l, n_m)  ->  (*([1] * (es.ndim - 2)), n_l, n_m)
```

`mode_weights` is sized against `x_hat`, which still has the ensemble dimension,
while `es` has already lost it — so the energy component carries two spurious
leading dims. Evidence and a ready-made check in
`analysis/verify_mode_weights_fix.py`:

| | before | after |
|---|---|---|
| E01 total (0.9/0.1) | 11.936110496520996 | **11.936110496520996** |
| E01 energy varies by channel | no | yes |
| pure-energy per-channel | `RuntimeError` | 5 channels |

**E01's total is bit-identical, so this cannot invalidate the running aug26
campaign or force a retrain** — which is what makes it easy to land. It *does*
change the per-channel diagnostic for every `D0` run (correctly); say so in the
PR so nobody is surprised when those plots move.

Unblocks sep26's `crps-energy` arm (L01), currently commented out of `RUNLIST`.

### B2. Generalize `get_energy_score` past two members

Bigger, and it needs care. Target term as the mean over members; internal term
as the mean over unique unordered pairs, following the
`torch.triu_indices(n_ens, n_ens, offset=1)` pattern `get_crps` already uses;
zero internal term at one member; a clear error at zero.

**The trap:** the current two-member code pulls the 0.5 out of the pairwise term
because a two-member mean over one pair makes it cancel. A naive generalization
**changes the M2 number and silently invalidates REF-S** and every comparison
against it. Pin the two-member value against the current implementation with a
regression test, plus forward-and-backward tests at one and three members.

Payoff: the member sweep could anchor on E01 directly (one factor from REF-S)
instead of on `crps-pure` (two factors), which is the stronger experiment.

### B3. PR the data-loader work to `main`

`time_buffer` does not exist on `main`, and `time_buffer: 10` with
`time_buffer_pool_size: 2` is what took the atmosphere from 3.155 to
0.925 s/batch. Both campaigns rest on ~1,660 lines of `fme/` code that lives
only on `e3sm/exps/hist-v2026.8.0`. Independent of everything else here.

---

## C. Decisions that gate further building

### C1. Run-id convention — built on the sparse delta, confirm or change

`sep26.atm.crps-pure_mem-1_noise-0.s01`. The alternative is a leading `A##`
field. See `PLAN.md` §3 for the argument and the objection. Changing it is cheap
*now* and expensive once anything has run.

### C2. Which arms to drop to fund seeds

Tier 0.1 measured discrimination at **1.8–2.7× the seed spread** at epochs 3–6,
so single-seed arms are marginal. Recommended trade: drop `roll-c2` (376 node-h,
and it depends on an aug26 E18 that may never run) and `alpha-095` (322 node-h,
most likely outcome a null), spend the ~700 node-hours on seeds 2 and 3 for the
four one-member mechanism cells. Same charge, one fewer axis, and the one block
that can support a claim.

This is a science call, not a budget one. It needs the design goals.

### C3. The decision rule needs changing

"Outside the parent's three-seed spread **at the same epoch**" reads a band that
was measured moving 1.00% → 3.18% between adjacent scored epochs. It will call
the same arm significant at epoch 9 and not at 12. Make it the spread pooled
over the last *k* scored epochs, or the max over them.

---

## D. Campaign work not started

### D1. The evaluation harness (`PLAN.md` §7)

Nothing exists yet. Needed:

* `config-eval-ensemble.yaml` — an `InferenceEvaluatorConfig` template with
  `checkpoint_path`, `experiment_dir` and `loader.dataset.data_path` as dotlist
  overrides, never baked in.
* `make_eval_config.py` — one config per (run id, epoch) from `MANIFEST.tsv`.
  **It must do the IC-divisibility arithmetic itself**: verified that
  `InferenceEvaluatorConfig.__post_init__` has no such check (only
  `InlineInferenceConfig` does), so a mismatch surfaces as a bare
  `AssertionError` inside the data loader, minutes into an allocation.
* `sbatch-scripts/run-eval.sh` and `submit-eval.sh`.
* Pass 1 (scores only, ~100 node-hours for sixteen arms) and pass 2
  (trajectories, cap at ~0.5 TB).

Use the offline evaluator, not inline inference — adding `n_ensemble_per_ic` to
the training configs would mean regenerating and re-running the references.

### D2. Offline metrics (`PLAN.md` §7)

Return periods (GEV by L-moments, and **do not quote a 50-year level** until the
effective sample size is estimated), relative economic value, calibration (rank
histograms, reliability, coverage — nothing in `fme` computes these), MJO.
Spread–skill and spectral tails need no new code.

### D3. Tier 0 reads still outstanding

Two of the six are done (epoch stability, and the degenerate-CRPS identity).
Remaining:

* **Compute-matching downward** — REF-S@epoch15 against REF-D@epoch30 is
  FLOP-matched and free. Blocked on A2.
* **Spectral tail metric** — a read of `power_spectrum_diagnostics.nc`, which is
  confirmed present per scored epoch. No new run.
* **One-step CRPS / SSR trace** — already logged every epoch, but to W&B rather
  than netCDF (checked: not in `val/epoch_NNNN/mean_diagnostics.nc`). Needs a
  W&B export, not a file read.
* **The seed and lagged ensembles** — three seeds on each pole plus sixteen
  staggered ICs already exist; exhaust them before writing a bred-vector harness.

---

## E. Housekeeping

* **Confluence.** The hackathon page is the source of truth for the run list and
  the factor alphabet. sep26 adds a campaign with a *different* convention, so
  it needs a page edit, not just a repo edit.
* **Allocation balance is unknown.** `iris` returned 403; check on a login node
  or the NERSC portal. The campaign is charge-bound rather than
  concurrency-bound (36 nodes out of 1408 `hbm40g`), so this is the binding
  number and nobody has it.
* **`REL_EXTRA` is empty and should stay that way** unless a new axis is
  measured to cost more than ~4%, which is this probe's floor.
* **`crps-half`, `noise-64`, `fdcrps-3`, `energy_score_whitening`** are recorded
  as out of scope. Note `fdcrps-3` was *measured* to cost nothing, so excluding
  it is a scientific choice, not a budgetary one.

---

## What is deliberately NOT a TODO

* Re-running aug26's E01 or E21. They are this campaign's references and
  re-running them is what the whole design avoids.
* Adding the offline metrics as `fme` aggregators. Separate upstream work.
* Hybrid mean + latent residual, bred vectors, a noise-off inference mode, the
  ocean, multi-realization training data. All recorded with reasons in
  `PLAN.md` §8.
