# TODO — sep26

Campaign is **built, checked, validated. Nothing queued.** Ordered by whether an
item blocks someone else.

---

## A. Blocking

### A1. RF02 has to run before anything can be read
Five arms difference against the deterministic pole. It is P1 and it is 567
node-hours (3 seeds × 189). Nothing in LG or RO03 means anything until it
finishes. **This is the schedule's first item.**

### A2. Per-channel loss plots from any `D0` run are wrong
Not a crash. E01's *total* loss is correct; the energy term's contribution to
the **per-channel** breakdown is a constant across all 50 channels. Ranking
survives, magnitudes and attribution do not. Fixed by B1. Until then, don't
present those comparisons — this affects aug26, which is running now.

---

## B. Upstream `ai2cm/ace` PRs — each its own branch

### B1. `EnergyScoreLoss` mode_weights shape — one line, verified
```
(*([1] * (x_hat.ndim - 1)), n_l, n_m)  ->  (*([1] * (es.ndim - 2)), n_l, n_m)
```
E01's total is **bit-identical** across it, so it cannot force a retrain.
Evidence: `analysis/verify_mode_weights_fix.py`. Unblocks `G2`.

### B2. Generalize `get_energy_score` past two members
**Trap:** the current code pulls the 0.5 out of the pairwise term because a
two-member mean over one pair makes it cancel. A naive fix changes the `M2`
number and silently invalidates RF01. Pin `M2` with a regression test first.
Payoff: the member sweep could anchor on RF01 directly (one factor) instead of
on `G1` (two factors).

### B3. PR the data-loader work to `main`
`time_buffer` exists only on `e3sm/exps/hist-v2026.8.0`, and it is worth 3.4× on
step time. Both campaigns rest on ~1,660 lines of experiment-branch code.

---

## C. Science, open

### C1. Seeds vs arms
Tier 0.1 measured discrimination at **1.8–2.7× the seed spread** at epochs 3–6,
so single-seed arms are marginal. Cheapest upgrade: seeds 2–3 on the LG block
(189 node-h each, 6 runs = 1,134 node-h) would give the mechanism 2×2 error
bars. Worth more than P5 (1,422 node-h) if the choice is forced.

### C2. The decision rule needs changing
"Outside the parent's three-seed spread **at the same epoch**" reads a band
measured moving 1.00% → 3.18% between adjacent scored epochs. Pool over the last
*k* scored epochs, or take the max.

### C3. RO04's setup cost is not in the model
Its 31-timestep windows make dataset setup slower than the 1-step arm's ~22 min,
and setup is paid on every requeue -- ~10 times over a 114 h run. `FIXED_HOURS`
is calibrated on the 1-step arm, so RO04 is probably ~117 h. Inside the cost
model's ~2% precision; a deeper rollout than 20 steps would not be.

### C4. Axes deliberately not run
`G2` (blocked, B1), `R3` (≤4 sampled — `R4` covers the question), `M2`/`M3`
under `G0` (blocked, B2), `energy_score_whitening` (untested knob, no level
defined). Adding any of these is a level, not an axis, so it renames nothing.

---

## D. Not started

### D1. Evaluation harness
Nothing exists. Needs `config-eval-ensemble.yaml`, `make_eval_config.py`,
`sbatch-scripts/run-eval.sh`, `submit-eval.sh`. **The generator must do the
IC-divisibility arithmetic itself** — `InferenceEvaluatorConfig` has no such
check (only `InlineInferenceConfig` does), so a mismatch surfaces as a bare
`AssertionError` minutes into an allocation. Pass 1 (scores, ~100 node-h for
sixteen arms), pass 2 (trajectories, cap ~0.5 TB).

### D2. Offline metrics
Return periods (GEV by L-moments; **do not quote a 50-year level** until
effective sample size is estimated), relative economic value, calibration (rank
histograms, reliability, coverage — nothing in `fme` computes these), MJO.
Spread–skill and spectral tails need no new code.

### D3. Tier 0 reads outstanding
Done: epoch stability, degenerate-CRPS identity. Left:
* compute-matching downward (RF01@ep15 vs RF02@ep30 is FLOP-matched, free) — blocked on A1
* spectral tail metric — a read of `power_spectrum_diagnostics.nc`, already on disk
* one-step CRPS/SSR trace — logged per epoch to W&B, not netCDF; needs an export
* the seed and lagged ensembles already on RF01 — exhaust before any bred-vector work

---

## E. Housekeeping

* **Confluence** is source of truth for the run list and factor alphabet. sep26
  uses a *different* convention and a *different* W&B project — needs a page.
* **Allocation balance unknown** (`iris` returned 403). The campaign is
  charge-bound, not concurrency-bound, so this is the binding number.
* **Reservation extension pending** (4–7 days requested). At 96 nodes × 7 days =
  16,128 node-hours, the full 19-run list is 36% of one window.

---

## Not a TODO

Re-running RF01 (it is aug26's E01, 3 seeds trained; ~970 node-hours to
reproduce). Adding the offline metrics as `fme` aggregators. Hybrid mean +
latent residual, bred vectors, noise-off inference, the ocean,
multi-realization training data — all in `PLAN.md` §8 with reasons.
