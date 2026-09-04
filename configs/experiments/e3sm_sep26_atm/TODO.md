# TODO — sep26

Campaign is **built, checked, validated. Nothing queued.** Ordered by whether an
item blocks someone else. Six code claims were measured on 2026-09-03; the
results are in `PLAN.md` §11 and the scripts in `analysis/`.

---

## A. Blocking

### A0. LG04 is config-validated but NOT smoke-verified
`runs/LG04...yaml` passes `fme.ace.validate_config`, the checker and the tests,
but no forward+backward pass has run on `D0` + `G0` + `M2` + `Z0` — an
objective whose energy-score dispersion term is identically zero. Every other
new axis this campaign added was smoke-tested before it was trusted, and the
two upstream blockers were both found that way, not by validating. **Smoke it
before P2 drains.** (A hand-built smoke config failed to parse on 2026-09-03;
that was the harness, not the arm. Build the next one by overriding the
generated yaml, not by editing a neighbouring smoke config.)

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

**B1, B2 and B5 are written, tested and committed** on branches off `main` in
a worktree at `$PSCRATCH/ace-fixes`. Each was red-then-green: the test fails on
`main` and passes with the fix.

| branch | item | tests |
|---|---|---|
| `fix/almost-fair-crps-epsilon` | B5 | red at M3/M5, green 7/7 |
| `fix/energy-score-mode-weights-shape` | B1 | red 2, green 93/93 `test_loss.py` |
| `feature/energy-score-any-ensemble-size` | B2 | red at M1/M3/M5, green 9/9 |

**They do not unblock the campaign yet.** They sit on `main`; sep26 runs from
`e3sm/exps/hist-v2026.8.0`. Portability was **tested, not assumed** -- each
commit was cherry-picked onto the campaign base:

* each one **alone: clean**, `loss.py`'s divergence notwithstanding;
* **all three in sequence: B5 then B2 conflict in `fme/core/test_ensemble.py`**
  and nowhere else, because both append their tests to the end of the same
  file. Keep both blocks; the sources merge untouched.

Until someone ports them, `validate()`'s blockers stay in force and `G2`,
`M1`/`M3`-with-energy and `Y1`-away-from-`M2` remain refused. That is the
correct state: the campaign must not depend on unmerged branches.

### B1. `EnergyScoreLoss` mode_weights shape — DONE, on a branch
```
(*([1] * (x_hat.ndim - 1)), n_l, n_m)  ->  (*([1] * (es.ndim - 2)), n_l, n_m)
```
E01's total is **bit-identical** across it, so it cannot force a retrain.
Evidence: `analysis/verify_mode_weights_fix.py`. Unblocks `G2`.

### B2. Generalize `get_energy_score` past two members — DONE, on a branch
The trap was real and is now guarded: the old code pulled the 0.5 out of the
pairwise term because a mean over one pair leaves it alone, so a naive fix
changes the `M2` value and silently reinterprets RF01.
`test_energy_score_is_unchanged_at_two_members` pins the old expression at
**zero tolerance**. Averaging over unique pairs -- the normalisation `get_crps`
already used -- reproduces `M2` exactly and generalises.
Payoff once ported: the member sweep could anchor on RF01 directly (one factor)
instead of on `G1` (two factors).

### B3. PR the data-loader work to `main`
`time_buffer` exists only on `e3sm/exps/hist-v2026.8.0`, and it is worth 3.4× on
step time. Both campaigns rest on ~1,660 lines of experiment-branch code.

### B4. Data-parallel ranks all draw the same conditioning noise
`set_seed` gives every rank the same CUDA seed and training never attaches a
`RandomState` (`apply_config_seed` is inference-only). MEASURED byte-identical
across ranks; at global batch 16 over 16 ranks an `Mn` update carries **n**
unique noise fields, not 16n. Unbiased, but batch size buys no noise averaging.
**Fix tested** (`analysis/rank_noise_fix.py`): offset only the CUDA seed by
rank — noise decorrelates, init stays identical across ranks.
**Trap:** landing this silently would break comparability with RF01, which is
aug26 E01 trained under the current behaviour. Ship it behind a config flag,
default off, so sep26 can run an arm on each side of it.

### B5. `get_crps` epsilon is `(1-alpha)/2` — DONE, on a branch
Exact at `M2`, 0.89% out at `M3`, 1.16% at `M4` (MEASURED against the analytic
AIFS definition). One line. `validate()` refuses `Y1` away from `M2` until it
lands, so no run is currently wrong.

---

## C. Science, open

### C1. Seeds vs arms — DECIDED, and the reason got sharper
Seeds 2–3 are now on LG01–LG03 (+1,134 node-h), paid for by parking CU01, NC02
and OI03 at priority 6 (−967). The measurement that forced it is not the
discrimination ratio but the **pairing** one: a contrast that changes `Z`
carries a full seed's worth of noise because the init stream is reshuffled
(§11.2). Loss contrasts are paired and cheap; noise contrasts are not.

Still open: whether LG04 deserves seeds 2–3 (+644). It is the only arm asking
whether noise helps under a dispersion-rewarding loss, and it differences
against a 3-seed RF01, so a single seed is triage rather than an estimate.

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

### C5. Controls the parked arms would need
* **CU01** — a 60-epoch stochastic-from-scratch run and a deterministic 30+30
  restart with the same optimizer/EMA reset. Without them the contrast moves
  seven things at once.
* **NC02** — a fixed-architecture `noise_scale` knob upstream, so `Z` stops
  moving capacity and init together. That would also give a proper
  architecture-present/noise-off control and post-hoc amplitude calibration.
* **OI03** — `Q` defined on an area-weighted physical scale rather than array
  indices, so "three levels" names three lengths.

### C6. Read RO02 with its scale confound
Both scored steps are **summed**, not averaged, so RO02−RO01 raises the
objective scale as well as adding the 6 h horizon. Divide by the scored-step
count or add a scale-matched control before attributing anything to the extra
horizon.

---

## D. Not started

### D0. Log the noise-conditioning amplitude per epoch
Cheapest useful telemetry in the campaign. Every `Z1` run starts with the noise
pathway zeroed and has to grow it; an arm whose noise weights stay near zero has
quietly become a deterministic model and its CRPS is MAE. E01 grows to ±5.0%
(1σ, layer-norm scale) and saturates around epoch 11 (§11.4).
`analysis/noise_amplitude.py` reads it from a checkpoint directory today; it
belongs in the training loop as a scalar.

### D1. Evaluation harness
Nothing exists. **Both inline rollout blocks run one member per initial
condition**, so nothing currently measures calibration, spread, or any proper
finite-ensemble score — the campaign cannot yet answer its own main question.
This is a launch gate, not a nice-to-have. Needs `config-eval-ensemble.yaml`, `make_eval_config.py`,
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
