# TODO — sep26

Campaign is **built, checked, validated. Nothing queued.** Ordered by whether an
item blocks someone else. Six code claims were measured on 2026-09-03; the
results are in `PLAN.md` §11 and the scripts in `analysis/`.

---

## A. Blocking

### A0. LG04 smoke test — DONE
`D0` + `G0` + `M2` + `Z0` had never run: an EnsembleLoss whose energy-score
dispersion term is identically zero. It trains. On 3 years of data, batch 4,
one node: 34 logged steps, `batch_loss` descending 0.9056 -> 0.8313, no NaN and
no traceback. Reproduce with `analysis/make_smoke_config.py LG04`.

### A0b. CU01's warm start — VERIFIED
CU01 loads a `Z0` checkpoint (no noise convs) into a `Z1` architecture that has
32 tensors the checkpoint does not contain. It trains -- but so would a silent
no-op, so the weights were compared directly
(`analysis/compare_warm_start.py`):

| | |
|---|---|
| shared tensors | 103 of the parent's 104 |
| median relative difference | **1.1e-02** (loaded, then ~10 steps of training) |
| an independent draw | **1.41e+00** |
| tensors only in the child | **32** -- the noise pathway, keeping its own init |

Two orders of magnitude between "loaded" and "not loaded", so this is not a
judgement call. Every arm in the run list is now smoke-verified.

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

**B1, B2 and B5 are fixed, and now cherry-picked ONTO THIS BRANCH.** They also
sit on branches off `main`, pushed to the fork for upstream review:

| branch | item | tests |
|---|---|---|
| `fix/almost-fair-crps-epsilon` | B5 | red at M3/M5, green 7/7 |
| `fix/energy-score-mode-weights-shape` | B1 | red 2, green 93/93 `test_loss.py` |
| `feature/energy-score-any-ensemble-size` | B2 | red at M1/M3/M5, green 9/9 |

Porting them changes **no arm in the current run list**, which is why it was
safe to do under a campaign that differences against an inherited RF01:

* **B5** is inert here. The only `Y1` arm is OI04, at `M2`, where the old
  `(1-alpha)/2` and the new `(1-alpha)/M` are the same number. Everything else
  runs at alpha 1.0, where epsilon is 0 either way.
* **B2** is bit-identical at `M2`, pinned at zero tolerance, and no arm has
  `M != 2` with an energy weight.
* **B1** leaves the scalar total bit-identical -- re-verified at
  11.936110496520996 before and after -- and only repairs the per-channel
  breakdown.

The 86 loss/ensemble tests and 68 campaign tests pass on the branch with all
three applied.

### B1. `EnergyScoreLoss` mode_weights shape — DONE, ported, VERIFIED IN SITU
Pure energy (`G2`) -- the config whose per-channel shape used to raise on the
first batch -- now trains: 6 steps, loss 1.1952 -> 0.9041, zero "Per-channel
loss has" errors. Both first-batch blockers are lifted from the generator; see
`PLAN.md` 12.
```
(*([1] * (x_hat.ndim - 1)), n_l, n_m)  ->  (*([1] * (es.ndim - 2)), n_l, n_m)
```
E01's total is **bit-identical** across it, so it cannot force a retrain.
Evidence: `analysis/verify_mode_weights_fix.py`. Unblocks `G2`.

### B2. Generalize `get_energy_score` past two members — DONE, ported, VERIFIED IN SITU
`M3` with `energy_score_weight: 0.1` -- the exact call that used to raise
`NotImplementedError` on the first training batch -- now trains: 7 logged steps,
loss 4.0444 -> 1.8906, zero `NotImplementedError`. Unit tests pin the score
against the unbiased estimator at M = 1, 2, 3 and 5.

**Consequence for the run list, not yet taken.** EN02 is `D0_G1_M3` -- pure CRPS
at three members -- and it was put on `G1` *only* because `G0` at `M3` was
blocked. It could now be `G0` at `M3`, which differences against RF01 on one
factor (member count) instead of against EN01 on two. Strictly cleaner, and it
renames the run, so it is a design call.
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

### C4. Axes deliberately not run — mostly no longer blocked
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
