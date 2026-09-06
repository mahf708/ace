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
| median relative difference | **1.1e-02** (loaded, then a few steps of training) |
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
first batch -- now trains: 250 steps, loss 1.1952 -> 0.2952, zero "Per-channel
loss has" errors. Both first-batch blockers are lifted from the generator; see
`PLAN.md` 12.
```
(*([1] * (x_hat.ndim - 1)), n_l, n_m)  ->  (*([1] * (es.ndim - 2)), n_l, n_m)
```
E01's total is **bit-identical** across it, so it cannot force a retrain.
Evidence: `analysis/verify_mode_weights_fix.py`. Unblocks `G2`.

### B2. Generalize `get_energy_score` past two members — DONE, ported, VERIFIED IN SITU
`M3` with `energy_score_weight: 0.1` -- the exact call that used to raise
`NotImplementedError` on the first training batch -- now trains: 209 logged steps,
loss 4.0444 -> 0.3705, zero `NotImplementedError`. Unit tests pin the score
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
AIFS definition). One line, ported onto this branch; `validate()` no longer
restricts `Y1` to `M2`. No run in the list was wrong either way, since OI04 is `M2`.

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

### D1. Evaluation harness — BUILT
Both inline rollout blocks run one member per initial condition, so nothing in
training measures calibration, spread, or any proper finite-ensemble score.
That was the launch gate. `make_eval_config.py`, `sbatch-scripts/run-eval.sh`
and `sbatch-scripts/submit-eval.sh` now exist, with 17 tests in
`test_campaign.py`.

* **Two passes.** `scores` = members per IC, no trajectory files, `ensembles`
  aggregators at 6 h / 1 d / 5 d / 30 d / 90 d / 1 y. `traj` = one member,
  prediction files written. The generator refuses an ensemble on the
  trajectory pass: per-trajectory statistics must be computed inside a
  trajectory, and averaging four members costs 8–41% of the variance
  (`analysis/noise_decomp/results/ens4_mean_vs_member.txt`).
* **IC divisibility, twice.** `InferenceEvaluatorConfig.__post_init__` now
  calls `loader.validate_initial_conditions_divisible()`, so the bare
  `AssertionError` inside `InferenceDataset.__getitem__` cannot be reached
  from a config; the generator repeats the arithmetic where the node count is
  chosen and names the node counts that would work.
* **Ensemble scores now reach disk.** `InferenceEvaluatorAggregator.
  flush_diagnostics` wrote only the non-ensemble sub-aggregators, so CRPS, SSR
  bias and ensemble-mean RMSE existed only in W&B. Fixed with a test.
* **The file glob is narrowed to the reachable years.** The template's pattern
  matches all 1,501 monthly files because training reads them all; an
  evaluation reads only from its ICs forward. MEASURED on 2026-09-04, three
  jobs sharing the filesystem: the full glob had every rank in uninterruptible
  I/O wait past 17 minutes, while the narrowed one (120 files) opened in ~5 and
  was at window 5 of 73 by then.
* **16 ICs again.** The stall that forced 8 was DVS, not the shape: staged on
  Lustre, a 16-IC run on four nodes finished in 17.5 minutes with every GPU at
  93-100% and nothing in D state. Worth the full block because at 8 the skill
  metrics are stable to 0.2-3% but the calibration statistics are not — 8 to 16
  moves one-year `ssr_bias` by 0.08 on both Tat2m and PS.
* **The data is staged off DVS.** `sbatch-scripts/stage-data.sh` copies the
  decade to Lustre in 77 s at 3.3 GB/s, and `--data-root` points a config at
  it. MEASURED 2026-09-05, the same two concurrent 8-IC evaluations either
  way: 84 s per window against CFS, ranks parked in
  `dvsipc_wait_for_response` while the GPUs that had data sat at 100%, and
  13.5 s per window staged. A single run against CFS managed ~25 s, so the
  filesystem was the bottleneck and concurrency made it worse; staged, two at
  once each beat one run on CFS by a factor of two.
* **The scores pass stops at its last scored lead.** It shared a five-year
  default with the trajectory pass, so four fifths of it produced no ensemble
  metric — only a better-sampled climatology, which is pass 2's job.
* **`analysis/eval_table.py`** reads the output. `--seeds` gives the
  seed-to-seed floor, `--ladder` puts each noise override in units of it.

**COSTED, 2026-09-05.** Scores pass, 8 ICs x 4 members x 1 year on 2 nodes,
staged: ~19 min wall, ~0.63 node-hours. Sixteen arms x 3 seeds is ~30 node-h,
not the ~100 estimated. Pass 2 output is still uncapped (~0.5 TB at three
fields over five years).

### D1b. The seed floor constrains what the campaign can conclude -- MEASURED
2026-09-05, three RF01 seeds pinned to epoch 22, 8 ICs x 4 members x 1 year.
Coefficient of variation of ensemble-mean RMSE across the three seeds:

| | 1 d | 5 d | 30 d | 90 d | 1 y |
|---|---|---|---|---|---|
| Tat2m | 4.0% | 8.3% | 6.3% | **28.6%** | **23.2%** |
| TS | 3.2% | 7.2% | 8.0% | **34.3%** | **26.8%** |
| Qat2m | 3.6% | 7.7% | 7.2% | **23.4%** | **21.0%** |
| PS | 14.8% | 19.0% | 7.6% | 17.8% | 12.9% |
| FLUT | 2.0% | 3.6% | 3.8% | 7.1% | 8.4% |
| U_6 | 4.3% | 7.6% | 2.7% | 6.2% | 6.8% |
| precipitation | 1.7% | 4.4% | 3.6% | 1.2% | 1.8% |

An arm difference smaller than the entry is not a result at three seeds. The
weather range is workable everywhere. At climate leads the thermodynamic fields
are not: a 90-day temperature comparison needs an effect of about a third, and
no arm in the run list is expected to move it that far. Precipitation, outgoing
longwave and the winds hold to 8% and can be read at 90 d and 1 y.

Open question this raises: **is three seeds enough for the climate-range claims
the campaign wants to make on temperature?** Either the claims move to 90 d on
the fields that hold, or the seed count goes up on the arms that need it, or
the climate-range temperature reads are dropped. Decide before P2 is queued --
it is a scoping decision, not an analysis one. See
`analysis/rf01_scores/FINDINGS.md`.

### D2. Offline metrics
Return periods (GEV by L-moments; **do not quote a 50-year level** until
effective sample size is estimated), relative economic value, MJO.
Spread–skill and spectral tails need no new code.

Calibration is **DONE**: `rank_bias` and `rank_dispersion` are the rank
histogram's first two moments, computed per grid cell alongside `crps` and
`ssr_bias` in `fme/ace/aggregator/one_step/ensemble.py`. They are what `ssr_bias`
cannot see — an ensemble of the wrong shape at the right width passes a
spread-skill test and fails a rank test, and shape is what these arms differ in.
The reference variance is the discrete uniform's, `(1 - (M+1)^-2)/12`, which at
four members is 4% from the continuous 1/12 and so is not a rounding detail.
Still absent: reliability diagrams and coverage, which need the full histogram
rather than its moments.

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
