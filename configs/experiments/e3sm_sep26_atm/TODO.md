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

### A1. RF02 has to run — SCOREABLE 2026-09-07, still running to 30
Five arms difference against the deterministic pole. It is P1 and it is 567
node-hours (3 seeds × 189).

All three seeds passed **epoch 10, the C2 scoring epoch, on 2026-09-07** and
are in epoch 11 with `ema_ckpt_0010.tar` on disk. LG and RO03 are no longer
blocked on RF02 *finishing* — they are unblocked now, at the epoch anything
gets scored at. Epochs 11–30 remain useful for the trajectory (C7) and for the
FLOP-matched read in D3, not for scoring.

At ~2.08 h/epoch (measured: 6.9–7.2 ks for a plain epoch, 8.1–10.5 ks when the
5-year inference runs every third one), a seed reaches epoch 10 in ~21 h on 4
nodes — 84 node-hours, not 189. **An arm becomes scoreable in a third of the
node-hours its full run costs**, which is what makes anything at all possible
before the reservation ends 2026-09-09 15:00.

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
* **Training is on Lustre too, and it is worth more than the eval staging was.**
  MEASURED 2026-09-06 on RF02 -- three 4-node seeds of one arm, same code, same
  reservation, same period, differing only in filesystem. The `Step N:` interval
  is bimodal, and the compute floor is identical either way -- min 66-68 s per
  100 batches on both, across a node change, so the difference is I/O and not
  the GPUs. Lustre costs 4-7% in the typical step (p10 67-68 -> 70-72, median
  69 -> 72-74) and pays for it at the tail: p90 319-439 s on CFS against
  77-82 s. About 1 s of that shift is the three seeds sharing one copy --
  S03 alone read 71 s median / 77 s p90, and 72 / 81 once S01 and S02 joined.
  Striping is not the lever: the 600 training files already spread over 365
  OSTs, at most 8 files each. CFS ran 8-29 minute stalls
  at **one per 24.5 min**, 22 of them over 539 min of node time. Lost
  wall clock 39-54%; effective throughput 1900-2100 batches/h against ~4970 on
  scratch. At 8217 batches/epoch, 30 epochs is 111-124 h on CFS and 48 h on
  scratch -- the difference between fitting the `_CAP_aigs_hist` window and not.
  All three seeds moved to `FME_DATA_ROOT=/pscratch/sd/m/mahf708/v3.LR.historical_0101.aigo/run`
  at 11:24; loss curves are indistinguishable across the switch, and the file
  set is identical (1501 files, 1940-2065, matching sizes). CONFIRMED over the
  next 3.2 h: 448 intervals across the three seeds, **zero stalls**, max 85-90 s
  against a 72-74 s median -- a 1.2x tail where CFS ran to 25x. The prediction
  was falsifiable and held. Measured against each filesystem's own median-implied
  ceiling, CFS captured 39-44% of it and Lustre captures 94-97% -- we give up 4%
  of the best case to stop losing 60% of it.
* **The read tail is the mechanism, not the read cost.** Replaying the loader's
  own pattern (55 variables x 12 consecutive timesteps) on idle nodes at 1, 16
  and 64 concurrent readers: CFS median 4.6 s, p99 46 s, **max 530 s**; scratch
  median 2.0 s, p99 3.0 s, **max 4.0 s** -- 115x versus 2x. Both medians are
  flat in the number of readers. With `num_data_workers: 8` and
  `prefetch_factor: 4` a rank holds ~32 batches, about 22 s of cover at
  0.69 s/batch, and PyTorch's loader returns batches in worker-rotation order,
  so one worker past that window blocks the rank and the all-reduce blocks the
  other 15. Two caveats, both measured: the slow reads are **not** clustered by
  node (3.1-4.7% per node, uniform), and the probe reopens a file per read while
  the real loader amortizes opens, so its 4.6% exceedance rate is not a stall
  rate. `analysis/io_tail.py` reproduces it, and `analysis/stall_rate.py`
  counts the stalls.
* **Not every long interval is a stall.** The epoch boundary at each multiple of
  8217 runs validation and writes three checkpoints, costing 330-900 s on every
  seed on every filesystem. `analysis/steprate.py` and any stall count must
  exclude intervals spanning a boundary.
* **The scores pass stops at its last scored lead.** It shared a five-year
  default with the trajectory pass, so four fifths of it produced no ensemble
  metric — only a better-sampled climatology, which is pass 2's job.
* **`analysis/eval_table.py`** reads the output. `--seeds` gives the
  seed-to-seed floor, `--ladder` puts each noise override in units of it.

**COSTED, 2026-09-05.** Scores pass, 8 ICs x 4 members x 1 year on 2 nodes,
staged: ~19 min wall, ~0.63 node-hours. Sixteen arms x 3 seeds is ~30 node-h,
not the ~100 estimated. Pass 2 output is still uncapped (~0.5 TB at three
fields over five years).

### D1b. The seed floor, and how to keep it small -- MEASURED
2026-09-05, three RF01 seeds, **averaged (EMA) weights all at epoch 22**, 8 ICs
x 4 members x 1 year. CV of ensemble-mean RMSE across seeds, Tat2m: **0.4% at
1 d, 3.5% at 5 d, 1.0% at 30 d, 14.5% at 90 d, 2.2% at 1 y.** Everything except
the thermodynamic fields at 90 d is under 10% at every lead. Climate-range
comparisons are workable.

**Scoring at `best_ckpt.tar` doubles that floor to 33%**, because its epoch is
whatever last improved validation loss and so differs per arm. So: **score every
arm at a fixed epoch with averaged weights.** `analysis/ema_checkpoint.py`
produces them from any `ckpt_NNNN.tar`, verified 135/135 tensors against the
epoch where a real `best_ckpt` exists.

Two earlier readings of this number were wrong and are withdrawn: a set at mixed
epochs, and a set at one epoch but raw weights, both give ~33% for different
reasons and looked like corroboration. Both controls are needed.
See `analysis/rf01_scores/FINDINGS.md`.

### C2. The checkpoint decision rule -- DECIDED: score at epoch 10
Averaged weights, swept on two seeds: one-day error falls monotonically with
training while one-year error has a broad minimum and then climbs steeply, so
the two ranges want different checkpoints. `best_ckpt.tar` is selected on
validation loss, which tracks the improving end, and therefore lands close to
the worst checkpoint available for climate.

Tat2m ensemble-mean RMSE, each seed against its own best:

| epoch | S01 1 d | S01 1 y | S02 1 d | S02 1 y |
|---|---|---|---|---|
| 4 | +46.9% | +1.3% | | |
| 8 | +22.9% | **+0.0%** | | |
| **10** | +16.9% | **+0.1%** | +22.3% | **+0.0%** |
| 14 | +9.8% | +5.7% | +14.0% | **+31.2%** |
| 22 | +1.1% | +91.7% | +5.2% | +89.3% |
| 28 | | | +0.0% | +134.7% |

**Epoch 10, because the seeds agree there and disagree at 14.** S01's one-year
basin still holds at 14 (+5.7%); S02's does not (+31.2%). At 10 both are at
their minimum and within 0.6% of their best 90-day. It costs 17-22% of one-day
skill against the fully-trained checkpoint, against a 92-135% one-year penalty
at that checkpoint, with 30-day flat across the whole range.

`SCORING_EPOCH = 10` in `make_eval_config.py` is now the default, so an arm
scored without thinking is scored comparably; `--epoch 0` restores
`best_ckpt.tar` for a one-off look at a single arm.

**Carry this caveat with any weather-range number.** At epoch 10 the model is
short of its final one-day skill, so a weather comparison there partly measures
which objective *converges faster*. Every arm being equally early contains that
without removing it. For weather claims, score a second time at the converged
end and label which epoch a number came from -- both epochs for 16 arms at 3
seeds is ~60 node-hours against a 6,230 budget.

**A third witness, on all three seeds, for free.** Every run already logs
`Inference error:` for the 5-year rollout at epochs 3, 6, 9, ... The C2 sweep
above is offline, two seeds, 1 d and 1 y; this is in-training, three seeds,
5 y, and it agrees on the shape.

| epoch | S01 | S02 | S03 | mean | seed spread |
|---|---|---|---|---|---|
| 3 | 0.0587 | 0.0835 | 0.1088 | 0.0836 | 60% |
| 6 | 0.0547 | 0.0435 | 0.0518 | **0.0500** | **22%** |
| 9 | 0.0611 | 0.0641 | 0.0422 | 0.0558 | 39% |
| 12 | 0.0450 | 0.0856 | 0.1347 | 0.0884 | 102% |
| 15 | 0.0521 | 0.1444 | 0.2239 | 0.1401 | 123% |
| 18 | 0.1138 | 0.3561 | 0.2270 | 0.2323 | 104% |
| 21 | 0.3736 | 0.3894 | 0.3527 | 0.3719 | 10% |
| 24 | 0.5165 | 0.3065 | 0.2879 | 0.3703 | 62% |
| 27 | 0.5197 | 0.3341 | 0.3331 | 0.3956 | 47% |
| 30 | -- | 0.3229 | 0.3220 | 0.3225 | 0% |

Read the spread column with the mean beside it. Epoch 21 is tight at 10% not
because the seeds agree usefully but because all three have converged to the
same bad place; the only row that is both low and tight is 6.

Validation loss over the same 30 epochs falls monotonically on all three seeds
(0.2099 -> 0.0925, seeds within 0.5% of each other at every single epoch). The
two metrics are therefore anti-correlated from about epoch 9 on, and
`best_ckpt.tar` -- selected on validation loss -- is a 6-10x worse climate model
than the epoch-6 checkpoint sitting next to it on disk. That is the C2 claim,
now with the curve behind it rather than two endpoints.

Two things this adds. S03 was not in the sweep, and its knee falls at 12 with
S02's rather than at 15 with S01's, so epoch 10 survives a third seed and epoch
14 is unsafe on two of three. And the 5-year knee is earlier than the 1-year
one: epoch 12 is already +77% over the epoch-6 mean here, against +5.7% and
+31.2% at epoch 14 in the sweep. That is the direction a longer rollout should
move it, and it means **epoch 10 sits inside the basin but nearer its edge the
longer the rollout being claimed**. Label the rollout length on a climate
number the way C2 already asks for the epoch.

The deterministic pole reproduces the measurement problem, and now over three
inference epochs. RF02's 5-year error spreads 31%, 32% and 74% across seeds at
epochs 3, 6 and 9, while its validation loss spans 0.8% or less at every epoch
(0.11435-0.11525 at epoch 4, 0.07434-0.07486 at epoch 10) and falls
monotonically throughout. Both poles therefore show the same thing: the metric
that selects checkpoints cannot see what the climate metric sees.

RF02's knee is EARLIER than RF01's, which the fixed-epoch rule has to carry.
RF01 gains 40% from epoch 3 to 6 and is only +12% above its best by 9; RF02 is
flat from 3 to 6 (0.0737 -> 0.0739) and +62% by 9. Scoring both at epoch 10 is
therefore not neutral between them -- it is further past RF02's knee than
RF01's. Keep the fixed epoch anyway (scoring each arm at its own best
inference epoch is selection on the reported metric), but report the pole
difference at epoch 10 AND at each pole's own best, and say so. Here it
survives both -- see C7.

Regenerate either table with `analysis/inference_error_trajectory.py`.

A side benefit: epoch 10 is a third of a 30-epoch run, so an arm becomes
scoreable long before it finishes.

### C7. The pole gap is measured, and LG is exactly what decomposes it

RF02 has passed the scoring epoch on all three seeds (`ema_ckpt_0010.tar` on
disk for S01/S02/S03, 2026-09-07). Its in-training 5-year rollout error, beside
RF01's at the same epochs and under a **byte-identical inference block** -- same
7300 forward steps, same 16 ICs, same reference run, the only difference being
the CFS-vs-Lustre path to the same staged data:

```
epoch   RF01 (D0 M2 Z1, stochastic)   RF02 (D1 M1 Z0, deterministic)
        S01     S02     S03    mean   S01     S02     S03    mean   ratio
    3   0.0587  0.0835  0.1088 0.0836  0.0833  0.0773  0.0606 0.0737  0.88
    6   0.0547  0.0435  0.0518 0.0500  0.0862  0.0731  0.0625 0.0739  1.48
    9   0.0611  0.0641  0.0422 0.0558  0.1660  0.1153  0.0772 0.1195  2.14
```

At epoch 3 the two poles interleave -- no signal. At epochs 6 and 9 the three
RF01 seeds are **all** below the three RF02 seeds with no overlap (0.0547 <
0.0625; 0.0641 < 0.0772). Under the null, complete separation of 3 against 3 in
a stated direction has probability 1/C(6,3) = 0.05 exactly. The two epochs are
the same six runs, so that is one p = 0.05, not two.

The same holds seed-by-seed against each seed's own best epoch, which is the
comparison the differing knees cannot distort: RF01 {0.0450, 0.0435, 0.0422}
against RF02 {0.0833, 0.0731, 0.0606}. RF01's **worst** seed beats RF02's best
by 26%; the means differ by 1.5x.

Three caveats, none of which the data can retire yet:

* RF02 has three inference points against RF01's ten, so its best is the more
  undersampled of the two and could sit between the sampled epochs.
* `time_mean_norm/rmse/channel_mean` is the 5-year **time-mean** error. It is a
  climate-bias metric and says nothing about variability or extremes.
* D, M and Z co-vary between the poles (EnsembleLoss/2 members/32 noise dims
  against MSE/1 member/none). That is one design choice with three config
  consequences rather than three confounders -- but it does mean the gap is not
  yet attributable to any one of them.

Which is what **LG01, LG02 and LG03 are for**, and they are the only 3-seed
arms in the campaign besides RF02 -- the seed count this comparison needs:

```
RF01   D0 G0 M2 Z1   ensemble loss, 2 members, noise      pole
LG04   D0 G0 M2 Z0   same loss and members, NO noise      1 seed
LG02   D0 G1 M1 Z1   CRPS only, 1 member, noise           3 seeds
LG01   D0 G1 M1 Z0   CRPS only, 1 member, no noise        3 seeds
LG03   D1 G0 M1 Z1   MSE, 1 member, WITH noise            3 seeds
RF02   D1 G0 M1 Z0   MSE, 1 member, no noise              pole
```

LG03 minus RF02 isolates the noise input under MSE; LG01 minus RF02 isolates
the proper scoring rule with everything else held at the deterministic setting.
That is the decomposition of the gap above, and it is the argument for running
LG first when capacity frees.

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
* **Allocation is NOT the constraint** — the opposite of what this line said
  while `iris` was returning 403. Measured 2026-09-07 (`iris` on a login node,
  no `-c balance` subcommand; plain `iris` prints the table): e3sm_g has
  **366,757 node-hours left of 685,674 (53%)**, and mahf708's own sub-allocation
  90,858 of 100,000. The 26-arm campaign to epoch 10 is order 2,000 node-hours.
  The campaign is **queue-bound**, not charge-bound, which inverts every
  trade that was being made to save charge.
* **Reservation extension pending** (4–7 days requested). At 96 nodes × 7 days =
  16,128 node-hours, the full 19-run list is 36% of one window. This is now the
  single highest-value ask — see the queue numbers below.

### E1. Queue physics off-reservation — MEASURED 2026-09-07

`_CAP_aigs_hist` ends **2026-09-09 15:00**, and slurm stops *starting* 12 h jobs
in it 12 h before that, so the reservation's real deadline for new work is
**09-09 03:00**. The next wall after that is **Perlmutter maintenance 09-16
06:00–22:00** (NERSC outage calendar); slurm carries the usual conservative
placeholder for it, `maintenance_20260916`, 7 days wide over all 5,248 nodes, so
nothing that cannot finish by 09-16 06:00 will start. `sbatch --test-only` for a
fresh 4-node 12 h regular job answers "to start at 2026-09-18T22:50".

**Walltime, not QOS, is the lever, and it is a cliff at 2 h.** GPU jobs of 2–16
nodes in `gpu_regular`/`gpu_preempt`, submitted *and* started 09-04 → 09-07 (so
no stale submissions inflating the wait):

```
tlimit     n    med wait   p90
  <=1h   289      1.6 h    7.9 h
  1-2h    52      3.0 h   12.9 h
  2-3h    41     10.7 h   64.4 h
  3-4h    22     26.1 h   39.3 h
  4-6h    14     35.8 h   54.1 h
 6-12h    17     38.9 h   50.8 h
  >12h    25     40.0 h   46.5 h
```

Past 2 h the wait climbs to a ~40 h plateau and **12 h is no worse than 6 h and
no better than 48 h** — the choice inside that range is not a choice. Below 2 h,
backfill takes the job almost immediately.

Why us in particular: `sshare` gives mahf708 in e3sm_g **FairShare 0.0072**, with
EffectvUsage 0.206 against NormShares 0.0029 — roughly 70× over share, mostly
from this campaign. Priority scheduling will not favour us; **backfill is the
only realistic path in, and backfill only takes short jobs**.

`preempt` vs `regular`: preempt has a quarter the queue depth (1,192 pending
against 4,859) at the same slurm priority (67679), but that does **not** convert
into a shorter wait — over 09-05 → 09-07, median 50.4 h against regular's 35.1 h
for 2–16 node jobs, better only in the tail (p90 85.7 h against 106.7 h). Take
preempt for its **0.25× charge and its 2 h preemption guarantee**, not for speed.
Preemption itself is barely happening: of ~299 preempt jobs that terminated
since 09-05, **4 were PREEMPTED (1.3%)**.

Which gives the off-reservation recipe: **`--qos preempt --time 02:00:00`**. A
2 h request sits under the backfill cliff *and* at exactly the preemption
guarantee, so it is effectively unpreemptable. Cost of chaining 2 h slots,
measured rather than assumed: ~3 min to first training step after a restart, and
at most one 1,000-batch checkpoint interval (~14 min, verified from the
84000/85000/86000 timestamps) of redone work — call it 10 min, **~8% of a 2 h
slot**. Against a median 3.0 h wait that is a 37% duty cycle, versus 23% for
12 h slots. Reaching epoch 10 (21 h of compute) costs ~57 h of wall clock at 2 h
against ~100 h at 12 h.

Two things this does not settle:

* A **requeued** job keeps its original submit time, so its age priority is
  preserved and may restart far faster than a fresh submission. If so, long
  slots recover most of the gap. Nothing in our history measures this — every
  requeue we have ever done was inside a reservation, where the wait is ~0.1 min.
  Settle it by running one seed at 2 h and one at 12 h and comparing duty cycle.
* An inference epoch costs 17–57 min on top of the ~1.94 h training epoch
  (8178 s at epoch 9 and 8091 s at epoch 3 against 6916–7158 s for a plain one),
  and inference is **not** checkpointed internally. A 2 h slot still clears it,
  because a restart resumes near the epoch end and then has the whole slot for
  inference — but 2 h is the floor for this workload, not a value to shave.

Also set `FME_MAIL_TYPE=NONE` on short slots: the default includes
TIME_LIMIT_90, which on a 2 h job mails every 1.8 h per seed.

**Confirmed at submission** (LG01-LG03, 9 seeds, `--qos preempt --time
02:00:00`, 2026-09-07): `sprio` on job 58031749 returns PRIORITY 67679 with
**AGE 0, FAIRSHARE 0, PARTITION 0** -- the whole number is the QOS constant, so
every job in `gpu_preempt` carries exactly the same priority and ours beats 6%
of the 11,094 pending jobs. Priority scheduling gives us nothing whatsoever;
**backfill is the entire mechanism**, which is the argument for 2 h restated as
a measurement rather than a projection.

One correction to the cost model above, from reading the trap in
`sbatch-train-atm.sh`: a *walltime* requeue is graceful. USR1 at T-300 s ->
`scancel --signal=TERM` -> FME's handler tears down the collectives and writes
a restart checkpoint -> `scontrol requeue`, which **keeps the job id and the
original submit time**. So a walltime exit costs the 5 min signal lead plus
~3 min of startup, not a redone checkpoint interval -- ~7% of a 2 h slot. The
~14 min of redone work applies only to an ungraceful kill, i.e. a real
preemption, which is 1.3% of preempt jobs. It also means the requeue keeps
whatever age priority the job has accrued, so the open question above is
half-answered in the direction that favours chaining.

---

## Not a TODO

Re-running RF01 (it is aug26's E01, 3 seeds trained; ~970 node-hours to
reproduce). Adding the offline metrics as `fme` aggregators. Hybrid mean +
latent residual, bred vectors, noise-off inference, the ocean,
multi-realization training data — all in `PLAN.md` §8 with reasons.
