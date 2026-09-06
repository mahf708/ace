# The first offline scoring of RF01

2026-09-05, one four-node interactive allocation. Fourteen scores-pass runs on
RF01's three trained seeds: the noise ladder at one seed, `keep` at three
seeds, and the same three seeds pinned to a common epoch. Regenerate any table
with `analysis/eval_table.py`.

Shape throughout: 8 held-out initial conditions from the 2040s, 4 members each,
1460 six-hourly steps (one year), scored at 6 h / 1 d / 5 d / 30 d / 90 d / 1 y.
Two nodes, ~19 minutes, 0.63 node-hours per run.

## 1. The seed floor, and what it permits

**This is the number the campaign needs before it reads any arm difference.**
Three seeds, **averaged (EMA) weights, all at epoch 22**, 8 initial conditions.
Coefficient of variation of ensemble-mean RMSE across the three:

| variable | 1 d | 5 d | 30 d | 90 d | 1 y |
|---|---|---|---|---|---|
| Tat2m | 0.4% | 3.5% | 1.0% | **14.5%** | 2.2% |
| TS | 0.8% | 2.7% | 1.5% | 17.4% | 8.5% |
| Qat2m | 0.7% | 2.6% | 1.1% | 11.2% | 2.3% |
| T_7 | 0.4% | 4.4% | 0.8% | 11.8% | 4.4% |
| PS | 1.1% | 6.9% | 2.5% | 9.1% | 3.3% |
| FLUT | 0.5% | 0.8% | 0.0% | 2.2% | 5.4% |
| U_6 | 0.7% | 3.4% | 2.2% | 0.8% | 4.2% |
| surface_precipitation_rate | 0.3% | 1.8% | 0.2% | 1.7% | 3.4% |

(Tat2m CRPS: 0.7 / 4.2 / 1.1 / 20.0 / 5.8%.)

Read this as: **an arm difference smaller than the entry is not a result at
three seeds.** Out to 30 days the floor is under 7% on everything, and even at
90 days -- the worst lead -- it is 15-17% on the thermodynamic fields and under
10% elsewhere. **Climate-range comparisons are within reach**, which is the
opposite of what the first attempt at this number said.

### Why the first attempt said 33%

Two controls are needed and neither alone is enough. The comparison must hold
the **epoch** fixed, and it must hold the **weight kind** fixed -- `best_ckpt.tar`
carries averaged weights, `ckpt_NNNN.tar` raw ones, and the evaluator loads
whichever is in `stepper` (§3).

| set | weights | epochs | 90 d CV | S01 | S02 | S03 |
|---|---|---|---|---|---|---|
| A `best_ckpt` | ema | 22/24/28 | 33.2% | 1.750 | 3.286 | 2.158 |
| B `ckpt_0022` | raw | 22 | 33.4% | 4.130 | 3.005 | 2.079 |
| **D folded** | **ema** | **22** | **14.5%** | 1.750 | 2.341 | 2.152 |

Set A is inflated by the epoch spread, set B by raw weights being both worse
and more variable -- S01 alone goes from 1.750 K to 4.130 K between averaged and
raw weights at the same epoch, which is larger than any seed or epoch effect
here. Because A and B happen to land on nearly the same number, comparing them
looked like evidence that the epoch did not matter. It was two different
problems reading alike.

**So: score every arm at a fixed epoch with averaged weights.** Using
`best_ckpt.tar` across arms does not do that, because its epoch is whatever last
improved validation loss and therefore differs per arm; that alone doubles the
floor. `analysis/ema_checkpoint.py` produces averaged weights at any epoch.

## 2. The noise ladder

RF01.S01, one checkpoint, four ways of driving the same weights. CRPS on Tat2m:

| mode | 6 h | 1 d | 5 d | 30 d | 90 d | 1 y |
|---|---|---|---|---|---|---|
| `keep` (trained) | **0.1553** | 0.2646 | **0.5725** | **1.0029** | **1.0878** | 2.3718 |
| `fixed` (one latent, held) | 0.1553 | **0.2623** | 0.5897 | 1.4687 | 2.5046 | 2.8998 |
| `half` (amplitude x0.5) | 0.1825 | 0.3456 | 0.8492 | 1.1183 | 2.3448 | 2.8115 |
| `double` (amplitude x2) | 0.2166 | 0.3982 | 0.9271 | 1.5095 | 1.4231 | **1.4812** |
| `off` (silenced) | 0.2428 | 0.4805 | 1.3633 | 1.5756 | 2.7121 | 3.0988 |

* **The trained amplitude is not a calibration problem to fix downward.**
  Halving it makes the ensemble markedly under-dispersed on both statistics
  (`ssr_bias` -0.091 to -0.370, `rank_dispersion` +0.070 to +0.524 at 1 d) and
  costs 31% of CRPS.
* **Nor upward at weather leads -- and this is the case that separates the two
  calibration statistics.** Doubling the amplitude puts `ssr_bias` at -0.0002 at
  one day, which is nominally perfect calibration, while making ensemble-mean
  RMSE **64% worse** (0.392 -> 0.642 K) and CRPS 50% worse. The same pattern
  holds on PS (ssr_bias -0.063 -> -0.097, RMSE 49.3 -> 117.7) and FLUT
  (-0.046 -> +0.155, RMSE 10.0 -> 18.4), and on all three seeds -- `ssr_bias`
  goes to -0.0002 / -0.0078 / -0.0051 while RMSE worsens by 63.6% / 59.9% /
  57.3%. Spread-skill is a ratio and is satisfied by inflating the spread and
  the error together; `rank_dispersion` reads -0.28 (over-dispersed) and ranks
  the five modes the way CRPS does. **Tuning a stochastic model to `ssr_bias`
  = 0 would have chosen this.** That is the argument for the rank statistics,
  demonstrated rather than asserted.
* **The best amplitude depends on the lead, and the crossover is between 30 and
  90 days.** Checked on all three seeds. `double` against `keep`, percent change
  in ensemble-mean RMSE for Tat2m (negative is better):

  | seed | 6 h | 1 d | 5 d | 30 d | 90 d | 1 y |
  |---|---|---|---|---|---|---|
  | S01 | +55.8% | +63.6% | +56.6% | +35.0% | +17.1% | **-33.3%** |
  | S02 | +55.3% | +59.9% | +63.5% | +29.5% | -36.8% | **-35.8%** |
  | S03 | +48.7% | +57.3% | +51.1% | +8.0% | -14.8% | **-42.3%** |

  Consistently much worse in the weather range, consistently much better at one
  year, on every seed; CRPS says the same. The 90-day column is where it flips
  and is the one that disagrees between seeds -- which is also where the seed
  floor is widest, so that is expected rather than surprising.

  **What improves is the ensemble, not the trajectory.** The year-mean bias of
  the trajectories themselves goes the other way, and `keep` wins it:

  | RF01.S01, year-mean | Tat2m bias RMS | PS bias RMS |
  |---|---|---|
  | `keep` | **1.336 K** | **288 Pa** |
  | `double` | 1.556 K | 370 Pa |
  | `half` | 2.174 K | 492 Pa |
  | `off` | 2.340 K | 542 Pa |

  So doubling widens an ensemble that is displaced and under-dispersed at one
  year, which improves CRPS and the ensemble mean, while each individual
  trajectory drifts slightly further. That reconciles this with
  `analysis/noise_decomp/`, which measured single trajectories and found
  `double` worse at one year: both are right about different quantities, and
  the disagreement was mine for not naming which.

  **Which one matters depends on the claim.** This campaign is building an
  emulator, so single-trajectory climate is usually the target, and there
  `keep` is the best amplitude at every lead. For a one-year *ensemble
  forecast*, a larger amplitude is better. Two consequences either way.
  **Score every arm at `keep`**, so comparisons are not confounded by an
  amplitude choice. And **no single amplitude is "calibrated"** -- the number
  that calibrates one lead, or one quantity, decalibrates another.
* **The noise has to be refreshed, but not immediately.** `fixed` reuses one
  latent field for the whole trajectory. At 6 h it is identical to `keep` by
  construction (same first draw) and at 1 d it is a near-tie -- `keep` wins on
  32 of 58 variables, `fixed` on 26. By 5 d `keep` wins 55 of 58, and at 90 d
  `fixed` costs 130% of CRPS. The refresh matters from about five days, not
  from step one.
* **`off` is not a deterministic control**, and now there is an ensemble
  measurement of why: the four members become identical, `ssr_bias` pins at
  -0.977 (the -1 floor, diluted by prescribed cells) and `rank_dispersion`
  reaches +0.970 against a theoretical ceiling of +1.0 for a symmetric
  collapse. The deterministic control is RF02.

## 3. The two checkpoint kinds, and the epoch effect they were hiding

`best_ckpt.tar` and `ckpt_NNNN.tar` do not hold the same weights at the same
epoch. Measured on RF01.S01 at epoch 22, where both files exist: all 135 tensors
of `ckpt_0022.tar`'s `ema.ema_params` are bit-identical to `best_ckpt.tar`'s
`stepper`, and differ from `ckpt_0022.tar`'s own `stepper` by up to 1.2e-2.
`best_ckpt.tar` has no `ema_params` key at all.

So `best_ckpt.tar` has folded the running average into the weights the evaluator
loads, and `ckpt_NNNN.tar` keeps raw weights there with the average beside them.
`analysis/checkpoint_epoch.py` reports which is which and warns when a set mixes
them; `analysis/ema_checkpoint.py` folds one into the other, verified at
135/135 tensors against the epoch where both exist.

With that controlled, the epoch effect is visible and large. **S02 at epoch 22
scores 2.341 K at 90 days; at epoch 28 it scores 3.286 K, 40% worse** -- same
seed, same weight kind, only the epoch differing -- while its validation loss
*improved* from 0.0973 to 0.0934. Epoch 28 is what `best_ckpt.tar` selected.

That is the checkpoint-selection question (TODO C2) with evidence behind it for
the first time. A systematic sweep is running: S01 at epochs 10/14/18/22/23 and
S02 across 22/24/26/28, all averaged weights, one seed at a time, to see whether
the degradation is monotone in epoch or particular to S02.

Two working rules regardless:

* Pin the weights before comparing anything, and pin the *same kind*. `eval.env`
  records path, size and mtime; size alone separates the two kinds, at 1.8 GB
  against 7.3 GB.
* Everything in §2 and §4 is a **within-seed, same-checkpoint** comparison -- one
  set of weights driven several ways -- so none of it is touched by any of this.

## 4. The calibration metrics, checked against the old ones

`rank_bias` and `rank_dispersion` were added for this campaign. Two independent
checks that they measure what they claim:

* **Where the ensemble is unbiased they agree with spread-skill.** At 1 d and
  5 d, `|rank_bias| < 0.01` and both statistics say mildly under-dispersed:
  `ssr_bias` -0.07 to -0.09, `rank_dispersion` +0.07 to +0.08, reproducing
  across seeds to about 0.01. At 6 h they disagree in sign, both within 0.06 of
  zero.
* **Under a perturbation of known sign they move together.** Halving the noise
  drives `ssr_bias` down and `rank_dispersion` up at every lead out to 30 d;
  silencing it drives both to their extremes.

Where they part company is the useful part. At 90 d and 1 y the histogram
tilts -- `rank_bias` reaches +0.20 (S01) and +0.27 (S02), meaning the truth
sits above the ensemble far too often. The year-mean bias map confirms it
independently: a global cold bias of -0.88 K (S01) and -1.94 K (S02), the same
ordering. A bias that large inflates the skill in `ssr_bias`'s denominator, so
`ssr_bias` = -0.30 at one year would be read as "the ensemble is too narrow"
when the measured failure is that it is displaced. **At climate leads
`ssr_bias` alone is not a spread diagnosis.**

## 5. What this changes

1. **Score every arm at a fixed epoch with averaged weights**, not at
   `best_ckpt.tar`, whose epoch varies per arm and doubles the seed floor.
2. **Read arm differences against §1.** Climate-range comparisons are workable;
   90-day temperature needs an effect above ~15%.
2. Score at 90 d rather than 1 y when a stable number is wanted at 8 ICs.
3. Pin checkpoints, and check `analysis/checkpoint_epoch.py` says one weight
   kind across the set.
4. Score arms at `keep`. The amplitude ladder is a separate axis, and mixing it
   into an arm comparison confounds two changes.
5. RF02 remains the only control that can answer "does stochastic training beat
   deterministic"; nothing in this file substitutes for it.
