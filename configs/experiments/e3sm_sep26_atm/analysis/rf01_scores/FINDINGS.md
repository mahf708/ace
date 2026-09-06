# The first offline scoring of RF01

2026-09-05, one four-node interactive allocation. Fourteen scores-pass runs on
RF01's three trained seeds: the noise ladder at one seed, `keep` at three
seeds, and the same three seeds pinned to a common epoch. Regenerate any table
with `analysis/eval_table.py`.

Shape throughout: 8 held-out initial conditions from the 2040s, 4 members each,
1460 six-hourly steps (one year), scored at 6 h / 1 d / 5 d / 30 d / 90 d / 1 y.
Two nodes, ~19 minutes, 0.63 node-hours per run.

## 1. The seed floor, and what it forbids

**This is the number the campaign needs before it reads any arm difference.**
All three seeds from their own `best_ckpt.tar` -- the same kind of checkpoint,
which matters (§3) -- coefficient of variation of ensemble-mean RMSE across the
three, 8 initial conditions:

| variable | 1 d | 5 d | 30 d | 90 d | 1 y |
|---|---|---|---|---|---|
| Tat2m | 2.6% | 5.1% | 2.6% | **33.2%** | 13.1% |
| CRPS, Tat2m | 2.6% | 5.6% | 3.8% | **46.1%** | 18.1% |

Read this as: **an arm difference smaller than the entry is not a result at
three seeds.**

* **Weather range is workable.** At 1 d and 5 d the floor is a few percent, so
  a 15-20% arm effect is comfortably readable.
* **The climate range is not, for temperature.** A comparison of 90-day
  temperature drift between two arms needs an effect of about a third to clear
  three seeds, and the campaign does not expect effects that large.
* It is driven by one seed. At 90 d, Tat2m RMSE is S01 1.75 K, S03 2.16 K,
  S02 3.29 K -- S02 is not on a tail, it is 88% above S01. With three seeds the
  standard deviation is a crude summary; the range is the honest one.

**Caveat, and it is a real one.** These three checkpoints are at epochs 22, 24
and 28, because `best_ckpt.tar` is whatever last improved validation loss. The
comparison is clean in checkpoint *type* and confounded in *epoch*. §3 explains
why the obvious fix did not work, and what would.

**Per-variable numbers are deliberately not tabulated here.** The wider table
this section used to carry came from a set that mixed checkpoint types (§3), so
it is withdrawn rather than re-quoted from a smaller sample.

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

## 3. Epoch, seed, and a confound I introduced

`best_ckpt.tar` is rewritten whenever validation loss improves, so RF01's three
seeds sat at three different epochs when first scored -- S01 at 22, S03 at 24,
S02 at 28 -- and their 90-day RMSE was monotone in epoch (1.75, 2.16, 3.29 K)
while validation loss improved monotonically (0.0975, 0.0954, 0.0934). That
reads as "the checkpoint chosen on validation loss has the worst climate", and
it would matter, because it is how every arm in this campaign will be selected.

The obvious test is to re-score all three at a common epoch, and the per-epoch
`ckpt_NNNN.tar` files make that possible. I ran it: S01 from its `best_ckpt`
(already epoch 22), S02 and S03 from `ckpt_0022.tar`. The spread barely moved,
which I read as "seed, not epoch".

**That test does not hold.** `best_ckpt.tar` and `ckpt_0022.tar` at the *same
epoch* contain different weights -- compared directly on S01, where both are
epoch 22, `max|diff|` on a position embedding is 5.1e-3 against a mean magnitude
of 1.9e-2. The two files store different things (almost certainly averaged
against raw weights). So the "epoch-matched" set matched epoch and mismatched
checkpoint type, and its numbers cannot separate the two.

The signature is visible in the data once you look: going from 8 to 16 initial
conditions leaves S01 essentially unchanged (Tat2m 1 d RMSE 0.3922 -> 0.3915)
while S02 and S03 both degrade by 10% (0.418 -> 0.464, 0.422 -> 0.466). One
model generalises to the later initial conditions and two do not, which is a
statement about weights, not seeds.

**So: the epoch question is open, not settled.** The experiment that settles it
is one more run -- S01 from `ckpt_0022.tar`, 8 ICs, about 19 minutes -- which
makes all three the same checkpoint type at the same epoch. Until then, quote
the §1 floor, which at least uses one checkpoint type throughout.

What survives regardless:

* Everything in §2 and §4 is a **within-seed, same-checkpoint** comparison --
  one set of weights driven several ways -- so none of it is touched by this.
* Pin the weights before comparing anything, and pin the *same kind* of
  weights. `eval.env` records path, size and mtime; size alone would have
  caught this one, since the two files are 1.8 GB and 7.3 GB.
* At 8 initial conditions the annual numbers are unstable in both seed and
  epoch, so read 90 days.

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

0. **Re-score S01 from `ckpt_0022.tar`** and settle §3. One run, 19 minutes.
1. **Read arm differences against §1.** In particular, do not plan to read
   90-day or annual temperature drift between arms at three seeds.
2. Score at 90 d rather than 1 y when a stable number is wanted at 8 ICs.
3. Pin checkpoints before any cross-arm comparison.
4. Score arms at `keep`. The amplitude ladder is a separate axis, and mixing it
   into an arm comparison confounds two changes.
5. RF02 remains the only control that can answer "does stochastic training beat
   deterministic"; nothing in this file substitutes for it.
