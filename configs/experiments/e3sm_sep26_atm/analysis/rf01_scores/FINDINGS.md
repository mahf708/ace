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
All three seeds at **epoch 22** (see §3 for why that matters), coefficient of
variation of ensemble-mean RMSE across the three:

| variable | 1 d | 5 d | 30 d | 90 d | 1 y |
|---|---|---|---|---|---|
| Tat2m | 4.0% | 8.3% | 6.3% | **28.6%** | **23.2%** |
| TS | 3.2% | 7.2% | 8.0% | **34.3%** | **26.8%** |
| Qat2m | 3.6% | 7.7% | 7.2% | **23.4%** | **21.0%** |
| T_7 | 5.2% | 10.2% | 5.6% | **22.2%** | 16.5% |
| PS | 14.8% | 19.0% | 7.6% | 17.8% | 12.9% |
| U_6 | 4.3% | 7.6% | 2.7% | 6.2% | 6.8% |
| FLUT | 2.0% | 3.6% | 3.8% | 7.1% | 8.4% |
| surface_precipitation_rate | 1.7% | 4.4% | 3.6% | 1.2% | 1.8% |

Read this as: **an arm difference smaller than the entry is not a result at
three seeds.**

* **Weather range is workable.** At 1 d and 5 d the floor is 2-10% for
  everything except surface pressure, so a 15-20% arm effect is readable.
* **The climate range is readable for some fields and not others.**
  Precipitation, outgoing longwave and the winds hold to 1-8% at 90 d and 1 y.
  The thermodynamic fields do not: 22-34% at 90 d. A comparison of 90-day
  temperature drift between two arms needs an effect of about a third to clear
  three seeds, and the campaign does not expect effects that large.
* It is driven by one seed. At 90 d, Tat2m RMSE is S01 1.75 K, S03 2.08 K,
  S02 3.01 K -- S02 is not on a tail, it is 70% above the other two. With three
  seeds the standard deviation is a crude summary; the range is the honest one.

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
  (-0.046 -> +0.155, RMSE 10.0 -> 18.4). Spread-skill is a ratio and is
  satisfied by inflating the spread and the error together; `rank_dispersion`
  reads -0.28 (over-dispersed) and ranks the five modes the way CRPS does.
  **Tuning a stochastic model to `ssr_bias` = 0 would have chosen this.** That
  is the argument for the rank statistics, demonstrated rather than asserted.
* **At one year, doubling helps, on everything.** CRPS 2.37 -> 1.48, RMSE
  3.20 -> 2.14, both calibration statistics closer to zero, and likewise for PS
  and FLUT. At ninety days it moves the other way. Recorded as observed: eight
  initial conditions make the annual numbers the least stable here, and one
  non-monotonic pair is not a mechanism. If it survives more initial conditions
  it is a real finding -- the trained amplitude would be right for weather and
  too small for climate stability -- and it is cheap to test.
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

## 3. Epoch is not the story; seed is

`best_ckpt.tar` is rewritten whenever validation loss improves, so RF01's three
seeds sat at three different epochs when first scored -- S01 at 22, S03 at 24,
S02 at 28 -- and their 90-day RMSE was monotone in epoch (1.75, 2.16, 3.29 K)
while validation loss improved monotonically (0.0975, 0.0954, 0.0934). That
reads as "the checkpoint chosen on validation loss has the worst climate".

**It is not.** Re-scored with all three pinned to epoch 22, the ordering is
unchanged and the spread barely moves: 1.75, 2.08, 3.01 K, a 28.6% coefficient
of variation against 33.2% before. The apparent epoch trend was the seed
ordering coinciding with the epoch ordering. Within S02, going from epoch 22 to
28 costs 9% at 90 d -- real, but small next to the 70% gap between seeds at a
fixed epoch.

Two things survive from it anyway:

* Pin the weights before comparing anything. E01.S02's `best_ckpt.tar` was
  rewritten at 17:20 on 2026-09-05, twenty minutes before an evaluation read
  it. `eval.env` now records the checkpoint's size and mtime, and
  `analysis/checkpoint_epoch.py` says which epoch a file is.
* The one-year numbers are unstable in both seed and epoch -- S03 gives 3.29 K
  at epoch 24 and 2.19 K at epoch 22 -- so at 8 initial conditions treat one
  year as indicative and read 90 d.

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

1. **Read arm differences against §1.** In particular, do not plan to read
   90-day or annual temperature drift between arms at three seeds.
2. Score at 90 d rather than 1 y when a stable number is wanted at 8 ICs.
3. Pin checkpoints before any cross-arm comparison.
4. RF02 remains the only control that can answer "does stochastic training beat
   deterministic"; nothing in this file substitutes for it.
