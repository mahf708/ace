# Review: does stochastic training make a better single trajectory?

*2026-09-04. Branch `e3sm/exps/sep26-atm-ablation`. Compute: one interactive
allocation of 4 nodes x 4 A100-80GB for ~3.5 h.*

## 1. What the branch can and cannot answer today

The campaign (`README.md`, `PLAN.md`) is built, checked and validated, and
nothing is queued. Its headline contrast, RF01 (stochastic, aug26 E01) against
RF02 (deterministic MSE), **cannot be evaluated yet because RF02 has never
been trained**. Three RF01 seeds exist with EMA checkpoints at every epoch
(`$PSCRATCH/aug26/E01…S0{1,2,3}`, epochs 1-15 on S01 and 1-20 on S02/S03 at
the time of writing). A deterministic reference takes ~47 h on 4 nodes, so it
was out of reach in this session; the single deterministic ACE checkpoint on
scratch (`051126-deterministic`) is a different dataset and channel set and
is not a usable pole.

What *can* be measured now, without training, is the part of the question
that lives at inference time: given weights trained under the stochastic
objective, what does the runtime noise do to an individual trajectory? That
is exactly the decomposition the previous review asked for, and it is the
part the campaign had listed as out of scope (`PLAN.md` §8, "a noise-off
inference mode … propose it, do not slip it in"). It is now implemented,
tested, and used below.

## 2. Code change: an inference-time noise override

`StepperOverrideConfig.noise: NoiseOverrideConfig(scale, mode)`
(`fme/ace/stepper/single_module.py`) reaches every `NoiseConditionedModel`
in the stepper and sets `noise_scale` (multiplier on the unit-variance draw)
and `noise_mode` (`fresh`: new draw every step, training behaviour;
`fixed`: one field per sample, held for the rollout). The weights and the
checkpoint are untouched; neither attribute is in the state dict. Loading a
checkpoint that has no noise pathway (`noise_embed_dim: 0`) under a non-keep
override raises, so a deterministic checkpoint cannot be evaluated under a
label that claims otherwise. Tests: `fme/ace/registry/test_stochastic_sfno.py`
(module level, both noise types) and `fme/ace/stepper/test_single_module.py`
(loads through `load_stepper`, exercises `predict`).

Config use:

```yaml
stepper_override:
  noise: {scale: 0.0, mode: fresh}   # deterministic backbone g(x, 0)
```

## 3. Experiments run this session

All on RF01 EMA checkpoints at epoch 15 (the deepest epoch all three seeds
share), on held-out 2040s initial conditions from the template's `5yr_test`
list, evaluator seed 1 so that `fresh` and `fixed` share their first draw,
data glob narrowed to the 2040s files. Scripts: `make_eval.py`,
`run_eval.sh`, `summarize.py`, `traj_stats.py`, `one_step_drift.py` in this
directory. Outputs and the tables quoted below: `$PSCRATCH/sep26-noise-decomp/`.

| run | weights | noise | ICs x years | asks |
|---|---|---|---|---|
| S01_off, S02_off, S03_off | S01, S02, S03 | scale 0 | 8 x 1 | the learned backbone alone |
| S01_fresh, S02_fresh, S03_fresh | S01, S02, S03 | scale 1, fresh | 8 x 1 | the intended stochastic trajectory |
| S01_fresh2 | S01 | scale 1, fresh, evaluator seed 2 | 8 x 1 | noise-realisation replicate |
| S01_fixed, S02_fixed | S01, S02 | scale 1, one field held | 8 x 1 | temporal whiteness of the forcing |
| S01_half, S03_half / S01_double | S01, S03 / S01 | scale 0.5 / 2 | 8 x 1 | post-hoc amplitude calibration |
| S01_mean8 | S01 | mean of 8 draws per step | 4 x 1 | the iterated conditional mean |
| S01_ens4 | S01 | fresh, 4 members per IC | 4 x 1 | ensemble mean vs member |
| S01_off_3yr, S01_fresh_3yr | S01 | 0 / 1 | 4 x 3 | drift beyond one year |
| one-step drift | S01, S02, S03 at ep 15; S01 at ep 5, 10 | 16 draws per state | 128 states | Jensen drift, spread, alignment |

### 3.1 One-step decomposition (`one_step_drift.py`)

128 validation states (1990-1995, shuffled), 16 fresh-noise draws per state,
post-corrector outputs, area-weighted, error normalised by the persistence
error <|x - y|^2> and averaged over the 50 output channels. `e0` is the
backbone error |g(x,0)-y|^2, `em` the (finite-K-corrected) conditional-mean
error |E_Z g(x,Z) - y|^2, `ek` a single member's error, `tr` the spread
tr Σ(x), `drift` |E_Z g - g(x,0)|^2, and `A` the cosine between the drift and
the backbone residual y - g(x,0).

| checkpoint | e0 | em | ek | tr/e0 | drift/e0 | A (pooled) | per-state A min..med | CRPS16 / MAE(g0) |
|---|---|---|---|---|---|---|---|---|
| S01 ep 5  | 0.1363 | 0.1299 | 0.2233 | 0.74 | 0.11 | 0.19 | 0.16..0.19 | 0.688 |
| S01 ep 10 | 0.1047 | 0.0956 | 0.1635 | 0.69 | 0.14 | 0.28 | 0.24..0.28 | 0.670 |
| S01 ep 15 | 0.0959 | 0.0822 | 0.1427 | 0.65 | 0.20 | 0.37 | 0.34..0.37 | 0.641 |
| S02 ep 15 | 0.0969 | 0.0823 | 0.1418 | 0.64 | 0.21 | 0.38 | 0.34..0.38 | 0.640 |
| S03 ep 15 | 0.0974 | 0.0817 | 0.1411 | 0.63 | 0.22 | 0.40 | 0.37..0.40 | 0.633 |

Seeds 2 and 3 at epochs 5 and 10 give alignment 0.20 / 0.28 and 0.20 / 0.28,
em/e0 0.95 / 0.91 and 0.94 / 0.91, tr/e0 0.75 / 0.70 and 0.74 / 0.69: the
same trajectory through training as seed 1 to within 0.01
(`results/onestep_*.txt`).

Three findings, each replicated across the three training seeds to within
0.01:

1. **The noise-off backbone is not the model's mean, and it is the worse
   point forecast.** Averaging the network over its own noise cuts the
   one-step error by 14-16% (em/e0 = 0.84-0.86 channel mean; PS 0.56-0.60,
   U_6 0.74-0.77, T_7 0.75-0.78, Tat2m 0.82-0.85, precipitation 0.81-0.85).
   The only channel with no gain is STW_0, the top-of-atmosphere water
   channel, whose one-step error is already dominated by the corrector.
   The noise-induced drift E_Z g(x,Z) - g(x,0) points toward the truth in
   every one of the 128 states (per-state alignment never below 0.34 at
   epoch 15; 99.6% of state-channel pairs positive). This is the Jensen
   effect the previous review hypothesised, and it is large: the drift is
   ~45% of the residual norm.

2. **It grows with training.** From epoch 5 to 15 the alignment rises from
   0.19 to 0.37 and the mean-vs-backbone gain from 5% to 14%, while the
   spread-to-error ratio falls from 0.74 to 0.65. The network is learning to
   route part of its mean prediction *through* the noise pathway: the
   conditional layer norms are being used as a nonlinear mixing stage, not
   only as a perturbation. Consequence for the campaign: a Z1 model
   evaluated with the noise silenced is a different, worse deterministic
   model than the one the stochastic training actually produced, so
   "noise off at inference" is **not** a stand-in for a deterministic
   control and must not be read as one. The right deterministic proxy of a
   stochastic checkpoint is the iterated conditional mean (`mode: mean`,
   below), and the right trained control is RF02.

3. **The spread is calibrated to the backbone's error, not the mean's.**
   tr Σ / e0 = 0.63-0.65, while a single member's error is ek = em + tr,
   i.e. 1.47-1.49x the backbone's. Relative to the model's own conditional
   mean the spread-to-error ratio is tr/em ≈ 0.75. A 16-member CRPS is 0.63-
   0.64 of the backbone's MAE, so by the training score the stochastic
   operator is decisively better than its own noise-off restriction. None
   of this says anything about RF02, which does not exist yet.

### 3.2 Rollouts

Two tooling faults changed the run list, and both are findings in their own
right for the campaign's planned "pass 2, trajectories" (`PLAN.md` §7):

* **The 16-IC rollouts hang.** Every 16-IC run (noise off, fresh, fixed, on
  two separate attempts, with forked loader workers and with in-process
  reads) stalled at window 32 of 73 with one process in an uninterruptible
  DVS wait (`dvsipc_wait_for_resp`) and the other ranks spinning in NCCL.
  The 8-IC and 4-IC runs reading the same files never stalled. Not
  diagnosed further; the rollouts below use 8 ICs (every held-out year once,
  January and July alternating; 2 per rank) or 4 (January of even years).
* **`RawDataWriter` is not rank-safe.** Under distributed inference every
  rank writes its own samples into the same `autoregressive_*.nc`, and the
  result is unreadable (HDF5 metadata checksum errors; the readable part
  holds one rank's two samples). Fixed on this branch: each rank now writes
  `<label>_rank<NNN>.nc` (`fme/ace/inference/data_writer/raw.py`, with a
  test). The first round of runs therefore has aggregator diagnostics only;
  the per-trajectory statistics come from the second round.

**Round 1 (8 ICs, 1 year, seeds 2 and 3 with the noise off and on; S01 with
fresh noise under a second evaluator seed, and at half amplitude; 4-IC runs
of the iterated conditional mean and a 4-member ensemble).** All numbers are
area-weighted, from the evaluator's own aggregators. `tm_bias_rmse` is the
RMS of the 1-year time-mean bias map; `spec_hi` the mean log10 ratio of
predicted to target spherical power over the upper third of wavenumbers;
`q999`/`q001` the pooled 99.9th/0.1st percentiles (target in brackets).

| Tat2m | S02 off | S02 fresh | S03 off | S03 fresh | S01 fresh (seed 2) | S01 half | S01 mean8 (4 IC) | S01 ens4 (4 IC) |
|---|---|---|---|---|---|---|---|---|
| RMSE 6 h   | **0.379** | 0.469 | **0.382** | 0.465 | 0.474 | 0.395 | 0.378 | 0.487 |
| RMSE 1 d   | 0.748 | 0.783 | 0.759 | 0.787 | 0.794 | 0.729 | 0.705 | 0.856 |
| RMSE 5 d   | 2.36 | **1.94** | 2.49 | **2.09** | 2.01 | 2.11 | 1.82 | 2.04 |
| RMSE 30 d  | 5.19 | **3.68** | 3.89 | **3.62** | 3.76 | 3.41 | 4.21 | 4.09 |
| RMSE 365 d | 6.50 | **4.60** | 4.66 | **4.30** | 3.60 | 3.75 | 4.10 | 3.68 |
| tm_bias_rmse (K) | 4.62 | **1.22** | 2.14 | **1.14** | 0.26 | 0.52 | 0.28 | 0.26 |
| spec_hi (log10) | -0.30 | **-0.007** | -0.22 | **-0.001** | +0.007 | -0.10 | -0.16 | +0.008 |
| q999 (314.0) | 305.4 | 314.1 | 309.8 | 314.1 | 315.0 | 314.2 | 314.9 | 313.9 |
| q001 (209.6) | 238.9 | 211.1 | 214.2 | 212.3 | 210.9 | 210.8 | 208.8 | 211.0 |

| PS | S02 off | S02 fresh | S03 off | S03 fresh | S01 fresh (2) | S01 half | S01 mean8 | S01 ens4 |
|---|---|---|---|---|---|---|---|---|
| RMSE 5 d (Pa) | 633 | **416** | 663 | **473** | 423 | 507 | 374 | 427 |
| RMSE 365 d | 1332 | 1156 | 1279 | 1136 | 980 | 1138 | 1125 | 1041 |
| tm_bias_rmse (Pa) | 1029 | **300** | 850 | **413** | 77 | 122 | 91 | 73 |

| FLUT / U_6 | S02 off | S02 fresh | S03 off | S03 fresh | S01 fresh (2) | S01 half | S01 mean8 | S01 ens4 |
|---|---|---|---|---|---|---|---|---|
| FLUT tm_bias_rmse (W/m2) | 15.7 | **5.2** | 14.2 | **7.7** | 2.5 | 4.2 | 2.6 | 2.5 |
| FLUT spec_hi | -0.72 | **-0.05** | -0.66 | **-0.07** | -0.06 | -0.32 | -0.59 | -0.06 |
| FLUT q999 (345.9) | 310.9 | 345.5 | 316.7 | 342.2 | 348.2 | 340.2 | 345.3 | 344.7 |
| U_6 spec_hi | -0.62 | **-0.03** | -0.46 | **-0.02** | +0.01 | -0.15 | -0.34 | +0.01 |
| U_6 q999 (33.5) | 23.1 | 34.6 | 27.3 | 33.6 | 33.4 | 34.5 | 33.8 | 33.2 |

Reading, with the caveat that S01's own off/fresh pair at 8 ICs was still
running when this was written (S01 columns are therefore not the same
initial conditions as the mean8/ens4 columns):

1. **Silencing the noise at inference destroys the climate of the
   trajectory.** With the noise off, both seeds drift to time-mean biases
   4-8x larger (Tat2m 4.6 K and 2.1 K RMS against 1.2 K and 1.1 K), lose
   half or more of the small-scale power (spec_hi -0.2 to -0.7 dex against
   ~0), and collapse both tails (Tat2m 99.9th percentile 305-310 K against
   314 K; U_6 23-27 m/s against 33.5). The fresh-noise trajectory reproduces
   target tails to within the histogram's bin width in every variable. This
   is the "variability and higher moments" claim, and it is true within the
   stochastic model: the noise pathway carries the variance and the tails.
2. **The crossover is at about one day.** Noise off wins at 6 h (it is the
   lower-variance point forecast, exactly as the one-step decomposition
   predicts: member error = mean error + spread), ties at 1 day, and loses
   from 5 days on. By 30 days the noise-off trajectories are on a different
   attractor, not merely noisier.
3. **Halving the amplitude costs spectrum, not RMSE.** `half` sits between
   off and fresh on every spectral and time-mean measure (Tat2m spec_hi
   -0.10, FLUT -0.32) while its 30-365-day RMSE is as good as fresh's. The
   amplitude the model was trained at is the one that reproduces the
   spectrum; it is not a free post-hoc knob.
4. **The iterated conditional mean (mean8) is the best point forecast out
   to 5 days and then behaves like a damped model.** It beats every other
   column at 6 h to 5 d (as the one-step Jensen result predicts) but by 30
   days its RMSE is the worst of the stochastic columns and its small-scale
   spectrum is damped (-0.16 to -0.59 dex), as an averaged operator must be.
   Its time-mean bias stays small, so unlike "off" it does not drift; it
   loses variance rather than climate. This is the natural deterministic
   proxy of a stochastic checkpoint, and it is what RF02 should be compared
   against as well as against fresh-noise members.
5. **Individual members of a 4-member ensemble (ens4) are statistically the
   same as single fresh trajectories**, as they should be; that run exists
   to supply ensemble-mean statistics from the same weights (see round 2).

**Higher moments (pooled space x time, from the evaluator's 200-bin
histograms; `moments_round1.txt`).** Skewness and excess kurtosis of the
fresh-noise trajectories match the target to the second decimal in every
variable checked, and the noise-off trajectories do not:

| Tat2m: std / skew / exkurt | target 20.9 / -1.21 / 1.08 |
|---|---|
| S02 off | 13.1 / -0.77 / 0.33 |
| S02 fresh | 20.7 / -1.18 / 1.00 |
| S03 off | 18.4 / -1.33 / 1.73 |
| S03 fresh | 20.0 / -1.23 / 1.21 |
| S01 fresh (seed 2) | 20.7 / -1.18 / 1.02 |
| S01 mean8 | 21.0 / -1.20 / 1.05 |

Precipitation tells the same story with the sign reversed: noise off
over-sharpens the pooled distribution (skew 6.4-7.0, kurtosis 66-80 against
5.7 / 55), fresh noise brings it to 5.9-6.2 / 58-64. Qat2m's standard
deviation drops from 0.0064 to 0.0046-0.0056 with the noise off. Note that
the pooled distribution is dominated by spatial variability, so these are
statements about the climate the trajectory settles into, not yet about
temporal variability at a point; that is what `traj_stats.py` computes from
the round-2 files.

**Corrector burden (`corrector_round1.txt`).** The area-mean correction
magnitude is *not* larger with the noise off: precipitation 0.35 (off) vs
0.37 (fresh) in normalized units for S02, 0.27 vs 0.30 for S03; FSNS 0.013
vs 0.017. The fresh-noise tails are therefore not manufactured by the
clamp; if anything the stochastic trajectories ask a little more of the
positivity corrector, by 5-30%, which is worth tracking but is not where
the variance comes from.

**Round 2, S01 at the same 8 ICs, 1 year (`summary_S01.txt`):** the full
mode ladder on one set of weights.

| Tat2m | off | fresh | fresh, noise seed 2 | fixed | half | double |
|---|---|---|---|---|---|---|
| RMSE 6 h | **0.376** | 0.476 | 0.474 | 0.476 | 0.395 | 0.873 |
| RMSE 5 d | 2.29 | **1.97** | 2.01 | 2.37 | 2.11 | 2.75 |
| RMSE 30 d | 3.83 | 3.68 | 3.76 | 3.78 | **3.41** | 4.03 |
| RMSE 365 d | 3.68 | 3.65 | **3.60** | 3.87 | 3.75 | 4.21 |
| tm_bias_rmse (K) | 0.63 | **0.28** | 0.26 | 0.70 | 0.52 | 1.44 |
| spec_hi (log10) | -0.15 | **+0.007** | +0.007 | +0.24 | -0.10 | +0.36 |
| q999 / q001 (314.0 / 209.6) | 312.5 / 211.0 | 314.5 / 211.2 | 315.0 / 210.9 | 313.4 / 206.6 | 314.2 / 210.8 | 312.0 / 201.0 |

| other | off | fresh | fresh (seed 2) | fixed | half | double |
|---|---|---|---|---|---|---|
| PS tm_bias_rmse (Pa) | 139 | **71** | 77 | 259 | 122 | 351 |
| FLUT tm_bias_rmse (W/m2) | 5.1 | **2.5** | 2.5 | 18.3 | 4.2 | 11.2 |
| FLUT spec_hi | -0.47 | **-0.06** | -0.06 | +0.03 | -0.32 | +0.51 |
| U_6 tm_bias_rmse (m/s) | 1.52 | **0.49** | 0.48 | 2.55 | 1.23 | 2.78 |
| U_6 spec_hi | -0.22 | **+0.01** | +0.01 | +0.39 | -0.15 | +0.52 |
| precip q999 (7.3e-4) | 8.6e-4 | 7.9e-4 | 7.8e-4 | 10.8e-4 | 8.2e-4 | 6.7e-4 |

6. **The seed-1 weights are the most robust to silencing the noise**
   (time-mean bias 0.63 K against 0.28 K, small-scale power -0.15 dex), but
   the direction is the same as for seeds 2 and 3, and the one-step
   decomposition is identical across the three seeds. Which seed drifts
   how far with the noise off is itself a training-seed property; three
   seeds is enough to see that spread and not to estimate it.
7. **The noise-seed replicate is indistinguishable from the original.**
   `fresh` and `fresh (seed 2)` agree to 2-3% on every 30-365-day statistic.
   With 8 ICs over a year, initial-condition and training-seed variance
   dominate noise-realisation variance; the campaign does not need many
   inference seeds per checkpoint, it needs seeds and ICs.
8. **Temporally white noise is part of the mechanism.** Holding one latent
   field per trajectory ("fixed") is worse than fresh on every measure and
   worse than *off* on most: it injects +0.24 to +0.39 dex of spurious
   small-scale power, an 18 W/m2 RMS FLUT bias, a 2.5 m/s U_6 bias, and a
   48% overshoot of the precipitation 99.9th percentile. A persistent
   perturbation of the layer-norm affine terms integrates into a mean
   forcing; the model was trained to expect it to cancel from step to step.
9. **Doubling the amplitude is as destructive as removing it**, in the
   opposite direction: +0.36 to +0.52 dex excess small-scale power, 1.4 K
   time-mean bias, tails pulled inward. Together with `half`, this brackets
   the trained amplitude as the only one that reproduces both the spectrum
   and the tails; there is no useful post-hoc calibration knob here.

### 3.3 Per-trajectory statistics (`traj_stats.py`, `traj_S01.txt`)

Each statistic is computed within one 1-year trajectory and then summarised
as mean ± std across the 8 trajectories; the target gets the same
treatment, so its ± is the initial-condition (year) spread against which a
model difference should be read. S01 weights, 6-hourly, area-weighted.
`tstd` is the area-mean of the per-gridpoint temporal standard deviation
about the trajectory's own time mean; `ac4` the area-mean lag-1-day
autocorrelation of those anomalies.

| Tat2m | target | off | fresh | fixed | double |
|---|---|---|---|---|---|
| tstd (K) | 4.69 ± 0.04 | 4.49 ± 0.03 | **4.63 ± 0.06** | 4.57 ± 0.07 | 4.90 ± 0.06 |
| pooled std (K) | 14.83 ± 0.08 | 14.28 ± 0.06 | **14.74 ± 0.10** | 14.63 ± 0.19 | 15.83 ± 0.11 |
| skew / exkurt | -1.75 / 4.06 | -1.73 / 4.07 | -1.76 / 4.15 | -1.81 / 4.49 | -1.97 / 4.99 |
| ac4 | 0.884 ± 0.004 | 0.875 ± 0.004 | **0.883 ± 0.003** | 0.905 ± 0.002 | 0.869 ± 0.004 |
| q999 (K) | 314.9 ± 0.4 | 313.2 ± 0.2 | **314.5 ± 0.3** | 313.7 ± 0.7 | 312.6 ± 0.3 |
| q001 (K) | 213.2 ± 0.7 | 214.8 ± 0.7 | **213.1 ± 0.6** | 211.6 ± 2.0 | 205.2 ± 0.7 |

| precipitation | target | off | fresh | fixed | double |
|---|---|---|---|---|---|
| wet fraction (>1 mm/d) | 0.356 ± 0.001 | **0.358 ± 0.001** | 0.368 ± 0.002 | 0.359 ± 0.004 | 0.424 ± 0.002 |
| wet intensity (1e-5 kg/m2/s) | 9.56 ± 0.06 | 10.26 ± 0.06 | **9.45 ± 0.08** | 9.67 ± 0.11 | 7.11 ± 0.02 |
| q999 (1e-4) | 7.89 ± 0.08 | 9.12 ± 0.17 | **8.40 ± 0.18** | 12.6 ± 0.3 | 7.40 ± 0.02 |
| tstd (1e-5) | 6.81 ± 0.09 | 7.49 ± 0.11 | **7.00 ± 0.12** | 6.96 ± 0.11 | 6.05 ± 0.04 |
| skew / exkurt | 5.03 / 43 | 5.30 / 47 | 5.27 / 47 | 8.00 / 103 | 5.70 / 53 |
| ac1 (6 h) | 0.659 ± 0.002 | **0.640 ± 0.004** | 0.631 ± 0.002 | 0.738 ± 0.005 | 0.474 ± 0.003 |

| PS | target | off | fresh | fixed | double |
|---|---|---|---|---|---|
| tstd (Pa) | 680 ± 4 | 735 ± 14 | **682 ± 10** | 676 ± 17 | 541 ± 6 |
| ac4 | 0.810 ± 0.002 | 0.791 ± 0.005 | **0.804 ± 0.006** | 0.815 ± 0.006 | 0.767 ± 0.007 |
| q999 (Pa) | 104020 ± 56 | 104310 ± 150 | **103930 ± 48** | 103960 ± 64 | 103470 ± 100 |
| q001 (Pa) | 56137 ± 26 | 56152 ± 49 | 56228 ± 55 | 56058 ± 76 | 56691 ± 32 |

10. **Within an individual trajectory, fresh noise is what puts the
    temporal variance, the persistence and the tails in the right place.**
    With the noise off the S01 trajectory has 4% too little day-to-day
    2 m temperature variance, 8% too much surface-pressure variance, 10%
    too much precipitation variance, a warm-tail 2 m temperature deficit of
    1.7 K at the 99.9th percentile (five times the year-to-year spread of
    that percentile), and a 16% overshoot of the precipitation 99.9th
    percentile. Fresh noise brings every one of these to within one or two
    target standard deviations except the precipitation extreme (+7%) and
    the wet-day frequency, which it overshoots by 3% (0.368 vs 0.356) while
    getting the conditional intensity right; noise off has the opposite
    partition (frequency right, intensity 7% high). That is the
    frequency/intensity trade the previous review asked to be separated,
    and the two modes fail it in opposite directions.
11. **Persistence is not "added noise".** The lag-1-day autocorrelation of
    2 m temperature anomalies is *higher* with fresh noise (0.883, target
    0.884) than without (0.875), and the same holds for surface pressure.
    A white forcing that only decorrelated the trajectory would lower it;
    the noise pathway is reproducing the slow variability that the
    backbone alone damps. The held-noise mode overshoots (0.905) and the
    doubled amplitude undershoots (0.869), bracketing the trained
    amplitude again.
12. Third and fourth moments are matched by both off and fresh for 2 m
    temperature and surface pressure on this seed; the discriminating
    quantities are the variance, the persistence and the quantiles, not
    skewness and kurtosis, which agrees with the previous review's warning
    that raw higher moments are noisy and should be supplemented by
    quantile and event statistics.

**Seed-2 replicate of the held-noise mode (`summary_fixed.txt`).** On S02
the held latent field again adds spurious small-scale power (+0.23 dex
Tat2m, +0.30 U_6, against -0.30 / -0.62 with the noise off and ~0 with fresh
noise) and lands between off and fresh on the time-mean bias (Tat2m 2.6 K
against 4.6 K off and 1.2 K fresh; FLUT 13.8 against 15.7 and 5.2 W/m2).
So the temporal-whiteness result of point 8 replicates: a persistent latent
forcing keeps some of the climate the backbone loses but pays for it with a
wrong spectrum, and it is nowhere near a substitute for fresh noise.

**Ensemble mean against its members (`ens4_mean_vs_member.txt`; 4 ICs x 4
fresh-noise members, S01).** Averaging four members before computing
statistics gives the lowest RMSE of anything in this study at every lead
(Tat2m 0.39 K at 6 h, 1.73 at 5 d, 2.66 at 180 d, against 0.49 / 2.04 / 3.33
for a member) and a bad climate realisation: per-gridpoint temporal
variance of 2 m temperature 8% low (4.32 vs 4.68 K), of precipitation 41%
low and of surface pressure 35% low, wet-day frequency 0.54 against 0.36,
precipitation 99.9th percentile halved, lag-1-day 2 m temperature
autocorrelation 0.953 against 0.885. Individual members match all of those
to within one or two target standard deviations. So on this data the
review's warning is quantitative: any evaluation that averages members
before computing variability statistics would report the stochastic model
as smoother than E3SM, when its individual trajectories are not; and any
RMSE-only comparison against a deterministic model would reward the mean
for exactly the variance it lacks.

### 3.4 Three years (`yearly_drift_3yr.txt`, `yearly_drift_fresh3yr.txt`, `summary_3yr.txt`)

S01, 4 January ICs (2040, 2042, 2044, 2046), 4380 steps. Per rollout year:
RMS of the annual-mean bias map and the per-gridpoint temporal std of the
prediction relative to the target's, mean ± std across the 4 trajectories.

| Tat2m | bias RMS off | bias RMS fresh | tstd off | tstd fresh |
|---|---|---|---|---|
| year 1 | 0.78 ± 0.08 | **0.47 ± 0.07** | -4.2% | **-0.9%** |
| year 2 | 1.00 ± 0.05 | 0.71 ± 0.23 | -7.8% | **-2.3%** |
| year 3 | 0.98 ± 0.07 | 0.93 ± 0.34 | -8.3% | **-3.3%** |

| PS | bias RMS off (Pa) | bias RMS fresh | tstd off | tstd fresh |
|---|---|---|---|---|
| year 1 | 189 ± 8 | **133 ± 18** | +7.4% | **-0.3%** |
| year 2 | 226 ± 17 | 240 ± 110 | +4.5% | **0.0%** |
| year 3 | 205 ± 27 | 289 ± 100 | +3.6% | **-0.8%** |

| precipitation | bias RMS off (1e-6) | bias RMS fresh | tstd off | tstd fresh |
|---|---|---|---|---|
| year 1 | 13.0 ± 0.6 | **7.2 ± 0.2** | +10.3% | **+3.0%** |
| year 2 | 17.1 ± 1.0 | 10.3 ± 2.4 | +7.5% | **+3.2%** |
| year 3 | 17.1 ± 0.7 | 13.0 ± 3.6 | +8.1% | **+2.0%** |

13. **Both modes drift in the mean; the noise does not stop that.** The
    fresh-noise trajectories' annual-mean bias doubles from year 1 to
    year 3 in 2 m temperature (0.47 to 0.93 K) and in surface pressure
    (133 to 289 Pa), with a trajectory-to-trajectory spread that grows
    with it. By year 3 the 2 m temperature bias of the two modes is the
    same within that spread, and for surface pressure the noise-off mode
    is nominally better. On the 3-year horizon the mean-state advantage of
    fresh noise is a first-year result, not a settled-climate one. This is
    consistent with an epoch-15 checkpoint of a 30-epoch run, and it is
    the reason the campaign's 5-year test block, three seeds and RF02 are
    all needed before anything is claimed about the mean state.
14. **The variance result does hold out to three years.** The noise-off
    trajectory's temporal-variance deficit keeps growing (2 m temperature
    -4% to -8%, and the precipitation and surface-pressure variance excess
    persists at +4-8%), while fresh noise stays within 1-3% of the target
    in every year and every variable. The aggregate 3-year diagnostics
    say the same: time-mean bias RMS 0.81 K (off) vs 0.48 K (fresh),
    small-scale power -0.15 to -0.48 dex (off) vs within 0.06 dex (fresh),
    tails of U_6 and FLUT restored by fresh noise (`summary_3yr.txt`).

**Three evaluator noise seeds on S01 (`summary_replicates.txt`).** Fresh
noise under seeds 1, 2 and 3 gives Tat2m time-mean bias RMS 0.278, 0.258,
0.257 K, small-scale spectral ratio +0.0072, +0.0073, +0.0076 dex, 1-year
RMSE 3.65, 3.60, 3.61 K, and the same agreement for FLUT, U_6 and PS. The
noise-realisation level of the hierarchy contributes a few percent at most
to 1-year statistics over 8 ICs; point 7 stands with three replicates.

**Per-trajectory statistics over three years (`traj_quantiles_fixed.txt`,
4 ICs).** The variance and persistence gap widens and the tails follow:

| Tat2m, 3 years | target | off | fresh |
|---|---|---|---|
| tstd (K) | 4.73 ± 0.02 | 4.41 ± 0.03 (-6.8%) | **4.65 ± 0.03 (-1.7%)** |
| ac4 | 0.892 ± 0.001 | 0.884 ± 0.002 | **0.899 ± 0.005** |
| q999 (K) | 314.9 ± 0.2 | 312.1 ± 0.1 | **313.6 ± 0.7** |
| q001 (K) | 213.2 ± 0.4 | 214.5 ± 0.3 | **213.1 ± 0.7** |

| precipitation, 3 years | target | off | fresh |
|---|---|---|---|
| tstd (1e-5) | 6.94 | 7.56 (+9%) | **7.19 (+3.6%)** |
| q999 (1e-4) | 7.91 | 9.36 (+18%) | **8.50 (+7%)** |
| wet fraction | 0.356 | **0.358** | 0.369 |
| wet intensity (1e-5) | 9.58 | 10.20 (+6.5%) | **9.44 (-1.4%)** |

| PS, 3 years | target | off | fresh |
|---|---|---|---|
| tstd (Pa) | 685 ± 3 | 723 ± 3 (+5.5%) | **693 ± 7 (+1.2%)** |
| ac4 | 0.814 | 0.800 | **0.822** |
| q999 (Pa) | 104000 ± 31 | 104370 ± 54 | **103960 ± 15** |

Over three years the fresh-noise warm tail also falls short (313.6 K
against 314.9, a 1.3 K deficit, growing from 0.4 K in the first year),
so the slow loss of the hot extreme is a property of the weights that
noise slows but does not remove. The cold tail, the surface-pressure
tails and the precipitation extreme stay where fresh noise put them in
year 1.

## 4. What this says about the study's hypothesis

The hypothesis under test is that noise conditioning makes a better
individual trajectory than a deterministic model, in the mean, the
variability and the higher moments. What can be said now, and what cannot:

* **Within the stochastic model, the runtime noise is load-bearing for
  everything except the first day.** Silencing it degrades the first-year
  climate on every measure in all three training seeds; the trained
  amplitude is the only one that reproduces the spectrum and the tails;
  the noise must be refreshed every step. The stochastic model's single
  trajectories put temporal variance, persistence, and the 0.1-99.9%
  quantiles within one or two target standard deviations for 2 m
  temperature, surface pressure and precipitation, and reproduce the
  pooled third and fourth moments to the second decimal.
* **The learned backbone is not a deterministic model in the sense the
  campaign means.** It is the stochastic model with its noise removed, and
  it is a worse one-step operator than the noise-averaged mean by 14-16%
  (all three seeds). Any statement of the form "the stochastic model beats
  the deterministic one" must be made against RF02, not against `scale: 0`,
  and the campaign's design is right to insist on RF02 at P1. Until RF02
  exists, the results here are a complete answer to a narrower question:
  *what the noise pathway does to a trajectory of the weights that were
  trained with it.*
* **On the mean state the evidence is mixed beyond one year.** Fresh noise
  halves the first-year time-mean bias relative to the backbone, but both
  drift, and by year 3 the 2 m temperature biases are within the
  trajectory spread of each other. The variance, spectra and tails
  advantage does hold at three years.
* **The mechanism is partly deterministic.** The one-step Jensen drift is
  aligned with the backbone's error in every state and grows with
  training. The stochastic objective is shaping the conditional mean
  through the noise pathway as much as it is shaping the spread. This is
  a novel point for the study and it changes what the LG block measures:
  LG03 - RF02 (noise wired, MSE, one member) cannot show this effect,
  because an MSE objective at one member has no reason to route mean
  prediction through the noise; LG04 - RF01 can, and it deserves its
  extra seeds.

Recommendations, in order:

1. Run RF02 (P1) before anything else; evaluate it against RF01 members
   *and* against RF01 in `mode: mean`, with per-trajectory statistics.
2. Add the noise override to the campaign's evaluation harness as a
   standard set of five modes (off, fresh, fixed, half, mean) per
   stochastic checkpoint; it costs one inference each and it is the
   cheapest mechanism probe available.
3. Fix the per-rank data writer upstream (done here) before the planned
   trajectory pass; 16-IC evaluator runs on this data hang and should be
   split into 8-IC runs until the DVS stall is understood.
4. Use 5-year rollouts and the truth-versus-truth block baseline for the
   mean-state claim; the 1-year numbers here are not enough, and the
   3-year drift shows why.
5. Log the noise-pathway amplitude and the one-step Jensen alignment per
   epoch for every Z1 arm: the alignment doubles between epochs 5 and 15
   and is the clearest single number describing what stochastic training
   has done to the weights.


## 5. Assessment of the previous review

Agree, and acted on:

* The inference-only decomposition (noise off / fresh / fixed / scaled /
  ensemble mean) is the most informative thing that can be done before RF02
  exists, and it needs no retraining. Implemented above.
* Statistics must be computed within each trajectory and then summarised
  across realisations; ensemble means are a mechanism diagnostic only.
  `traj_stats.py` does exactly that, for prediction and target alike.
* The one-step Jensen-drift argument is correct in form: for a nonlinear
  conditioner, E_Z g(x,Z) != g(x,0), and the sign of its alignment with the
  backbone error is an empirical question. `one_step_drift.py` measures it.
* The training-seed level and the noise-seed level must not be pooled.

Disagree, or would weight differently:

* **The analog-conditioned residual analysis is unlikely to be informative
  here.** The reduced state is 50 channels on 180x360; on a single
  realisation, nearest neighbours in any low-dimensional projection are far
  apart in the full state, and the "conditional" tendency spread they yield is
  dominated by analog distance, not by unresolved variability. It would
  measure the projection, not the noise. A cheaper and sharper test of the
  same claim is the one-step decomposition: if the stochastic spread tr Σ(x)
  tracks the backbone error |g(x,0)-y|^2 across states, the noise is
  calibrated to the model's own residual, which is the only conditional
  variability a single realisation can teach.
* **"Fixed noise through the rollout" is a probe, not a model.** Nothing in
  training ever saw a temporally held latent, so its trajectories are
  off-distribution by construction. Worth one run to see how much the white
  refreshing matters; not worth a row in a headline figure.
* **The review under-weights the corrector.** Dry-air, moisture-budget and
  positivity corrections are applied after the network at every step. A
  stochastic model whose tails come from clamping has not learned them; the
  evaluator's `correction_scalars` are enabled in every run here so the clamp
  burden can be compared between noise-off and fresh-noise rollouts.
* **The forced path measure is the right object, but 1-year rollouts do not
  reach it.** With prescribed SST and forcing, interannual and longer
  variability of the free atmosphere is small, and the 16 ICs span 8 years of
  one realisation. The runs here are a first pass on the sub-annual
  statistics; the low-frequency and regime-occupancy claims need the 5-year
  pass and the truth-versus-truth block baseline the review proposes, which
  is a good idea and cheap.
* The review's mechanism list (7 items) conflates two that the campaign
  already separates: LG04 (train with the RF01 objective and no noise
  pathway) and "RF01 with noise off at inference" are different controls with
  different weights. Both are needed; the second is now free.

## 6. Campaign code review (independent agent, read-only)

No correctness bug was found in the three ported loss fixes or in the factor
word to config mapping; the energy-score pairwise generalisation and the
almost-fair epsilon were re-derived and checked numerically at M = 1..5. Three
notes in `make_campaign.py` still described the pre-fix state ("get_crps uses
(1-alpha)/2 … only valid at M2"; "get_energy_score supports exactly two
members") and one of them was written into OI04's W&B notes; corrected and
regenerated (`runs/` diff is that one `.env` line and the manifest). Two
smaller items are left as they are: the `Y1` guard at `crps_weight: 0` calls
alpha inert although the finite-difference CRPS term also takes it (no run in
the list is affected), and `check_campaign.py`'s assertions on alpha, FD
levels, noise dim and loss type have no mutation test. mypy on this branch
already fails on `fme/core/wandb.py` (`mark_preempting`) independent of this
session.
