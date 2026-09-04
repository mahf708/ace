# sep26 — an atmosphere-only ablation campaign

Status: **built, checked, validated. Nothing queued.** 19 runs, 5,741
node-hours. This file is the design argument and the measurement record;
`README.md` is the reference, `TODO.md` what is left, `AGENTS.md` the history.

Superseded by the current scheme: §3 below described a sparse-delta run id that
has been replaced by an aug26-style fixed-order factor word plus a two-letter
study prefix. See `README.md` for the convention that is actually in use; §3 is
kept for the argument about *why* a naming scheme is load-bearing, not for its
specific proposal.

Branch: `e3sm/exps/sep26-atm-ablation`, off `e3sm/exps/hist-v2026.8.0`
(**not** off `main` — see "The unmerged dependency").

---

## 1. Why this is a new campaign and not E29–E39 of aug26

The plan this replaces added seven single-factor arms plus a four-run mechanism
block to the aug26 list, numbered E29–E39. The design is good; the container is
wrong, and the container was already deforming the design.

aug26's run id is a positional word:

    E21.aug26.atm.A0_B16_C0_L0_O5_W0_X0.D1_I0_M1_RF1_Z00.S01

`A0_B16_C0_L0_O5_W0_X0` is fixed-order, so widening it renames all 35 live run
ids — which are live W&B run names and live scratch directories. That is why
the stochastic block got a *second* dotted word rather than more positions.
Extending that block the same way needed a **third** word (`G_N_Q_Y`) and a
**fourth** (`T###`), and four optional dotted fields are only parseable if each
is identified by its first letter rather than its position. At that point the
alphabet is 20 of 26 letters spent (`A B C D E G I L M N O Q R S T W X Y Z`),
and the id still does not say what the run is testing without the table.

Five concrete costs of staying in aug26:

1. **Renaming pressure is structural, not incidental.** Every future axis needs
   either a new dotted word or a rename. The convention has no slack left.
2. **The E-sequence is shared across realms and campaigns.** E29–E39 would
   permanently interleave an atmosphere ablation with aug26's ocean and coupled
   runs in one numbering.
3. **The tuning word is dead weight here.** Every arm sits on E01's
   `A0_B16_C0_L0_O5_W0_X0`; 22 characters that never vary. `O5` — the *ocean*
   cadence — appears in every id of an atmosphere-only campaign.
4. **The control is a run, not a template.** `config-train-atm.yaml` *is* E01,
   so the generator cannot write the baseline — it has to *assert* that the
   baseline still matches `Training()`'s defaults and raise if it drifted. That
   guard exists because the design has no template. sep26 can have one.
5. **The reservation is a different resource.** `_CAP_aigs_hist` is 96 hbm80g
   nodes, 83% consumed, and ends **2026-09-05 15:00** — two days from now.
   Neither E18–E28 nor this block comes out of it.

sep26 is greenfield: no live ids, one realm, one tuning set, one question
family. It should pay none of that.

### What sep26 *does* inherit from aug26, deliberately

The two three-seed references, which are the reason this campaign is cheap:

| reference | aug26 id | what it is |
|---|---|---|
| **REF-S** | `E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S{01,02,03}` | stochastic pole: EnsembleLoss crps 0.9 / energy 0.1, 2 members, noise 32, 1 step |
| **REF-D** | `E21.aug26.atm.…D1_I0_M1_RF1_Z00.S{01,02,03}` | deterministic pole: MSE, 1 member, no noise, 1 step |

Every sep26 arm is one factor from one of these, and neither is re-run. REF-S
is running now (S01–S03 at epoch 10 of 30 as of 2026-09-03). **REF-D does not
exist yet** — it is E21, queued at P5 in aug26 and gated behind a window that
has not opened. That is a hard prerequisite for half of this campaign and it is
the first thing to check on the schedule.

---

## 2. 40 GB cards: the answer, and what it changes

**Short answer: yes for the atmosphere, with margin, and it is the single
biggest structural win available to this campaign.** Numbers below are measured
on 2026-09-03; see "Measurements".

The starting point was a model fitted to two aug26 points — 19.0 GB/GPU at
local batch 1 with 2 members, and 28.7 GB at local batch 2:

    per-GPU GB  ~=  9.3  +  4.85 * (local_batch * n_ensemble * n_scored_steps)

9.3 GB is fixed state (456 M parameters in fp32, gradients, two FusedAdam
moments, EMA weights); 4.85 GB is activations per effective batch element at
`checkpointing: 3`. **It was right about members and wrong about rollouts.**
Peak MiB measured on the card, `analysis/card-sweep.sh`, 2026-09-03:

| arm | predicted GB | **measured MiB (40 GB)** | 40 GB headroom | measured s/batch |
|---|---|---|---|---|
| 1 member, 1 step | 14.2 | **15,383** | 62% | 0.433 |
| **2 members, 1 step (REF-S)** | 19.0 | **21,155** | 48% | 0.903 |
| 3 members, 1 step | 23.9 | **24,651** | 40% | 1.304 |
| 2 members, 2 steps, **both scored** | 28.7 ✗ | **21,921** | 46% | 1.713 |

**The rollout term is far smaller than the model said, but it is not zero.** A
both-scored 2-step rollout was predicted at 28.7 GB and measures 21.9 — barely
above the 1-step number, because `optimization.use_gradient_accumulation: true`
runs each scored step's backward before the next forward, so the two steps'
activations are never held at once. A *fixed 20-step* last-step-only rollout
measures **23,233 MiB**, about 2 GB above the 1-step arm — so depth does cost
something, just sub-linearly and far less than the number of steps.

That 20-step figure is **single-card**, and the reason is worth recording. Its
80 GB counterpart read 17,471 MiB — 5.8 GB *below* the 1-step arm, which is
impossible for a strictly larger workload. The explanation is in the step count:
that run logged **zero** training steps, having spent its whole deadline
building 31-timestep windows, so its peak is the setup-and-model-build
high-water mark rather than a training peak. Every variant that actually
trained agrees across the two cards to within 5.7%, and the four that got their
full step budget agree to 2–4 MiB.

The harness now labels a peak `valid=NO-not-a-training-peak` when the run
logged fewer than three steps. Without that, an invalid measurement is
indistinguishable from a real one and reads as a cross-card disagreement.

Memory is therefore driven by **members**, with a mild rollout-depth term on
top. The worst arm in this campaign peaks at **24,651 MiB of 40,960, with 40%
headroom**, and it is the three-member one rather than any rollout one.

Host memory is the other axis and it is also fine: `time_buffer: 10` with
`num_data_workers: 8`, `prefetch_factor: 4` was measured at 19 GB/node for the
atmosphere, against 256 GB on a 40 GB node. (The *ocean* costs 84 GB there and
is the reason the aug26 sbatch scripts say "all three configs require 80 GB
cards". That comment is about the ocean and the coupled config. An
atmosphere-only campaign is not bound by it.)

### What it unlocks

| | hbm80g | hbm40g |
|---|---|---|
| nodes in `gpu_ss11` | 256 | **1408** |
| idle at the time of writing | 1 | 27 |
| needs a reservation to run 10 arms concurrently | yes | no |

5.5x the pool. The campaign stops being **concurrency**-bound — 40 concurrent
nodes out of 1408 is noise — and becomes **charge**-bound, which is a much
better problem: it can drain over weeks on `regular` QOS instead of needing a
96-node window someone has to negotiate. `regular` caps walltime at 48 h, but
`requeueable-train.sh` already handles requeue-and-resume (measured: a requeue
costs the ~21 min dataset setup, not a partial epoch).

### The one real cost — measured: 9.4%, so take the 40 GB pool

A100-40GB is HBM2 at 1555 GB/s; A100-80GB is HBM2e at 2039 GB/s — **31% more
bandwidth** — and the two are otherwise the same 400 W SXM4 part. If the SFNO
step were bandwidth-bound, the cheap pool would also be the slow pool. The rule
set before measuring was: within ~10% on step time, take 40 GB; at 30% slower,
it becomes queue latency against charge and a judgement call.

Measured 2026-09-03, the identical config on both card types concurrently, so
the ratio is clean under shared CFS load:

| variant | 40 GB s/batch | 80 GB s/batch | ratio |
|---|---|---|---|
| 2 members, 1 step (REF-S) | 0.903 | 0.826 | **1.094** |
| 1 member, 1 step | 0.433 | 0.390 | **1.110** |
| 3 members, 1 step | 1.304 | 1.177 | **1.108** |
| 2 members, 2 steps both scored | 1.713 | 1.551 | **1.105** |

Peak memory agrees between the cards to within 4 MiB on the reference arm, as
it should.

**The ratio is 1.094–1.110 across every variant — 9.4 to 11% slower, and
stable. Take the 40 GB pool.** 9.4% more node-hours in exchange for 5.5x
the node pool and no reservation dependency is not a close call. The gap being
far below the 31% bandwidth deficit says this configuration is compute-bound
rather than bandwidth-bound, which is what `checkpointing: 3` buys — it trades
recompute for activation memory, so the arithmetic-to-traffic ratio is high.

Peak memory is identical on both cards to within 4 MiB, which is the expected
result and a useful check that nothing about the measurement was card-specific.

**The measurement nearly went wrong, and the way it did is worth recording.**
The end-to-end mean over the 70-batch probe gave a 1.72x ratio — larger than the
bandwidth deficit allows, which is what flagged it. The cause was a single
3.73 s/batch interval in the 40 GB run against a steady 0.87–0.93 elsewhere:
the `time_buffer` window refill that aug26 already documents as **bimodal, not
noisy** ("twenty steps at 17-18 s, then one interval at 163-216 s"). Over 70
batches exactly one such stall lands in one run and not the other, so an
end-to-end mean is a coin flip on whether the refill was caught. `analysis/`
reports the **median of the per-window rates** with the max beside it, so the
stall stays visible instead of being averaged into the answer.

---

## 3. A naming convention with slack in it

### The rule

    <campaign>.<realm>.<delta>.s<seed>

    sep26.atm.base.s01
    sep26.atm.crps-pure.s01
    sep26.atm.crps-pure_mem-1_noise-0.s01
    sep26.atm.roll-c2.s01

`<delta>` is the **sparse, canonically sorted** set of axes that differ from the
campaign baseline, each rendered `key-value`, joined by `_`. Axes at their
baseline value are omitted. The empty delta renders as the literal `base`.

### Why this shape

Three properties, each fixing a specific aug26 failure:

* **Sparse ⇒ adding an axis never renames anything.** A new axis simply does
  not appear in ids that do not use it. This is a property of the encoding, not
  a rule someone has to remember and a checker has to enforce. It is the whole
  reason for the change.
* **Sorted ⇒ there is no position to widen.** `crps-pure_mem-1` and
  `mem-1_crps-pure` are the same delta and render identically, so the ordering
  question that forced the second and third words cannot arise.
* **Named keys ⇒ the id is the caption.** `crps-pure_mem-1_noise-0` reads
  without a lookup table. The single-letter alphabet is not a resource that can
  run out, because there is no alphabet.

It is also *shorter* than what it replaces: `sep26.atm.crps-pure_mem-1_noise-0.s01`
is 38 characters against
`E21.aug26.atm.A0_B16_C0_L0_O5_W0_X0.D1_I0_M1_RF1_Z00.S01` at 56.

### What is dropped, and the objection to dropping it

**There is no `E##`.** The delta *is* the identity, which means a run id is
content-addressed: two people who generate the same arm get the same id, and
inserting an arm renumbers nothing. The cost is that slides and conversation
want a short handle ("E29 beats E01"), and `sep26.atm.roll-c2.s01` is not one.

Mitigation: each run carries a **stable short label** — `L01`, `R02`, `M03` —
assigned once, never reused, recorded as a `MANIFEST.tsv` column and a W&B tag,
and *not* part of the id or the directory name. Handles for humans, content
addressing for the filesystem. If a handle is ever wrong it can be changed
without moving a byte on disk, which is the opposite of aug26's situation.

**Alternative, if the team wants numbers in the path:** keep `A##.` as a
leading field (`A03.sep26.atm.crps-pure.s01`) and accept that inserting an arm
mid-block either renumbers or leaves gaps. Gaps are fine. This is the smaller
change from current practice and I will take it if preferred — but the sparse
delta word should be kept either way, because that is where the renaming
pressure actually lives.

### The axis vocabulary

Baseline = REF-S = aug26 E01. Every key below is omitted at its baseline value.

| key | baseline | levels | config site |
|---|---|---|---|
| `obj` | `crps` | `mse` | `stepper_training.loss.type` |
| `crps` | *(0.9/0.1)* | `pure` (1.0/0.0) · `energy` (0.0/1.0) · `half` (0.5/0.5) | `loss.kwargs.{crps_weight,energy_score_weight}` |
| `mem` | `2` | `1` · `3` | `stepper_training.n_ensemble` |
| `noise` | `32` | `0` · `64` | `builder.config.noise_embed_dim` |
| `ntype` | `iso` | `gauss` | `builder.config.noise_type` |
| `roll` | `f1` | `c2` (2 steps, last only) · `f2` (2 steps, both scored) · `s04` · `s20` | `n_forward_steps` + `optimize_last_step_only` |
| `fdcrps` | `0` | `1` · `3` (levels, weight 0.1) | `loss.kwargs.finite_difference_crps_*` |
| `alpha` | `100` | `095` | `loss.kwargs.almost_fair_crps_alpha` |
| `ep` | *(campaign default)* | any integer | `max_epochs` |
| `init` | `scratch` | a run id to warm-start from | `parameter_init.weights_path` |

Every key is atmosphere-only by construction; there is no ocean in this
campaign, so the "raise on an ocn run" guard aug26 needs does not apply.

Guards to carry over from `check_campaign.py` verbatim, because each cost
someone a run:

* `noise-0` must also set `noise_type: gaussian`. Isotropic noise at zero
  channels calls an inverse SHT on a zero-channel tensor and dies in the MKL
  FFT. Verified in aug26, reproduced.
* No `/pscratch/` in a generated file; warm starts carry a **run id** in the
  `.env` and `OVERRIDE_ME_WARM_START` in the yaml, resolved at submit time with
  a refusal if the parent checkpoint is absent.
* Inference IC count must divide the rank count, checked on a login node.
* Every weighted inference block's whole trajectory must stay out of the
  validation window. The held-out `5yr_test` block carries `weight: 0.0` and
  starts in 2040, past the 2000–2040 training window.
* `runs/` must be byte-identical whoever regenerates it: no username, no
  scratch path, no timestamp.

New guard this campaign needs (see §4): a non-default `crps`, `fdcrps` or
`alpha` on an `obj-mse` run is a lie in the id, because `LossConfig.build`
drops every `EnsembleLoss` kwarg for `MSE`.

---

## 4. BLOCKER: two aug26 runs cannot train, and it constrains this campaign too

**Verified today twice over: by direct call, and then in the real training path
on a GPU node.**

The second one is what makes this a production failure rather than a code
reading. Both arms were run: E01's config with `n_ensemble: 1` (E25's case) and
with `n_ensemble: 3` (E26's case), nothing else changed, on 4 GPUs, on **both**
card types. All four combinations raise identically and none reaches a single
training step:

| arm | 40 GB | 80 GB |
|---|---|---|
| `n_ensemble: 1` — E25 | `NotImplementedError`, 0 steps | `NotImplementedError`, 0 steps |
| `n_ensemble: 3` — E26 | `NotImplementedError`, 0 steps | `NotImplementedError`, 0 steps |

The traceback, from the one-member run:

    INFO - Number of trainable model parameters: 456223488
    INFO - Starting Training Loop...
    [rank0]   fme/core/loss.py:907  in forward
    [rank0]   fme/core/loss.py:255  in __call__
    [rank0]   fme/core/loss.py:452  in forward
    [rank0]   fme/core/loss.py:767  in forward      <- EnsembleLoss
    [rank0]   fme/core/loss.py:628  in forward      <- EnergyScoreLoss
    [rank0]   fme/core/ensemble.py:81 in get_energy_score
    [rank0]     raise NotImplementedError(
    [rank0] NotImplementedError: Energy score is written here specifically for
            2 ensemble members, got 1 ensemble members.

Note where it dies: **after** config validation, **after** dataset
construction, **after** the model is built and its parameter count logged, on
the first batch of the training loop. So `fme.ace.validate_config` cannot catch
it, and neither can anything else that stops short of a real forward pass. In
production that is a run which claims 4 nodes, pays the full ~22 min dataset
setup, and dies at step 1 — twice over, for E25 and E26.

The direct-call probe, for the record:

`fme/core/ensemble.py:80` opens `get_energy_score` with
`if gen.shape[1] != 2: raise NotImplementedError`. `EnergyScoreLoss.forward`
(`fme/core/loss.py:628`) calls it unconditionally, and `EnsembleLoss.forward`
(`:766`) calls `EnergyScoreLoss` whenever `energy_score_weight > 0`.
`config-train-atm.yaml` sets `energy_score_weight: 0.1`.

Probed directly on CPU:

    energy_score n_ens=1: NotImplementedError: Energy score is written here
                          specifically for 2 ensemble members, got 1
    energy_score n_ens=2: OK  torch.Size([2, 4, 5])
    energy_score n_ens=3: NotImplementedError: ... got 3

So **every `D0` arm inherits a hard requirement of exactly two training
members**, and aug26's **E25 (`M1`) and E26 (`M3`) raise on their first
training step**. `check_campaign.py` passes them because it verifies that a
config agrees with its run id, not that the config runs. The test suite does
not catch it either: every `EnergyScoreLoss` test in `fme/core/test_loss.py`
uses two members, and the one `n_ensemble = 3` test there goes through
`MSELoss`.

Consequences, in order of who should hear about them:

1. **aug26 is about to burn ~1,150 node-hours on two runs that crash at step 1.**
   E25 and E26 are queued at P6. They should be dropped or fixed before that
   window opens. This is the most urgent thing in this document and it is not a
   sep26 problem.
2. Two rows of aug26's decision table are unreachable as written — "the
   ensemble objective is what does it, not the noise" (via E25) and "more
   members help" (via E26).
3. `EXPERIMENTS.md` asserts in two places that at one member "the objective
   becomes 0.9 × MAE + 0.1 × spectral L1: still a deterministic loss". It does
   not; it raises.
4. **Any sep26 arm at one or three members hits the same wall**, which is most
   of the mechanism block in §5.

### The two ways out. Both, in this order.

**(a) Re-anchor on pure CRPS. No code change, available today.**
`EnsembleLoss.forward` gates the energy score on `energy_score_weight > 0`, so
`crps-pure` (1.0 / 0.0) skips it entirely and **any** member count runs. The
member sweep becomes `crps-pure_mem-1` → `crps-pure` → `crps-pure_mem-3`,
anchored on a pure-CRPS control rather than on REF-S. The cost is that those
arms are two factors from REF-S rather than one, so they answer "does the member
count matter to a pure CRPS objective" — narrower, still publishable, and it is
what makes §5's mechanism block runnable at all.

**(b) Generalize `get_energy_score`. Upstream, its own PR.**
Target term as the mean over members; internal term as the mean over unique
unordered pairs, following `get_crps`'s existing
`torch.triu_indices(n_ens, n_ens, offset=1)` pattern; zero internal term at one
member, mirroring `get_crps`'s `n_ens == 1` branch; a clear error at zero.

**The trap:** the current two-member code pulls the 0.5 out of the pairwise
term, because a two-member mean over one pair makes it cancel. A naive
generalization *changes the M2 number* and silently invalidates REF-S and every
comparison against it. The PR must pin the two-member value against the current
implementation with a regression test, plus forward-and-backward tests at one
and three members.

This is an `ai2cm/ace` change, not a config change. It is small, it is
well-scoped, and it should not be smuggled into a campaign branch.

---

## 5. The design: three questions with replication, not eleven without it

This is the part I would change most from the plan being replaced, and the
argument is about the decision rule rather than about any single arm.

The campaign's rule is: *a single-seed arm counts only if it falls outside its
parent's three-seed spread on the same metric at the same epoch.* That is a
sound triage rule. But three seeds is enough to **see** a spread and not enough
to **estimate** one, and an arm that lands inside the spread yields no
conclusion at all — not "no effect", just "no measurement". Eleven single-seed
arms against a three-seed parent is therefore eleven chances to learn nothing.

For a campaign whose stated purpose is *ablation* — attributing a known
system-level difference to its parts — the currency is not the number of axes
touched. It is whether each difference is separable from noise. Two levers
change that ratio without buying more nodes:

* **The cheap arms are genuinely cheap.** A one-member arm is `rel 0.50`. Four
  seeds of a one-member arm cost less than two seeds of a three-member one.
* **Epochs may be the wrong unit.** See Tier 0 below: if arm *ranking* at
  epoch 12 predicts ranking at epoch 30, the campaign runs at 12 epochs and
  every arm gets 2.5x the replication for the same charge. That is a free
  measurement off aug26's existing per-epoch diagnostics and it should be made
  before a single sep26 node is requested.

So the proposal is tiered, and Tier 0 gates the rest.

### Tier 0 — five findings that cost zero GPU-hours. Do these first.

Verified today that the inputs exist on disk.
`$PSCRATCH/aug26/E01.…S01/output/` contains `val/epoch_NNNN/` for every epoch
and `5yr_test/epoch_NNNN/`, `inference/epoch_NNNN/` every third epoch, each
holding `time_mean_diagnostics.nc`, `power_spectrum_diagnostics.nc`,
`histogram_diagnostics.nc`, `enso_index_diagnostics.nc`,
`mean_step_20_diagnostics.nc` and `annual_diagnostics.nc`. Per-epoch
checkpoints (`ckpt_0001.tar` … , plus `ema_*`) are there too.

1. **Does the arm ranking at epoch 12 predict epoch 30? MEASURED 2026-09-03.
   The answer is no — do not shorten the campaign.** Run in full below under
   "Measurements"; the summary is that the metric which says "run short" is the
   wrong metric.

   On **one-step validation** the case for running short looks strong: REF-S's
   three-seed spread is under 1.5% of the mean at every epoch (0.99% at epoch 1,
   0.36% at epoch 10) and the four-arm ordering of the comparable `A3_C1` family
   is *identical at every epoch*, Spearman ρ = 1.000 against the deepest epoch.

   On the **held-out 5-year rollout** — the metric the decision rule actually
   uses — the same four arms reorder (ρ = 0.800 at epochs 3 and 6, reaching
   1.000 only at epoch 9), and the three-seed spread does **not** narrow
   monotonically: 3.50% at epoch 3, 3.26% at 6, 1.00% at 9, 3.18% at 12, 1.65%
   at 15. Arm differences among comparable arms are 0.040–0.050 at epochs 3–6
   against a seed spread of 0.019–0.022, i.e. a discrimination ratio of
   **1.8 to 2.7**. The ratio is 29 at epoch 9, but that is one epoch's low
   spread, not a trend — epoch 12 puts it back to 3.18%.

   Three consequences, and they matter more than the saving would have:
   * **Do not run at 12–15 epochs.** The compute saving is real; the loss of
     discrimination is worse. Keep 30.
   * **The decision rule is reading a spread that fluctuates 3x between
     adjacent scored epochs.** "Outside the parent's three-seed spread at the
     same epoch" will call the same arm significant at epoch 9 and not at epoch
     12. It has to become "outside the spread pooled over the last *k* scored
     epochs" (or the max over them), or the rule is a coin flip dressed as a
     threshold.
   * **Buy seeds, not arms.** With a 1–3.5% seed band on the decision metric, a
     single-seed arm needs a large effect to clear it. This was an argument in
     the previous draft of this file; it is now a measurement.

   Note the LR-schedule caveat still stands and now cuts the other way: the
   cosine schedule runs to 30 epochs, so these are *interrupted* reads. A real
   15-epoch run would anneal properly and might discriminate better than an
   epoch-15 read of a 30-epoch run. That is a further experiment, not a reason
   to assume it.
2. **Compute-matching downward instead of upward.** The plan being replaced adds
   a 60-epoch deterministic control (`T060`) because at equal epochs the
   one-member pole gets half the FLOPs of the two-member pole. But the matching
   can go the other way for free: REF-S at epoch 15 is FLOP-matched to REF-D at
   epoch 30, and both will be on disk. If REF-S@15 already beats REF-D@30, the
   per-FLOP claim is made with **zero** new compute. Only if it does not does the
   60-epoch run become necessary. Same LR-schedule caveat, and it makes the
   free version a *bound* rather than an equivalence — which is exactly the
   right decision structure: cheap test first, expensive run only if the cheap
   test is inconclusive.
3. **The deterministic pole's CRPS is its MAE, exactly.** Verified today:
   `get_crps` zeroes the pairwise term at one member, and
   `crps(n=1) == MAE` returned `True`. One scalar off an existing checkpoint, no
   ensemble harness. **Label it the *degenerate* CRPS and do not confuse it with
   Gneiting et al. 2025's *potential* CRPS**, which is what a point forecast
   achieves *after* EMOS-style postprocessing dresses it into a distribution and
   is scored out of sample. The degenerate number is a free lower bound on the
   deterministic pole and is maximally unfavourable to it; a comparison that
   quotes only the degenerate CRPS and concludes the stochastic model wins on
   probabilistic skill is not defensible.
4. **Spectral tails, from files that already exist.** `power_spectrum` runs
   inline on the 38 plotted channels and lands in
   `power_spectrum_diagnostics.nc` per scored epoch — confirmed present. Define
   the tail metric as the generated/target ratio of power integrated over the
   top decile of wavenumbers, per variable, at the time mean and at fixed
   rollout steps. No new run, no new rollout.
5. **A three-member multi-model ensemble and a 16-member lagged ensemble
   already exist on both poles.** REF-S S01–S03 and REF-D S01–S03 were bought as
   error bars and are exactly the "train with different seeds" ensemble; the 16
   staggered inference ICs are a lagged ensemble on top. Exhaust both before
   anyone writes a bred-vector harness.
6. **One-step CRPS, SSR bias and ensemble-mean RMSE are already logged every
   epoch — in W&B, not on disk.** The validation aggregator's `ensemble_denorm`
   metric is active for every `n_ensemble > 1` run, so REF-S already has a
   per-epoch one-step CRPS and spread–skill trace. Checked: those scalars are
   *not* in `val/epoch_NNNN/mean_diagnostics.nc` (175 variables, none matching
   `crps`, `ssr` or `ensemble`) — only `mean`, `mean_norm` and `power_spectrum`
   diagnostics are written per epoch. So this one is a W&B export, not a netCDF
   read. It gives the one-step calibration trace for free; the *rollout* SSR
   still needs pass 1.

   **Does stochastic training improve the mean response?** (Berner et al. 2017,
   noise-induced transitions) is answered by three quantities from one pass-1
   sweep: REF-S's `ensemble_mean_rmse`, REF-S's single-member RMSE, and REF-D's
   single-trajectory RMSE. If the ensemble mean beats both, the gain is in the
   averaging; if REF-S's single member also beats REF-D, the gain is in the
   weights. One eval pass, no training.

### Tier 1 — the mechanism block: decompose REF-S − REF-D

REF-S − REF-D moves three things at once (loss family, noise conditioning,
member count) and no existing arm separates them. Hold the member count at one
and cross the other two:

|  | `noise-0` | noise wired (32) |
|---|---|---|
| **MSE** | **REF-D**, 3 seeds, free | `obj-mse_mem-1` |
| **pure CRPS** (= MAE at one member) | `crps-pure_mem-1_noise-0` | `crps-pure_mem-1` |

At one member the CRPS pairwise term is identically zero, so the lower row is
MAE. That is the point: the **row** difference is loss geometry (L1-type against
L2-type), the **column** difference is noise conditioning with nothing in the
objective rewarding it. Two independent estimates of the noise main effect, and
their difference is the interaction.

Then the member sweep on that same pure-CRPS family — the runnable replacement
for the aug26 arms that cannot train:

    crps-pure_mem-1  ->  crps-pure  ->  crps-pure_mem-3

`crps-pure` doubles as the two-member cell and as Tier 2's "what does the 0.1
energy-score term buy" arm.

Cost model, so the `rel` column below is reproducible. REF-S measured 63.6 h of
training + 14.2 h of inline inference + ~3 h of setup = 81 h on 4 nodes for 30
epochs. `rel` scales the *training* term only, because inline inference is a
single-member rollout: **verified** that neither `inference` block in
`config-train-atm.yaml` sets `n_ensemble_per_ic`, so it defaults to 1 and is
independent of `stepper_training.n_ensemble`. Hence

    run_h  ~=  63.6 * rel  +  17

giving 49 h at rel 0.50, 81 h at 1.00, 102 h at 1.33, 112 h at 1.50, 144 h at
2.00. Multiply by 4 nodes for node-hours. This is also why the ensemble rollout
metrics are logged by no current run — `EnsembleMetricConfig` needs more than
one member per IC to build.

| id (seed omitted) | rel | run h | node-h | isolates |
|---|---|---|---|---|
| `obj-mse_mem-1` | 0.50 | 49 | 196 | noise conditioning under MSE |
| `crps-pure_mem-1_noise-0` | 0.50 | 49 | 196 | loss geometry, no noise |
| `crps-pure_mem-1` | 0.50 | 49 | 196 | noise conditioning under MAE |
| `crps-pure` | 1.00 | 81 | 324 | pure CRPS at 2 members |
| `crps-pure_mem-3` | 1.50 | 112 | 448 | three members, pure CRPS |

5 runs, 1 seed each: **1,360 node-hours.** Because four of the five are
`rel ≤ 1.0`, this is the block where extra seeds are affordable.

**aug26's `check_campaign.py` forbids two of these cells by name.** It refuses
`Z00` with `D0` ("scores a degenerate ensemble at full ensemble cost") and
noise with `D1/M1` ("noise conditioning is wired up but nothing in the objective
can reward using it"). Both guards are right about the waste — and **the waste
is the mechanism probe.** Carry them over as warnings behind an explicit
`allow_degenerate` field that the generator writes into the run notes and the
`.env`, so the intent is recorded in the artifacts rather than in someone's
memory. Do not delete them: they are the only thing between the campaign and an
accidental 49-hour no-op.

### Tier 2 — the objective internals, all at two members off REF-S

Each is one config line and none needs the upstream fix, because all sit at two
members.

| id | rel | run h | node-h | isolates |
|---|---|---|---|---|
| `crps-pure` | 1.00 | 81 | 322 | *(shared with Tier 1)* what the 0.1 energy term buys |
| ~~`crps-energy`~~ | — | — | — | **blocked** — see §4; it cannot train |
| `fdcrps-1` | **1.00** ✓measured | 81 | 322 | spatially pooled CRPS as a training objective |
| `alpha-095` | 1.00 | 81 | 322 | almost-fair CRPS |
| `ntype-gauss` | **1.00** ✓measured | 81 | 322 | the noise's **spatial correlation** |

3 new runs: **~966 node-hours**, with `crps-energy` blocked.

**`fdcrps-1` and `ntype-gauss` both cost nothing measurable, and the noise floor
is the finding.** Guessed at ~1.05 and ≤1.00 respectively; measured at 0.870 and
0.886 (`fdcrps-1`, two nodes) and 0.871 (`ntype-gauss`) — against an **E01
baseline that itself measured 0.903 and 0.868 on two 40 GB nodes**, a 4.0%
spread on an identical config. Every difference is inside the baseline's own
variation.

So this probe — medians over ~5 windows of 10 batches — resolves nothing below
~4%, and both axes are 1.00 ± 0.04. Reading `fdcrps-1` as 0.96 because one run
beat one baseline would be reading noise: extra work does not make training
faster. It also means the prediction that gaussian noise would be *cheaper*
(one fewer inverse SHT per step) is unconfirmed — the SHT is not a resolvable
fraction of a step at this precision.

**`ntype-gauss` has a same-node measurement too**: 0.883 against that node's
0.903 baseline, and 0.871 on the other node — a 1.0% difference, inside the
floor. Its two measurements span only 1.4%, tighter than the baseline's own
4.0%, so the arm is if anything better determined than the thing it is compared
against.

**On the precision of the `rel` table.** The three-decimal figures are good to
about ±2%, not to their last digit. The baseline varies 4% *between nodes*, but
that variation largely cancels in a same-node ratio — which is why each `rel`
was computed against its own node's baseline, and why the two card types then
agree on every one to ~1% (M1 0.480/0.472, M3 1.444/1.425, `c2` 1.216/1.205,
`f2` 1.897/1.878). Read 0.476 as distinguishable from 0.50, not from 0.48.

**`fdcrps-3` was measured too, and it is free as well**: 0.878 and 0.880 on the
two 40 GB nodes, 0.801 on the 80 GB one, against `fdcrps-1`'s 0.870/0.886/0.795.
So *three* coarsening levels cost no more than one, and neither costs more than
none. The finite-difference CRPS axis is free at every level this campaign
defines, which removes the cost argument for keeping `fdcrps-3` out of the run
list — if it is excluded it should be on scientific grounds, not budgetary
ones. (`FiniteDifferenceCRPSLoss` recurses on `avg_pool2d` coarsenings, each a
quarter the size of the last, so the added work is a geometric series against a
456 M-parameter SFNO forward. Cheap is the expected answer; it is now the
measured one.)

Two of these are more interesting than the plan being replaced credits:

* **`crps-energy` (crps 0.0 / energy 1.0) was scoped *out*, and it should be
  in.** The energy score is computed in *spectral* space through an SHT
  (`EnergyScoreLoss.forward` calls `self.sht` on both arguments before
  `get_energy_score`), so it is not a reweighting of CRPS — it is a different
  objective in a different basis. For a model whose selling point is spatial
  structure, "score the spherical-harmonic coefficients, not the grid" is the
  sharp test, and at two members it runs today with no fix.
  Two asymmetric consequences worth a comment in the checker, because they are
  easy to get backwards: `almost_fair_crps_alpha` becomes **inert** (it only
  parameterizes the CRPS module, which `forward` gates on `crps_weight > 0`), so
  `crps-energy` with a non-default `alpha` is a lie in the id and must be
  rejected; but `finite_difference_crps_weight` does **not** become inert
  (`forward` gates it on `self.diff_crps_loss is not None`, i.e. on its own
  weight alone), so `crps-energy_fdcrps-1` would silently run a
  pooled-CRPS-plus-energy objective. Force `fdcrps-0` under `crps-energy`.
* **`ntype-gauss` is the arm the source deck did not think to ask for.** The
  deck varies the noise *width*; the noise *type* — isotropic, drawn in
  spherical-harmonic space through an inverse SHT, against gaussian, i.i.d. per
  grid point — is a modelling claim about the perturbation's spatial
  correlation, and nothing has compared them. `noise_type` is
  `Literal["isotropic", "gaussian"]` at `fme/ace/registry/stochastic_sfno.py:274`,
  default `gaussian`; REF-S sets `isotropic`. Expect it marginally *cheaper* —
  it drops an inverse SHT per step — so quote ≤1.00 and measure.

`fdcrps-1` is `FiniteDifferenceCRPSLoss`, which computes CRPS on the field and
then on successively average-pooled coarsenings — structurally the spatially
pooled CRPS of Alet et al. 2025, used as a training objective rather than a
diagnostic. Budget it at 1.05 and **measure** rather than trust: the extra
pooling is cheap against a 456 M-parameter SFNO forward, but nobody has timed it
here. The `fdcrps-3` (three levels) variant is recorded as not-run.

### Tier 3 — separating "two steps" from "two scored steps"

| id | rel | run h | node-h |
|---|---|---|---|
| `roll-c2` | 1.33 | 102 | 408 |

`roll-c2` is 2 forward steps with `optimize_last_step_only: true`. With REF-S
(1 step), `roll-c2` (2 steps, 1 scored) and aug26's E18 (2 steps, both scored)
the two moves separate: `roll-c2` − REF-S is rollout length alone, E18 −
`roll-c2` is the second scored step alone. Cost is 1.33, not 2.00, because the
unscored step is a `no_grad` forward and a forward is ~1/3 of a training step —
the same arithmetic aug26 already uses for its sampled schedules.

**Risk:** E18 is 144 h and does not fit any window aug26 has had; it may never
run. Without it, `roll-c2` − REF-S is still a clean "rollout length" result, so
this arm does not depend on E18 — but the three-point reading does.

### What Tier 1–3 costs, and the seeds trade

| | runs | nodes | node-hours | critical path |
|---|---|---|---|---|
| Tier 1 | 5 | 20 | 1,360 | 112 h (`mem-3`) |
| Tier 2 (new) | 4 | 16 | ~1,308 | ~84 h |
| Tier 3 | 1 | 4 | 408 | 102 h |
| **total, 1 seed, 30 epochs** | **10** | **40** | **~3,080** | **112 h** |
| ~~same at 15 epochs~~ | ~~10~~ | ~~40~~ | ~~~1,540~~ | ~~56 h~~ |
| **recommended: drop Tier 3 + `alpha-095`, add 3 seeds on the four Tier-1 M1 cells** | 14 | 56 | **~2,940** | 112 h |

**Tier 0.1 ruled out the 15-epoch rows** — see above; the rollout metric does
not support them. So the replication has to come out of the arm list rather
than out of the epochs, and the recommended row makes that trade explicit: drop
the two weakest single-seed arms (`roll-c2` at 408 node-h, which also depends on
an E18 that may never run, and `alpha-095`, whose most likely outcome is a null)
and spend the ~730 node-hours on second and third seeds for the four
one-member mechanism cells, which are the cheapest runs in the campaign at
196 node-h each.

That buys the thing Tier 0.1 says is scarce: a factorial main effect with an
error bar on it, rather than four point estimates against a band that moves 3x
between epochs. Same charge, one fewer axis, and the one block that can actually
support a claim.

At 40–56 concurrent nodes out of 1408 hbm40g, none of these rows is
concurrency-bound. Charge is the constraint, and the allocation balance still
needs checking (`iris` returned 403 for me; check it on a login node or the
NERSC portal).

---

## 6. Decision rules for the new arms

Parent is REF-S unless stated. Same triage rule: an arm counts if it falls
outside its parent's three-seed spread on the same metric at the same epoch.

| claim | what would have to be true |
|---|---|
| the loss **geometry** does the work | `crps-pure_mem-1_noise-0` beats REF-D outside REF-D's spread. MAE against MSE, one member, no noise on either side. This is the arm aug26's E25 was supposed to be and could not run |
| noise conditioning does something on its own | `obj-mse_mem-1` − REF-D and `crps-pure_mem-1` − `crps-pure_mem-1_noise-0` agree in sign and both fall outside the reference spread. Either alone is a single-seed difference; the pair is a factorial main effect. If they disagree in sign, report the interaction and claim nothing about the main effect |
| more members help a pure-CRPS objective | `crps-pure_mem-3` beats `crps-pure` outside its spread. Narrower than aug26's E26 question because the family is pure CRPS, not 0.9/0.1 — say which control was differenced against |
| the 0.1 energy-score term earns its place | `crps-pure` is **worse** than REF-S outside REF-S's spread. A null says the term is decoration at that weight, which is a real result and worth reporting as one |
| the spectral basis is what matters | `crps-energy` beats REF-S on the spectral-tail metric specifically. A win on `time_mean/rmse/channel_mean` alone does not support this |
| spatial structure needs a pooled objective | `fdcrps-1` beats REF-S on the spectral tail. A win on the channel mean with a flat tail is a different claim |
| almost-fair CRPS at two members is worth it | `alpha-095` beats REF-S outside its spread. At two members the pairwise term is a single pair, which is where almost-fair is supposed to pay; a null says estimator noise is not binding at this scale |
| the noise's spatial correlation matters | `ntype-gauss` differs from REF-S outside its spread. A null says isotropic buys nothing over i.i.d. for this model — worth a sentence in any paper using `NoiseConditionedSFNO` |
| rollout length helps, independently of scoring more steps | `roll-c2` beats REF-S outside its spread. If `roll-c2` sits with REF-S and E18 does not, the second *scored* step is the whole effect and `RF2` should not be described as a rollout arm |
| stochastic beats deterministic **per FLOP** | REF-S@ep15 beats REF-D@ep30 outside both spreads (Tier 0.2, free), or a 60-epoch REF-D loses to REF-S. Until one of those lands, **every caption says "at equal epochs"** |

**On uncertainty.** The triage rule is a triage rule, not a significance test.
For any number that goes in a manuscript, report an effect size with a paired
block bootstrap over initial conditions and time blocks, and keep model-seed
variability as a *separate* level rather than pooling it with sampling error.
Three seeds is enough to see a spread and not enough to estimate one; say so.

---

## 7. Evaluation: what to build, and what not to

The plan being replaced is right that evaluation is the gap: nothing in aug26
computes a return period, an economic value, a spread–skill ratio at rollout, or
a spectral tail. Two structural points about how to close it.

**Offline, not inline.** `InlineInferenceConfig.n_ensemble_per_ic` would give a
rollout ensemble with one config line, but putting that line in the training
configs means regenerating and re-running the references — and the whole point
of `checkpoint_save_epochs: {step: 1}` is that every epoch of every run is
already on disk. Use `python -m fme.ace.evaluator` /
`InferenceEvaluatorConfig`, which carries the same `n_ensemble_per_ic` plus two
things the inline path does not: a `seed` (so a stochastic module produces a
reproducible noise sequence independent of `forward_steps_in_memory` — without
it, two arms are compared across different noise draws as well as different
weights) and a `data_writer` (netCDF trajectories, which the offline metrics
consume).

**Two passes, because storage is the constraint.**

*Pass 1 — scores only.* No prediction or monthly files. 16 held-out 2040s ICs,
`n_ensemble_per_ic: 8`, 5-year rollout. Configure an `ensembles` **ladder** —
step 1, 4, 20, 120, 1460 (6 h, 1 day, 5 days, 30 days, 1 year) — rather than the
single default step 20, with `strict: true` on the steps that must exist so a
typo is loud rather than warned-and-skipped, and `target: "norm"` on at least
one (`channel_mean` is only logged for normalized data). Matching `step_means`
ladder. `log_mean_maps: false` everywhere — the one-step ensemble-mean maps were
174 PNGs per atmosphere epoch and 47% of that run's map bytes. Enable
`enso_coefficient`; leave `ipo_index` off, it needs >80 years. Drop
`forward_steps_in_memory` from 20 to 5, because `broadcast_ensemble` puts 8
members in the batch dimension of each rank. ~1.2 h on 4 nodes per arm including
the 22.5 min dataset setup; sixteen arms is under 100 node-hours — negligible
against training.

**One gap to close in the generator:** `InlineInferenceConfig.__post_init__`
checks `n_initial_conditions % world_size` and raises readably, but
`InferenceEvaluatorConfig` has **no such check** — a mismatch surfaces as a bare
`AssertionError` inside the data loader, minutes into an allocation. The eval
config generator must do that arithmetic itself on a login node. 16 ICs covers
4, 8 and 16 ranks.

*Pass 2 — trajectories.* Only for arms getting the extremes analysis; start with
REF-S and REF-D, three seeds each. **`TimeCoarsenConfig` is `block_mean` only**,
so daily coarsening destroys the daily maxima a heatwave return period is
computed from. Extremes come off the uncoarsened 6-hourly stream on a narrow
`prediction_names` list — `Tat2m`, `surface_precipitation_rate`, `PS`, and
`Uat10m`/`Vat10m` for a cyclone proxy — with `time_coarsen` reserved for
variables where a mean is the quantity of interest. At 4 ICs × 8 members, 5
years 6-hourly, 3 variables on the 180×360 grid that is ~180 GB per arm; cap the
eval campaign at ~0.5 TB against 65 TB free of the 120 TB `$PSCRATCH` quota
(`lfs quota -h -u $USER /pscratch` on a **login** node).

### The offline metrics, and the two traps that decide the result

Not `fme` aggregators — a separate upstream PR if they ever should be.

* **Spread–skill ratio** and **short-vs-long skill** are the pass-1 ladders.
  Nothing to implement; `SSRBiasMetric` already does the bias correction and the
  `prescribed`-channel zeroing.
* **Spectral tails** are Tier 0.4 — a read of files that exist.
* **Spatially pooled CRPS** reuses `_get_finite_difference_crps_loss` as an
  evaluation function. Note the inline CRPS metric uses `alpha=0.95` while
  REF-S's training loss uses the default `alpha=1.0`; keep the evaluation alpha
  fixed across arms and state it.
* **Relative economic value** (Richardson 2000; Alet 2025). For event
  "variable exceeds quantile q" and cost/loss ratio r, act when p ≥ r; with hits
  h, false alarms f, misses m over n cases and base rate s = (h+m)/n:
  `E_f = r(h+f)/n + m/n`, `E_c = min(r, s)`, `E_p = r·s`,
  `V(r) = (E_c − E_f)/(E_c − E_p)`. Plot V(r) for both poles on one axis. **Say
  in the caption that the deterministic arm's p is an indicator in {0,1}**, so
  its V(r) is a single decision threshold while the ensemble's is a family of
  them — otherwise the structural gap reads as skill.
* **Return periods.** Block maxima of `Tat2m` and
  `surface_precipitation_rate`, GEV by L-moments (more stable than MLE at these
  sample sizes), 10/20-year return levels as a bias map against E3SM's own
  maxima over the same held-out years, plus regional QQ plots. Two traps, and
  either one is the whole result:
  * *Sample count.* The emulator has `n_traj × n_years` block maxima and E3SM
    has `n_years`. More samples constrain the GEV better, so the emulator looks
    better for a reason unrelated to the model. Either subsample to E3SM's count
    or report bootstrap intervals at both counts. Say which, in the caption.
  * *Independence.* Pass 1 is 16 ICs × 8 members × 5 years = 640 model-years,
    but not 640 independent samples: the ICs come from one 2040s window of one
    forced trajectory, members within an IC share an initial state, and every
    trajectory shares the same prescribed SST and forcing path. **Do not quote a
    50-year return level** until the effective sample size has been estimated;
    prefer 10- and 20-year levels. Longer return periods need more independent
    *start windows across the record*, which is a different pass-1
    configuration, not a longer rollout or more members.
* **Calibration, which the source deck does not ask for and should.**
  Spread–skill ratio is a first moment and hides shape. Add rank histograms
  (PIT for the continuous case), reliability diagrams, and coverage of nominal
  prediction intervals, by variable and lead time. Cheap, nothing in `fme`
  computes them, and they are what distinguishes "well spread" from "right for
  the wrong reason".
* **Modes of variability.** ENSO: `enso_index` and `enso_coefficient` exist;
  enable both in pass 1. MJO: not in `fme` — offline from pass-2, 20–100 day
  bandpass of tropical `FLUT` and `U_*`, EOF1/2, phase–amplitude diagram. Both
  fields are outputs of this configuration.

Atmospheric rivers, sudden stratospheric warmings and tropical cyclones each
need a detection algorithm. **Out of scope**, say so. `T_0`/`T_1` make an SSW
proxy cheap if one arm of it is wanted — index 0 is the top of the atmosphere,
verified today off the campaign's own `centering.nc`: `T_0` = 220.95 K,
`T_1` = 208.79 K (the coldest level, i.e. the tropopause), `T_7` = 277.47 K,
`Tat2m` = 278.62 K.

Two small things not to rebuild: `EnsembleMetricConfig.log_mean_maps` already
defaults to `False`, so setting it is belt-and-braces against the
aggregator-level default rather than a fix; and `strict` already defaults to
`False`, which is why an over-long ladder step is skipped with a warning
instead of failing the job.

---

## 8. Explicitly out of scope, with a reason each

* **Hybrid mean + latent stochastic residual** (Kossaifi et al. 2026). Not
  expressible in this config space; it needs a step type composing a
  deterministic module with a noise-conditioned residual. A PR against the step
  registry, not an axis level.
* **A noise-off inference mode**, for evaluating a stochastic checkpoint
  deterministically. `StepperOverrideConfig` carries only `ocean`, `multi_call`,
  `derived_forcings` and `prescribed_prognostic_names`, so this needs a
  `noise_scale` hook in `NoiseConditionedModel.forward`. Small, well-scoped,
  and still a code change — propose it, do not slip it in. **Worth proposing:**
  it would make "does the model's spread respond to a post-hoc noise scale"
  answerable off existing checkpoints with no training at all, which is the
  cheapest calibration lever available.
* **Bred vectors / IC-perturbation ensembles** for the deterministic arm. Needs
  an IC-perturbation hook in the inference loader. Exhaust Tier 0.5 first.
* **`energy_score_whitening`** (`SpectralWhitening`, `eps_frac`, `exponent`) —
  a further untested knob, one run, and a refinement of an axis this campaign
  only establishes.
* **`crps-half`, `noise-64`** — one run each, same reason.
* **`fdcrps-3`** — same, but note the cost argument is gone: it was measured at
  the same step time as `fdcrps-1` and as the baseline, so excluding it is a
  scientific choice about how many levels of pooling are worth a run, not a
  budgetary one.
* **The ocean, and anything coupled.** This campaign is atmosphere-only by
  construction. Samudra takes no noise input and its loss is plain MSE, so none
  of these axes mean anything there. (Note the coupled config's `n_ensemble: 2`
  on the ocean side already buys nothing for exactly this reason.)
* **Multi-realization training data.** The deck asks whether the model can
  capture *model* variability if trained on multiple E3SM ensemble members. The
  training data here is a single realization, `v3.LR.historical_0101.aigo`;
  there is no second historical member staged, so there is no E3SM ensemble
  distribution to compare against and nothing to train on that would answer the
  question as posed. Record it as a **data prerequisite**, not a config arm. Do
  not add a run that pretends otherwise.
* **Content hashes in the run id.** They conflict with content-addressed
  identity and duplicate what field-by-field assertion already does. Put them
  in `MANIFEST.tsv` and a JSON sidecar — resolved config, seed, git commit,
  dataset id, statistics checksums, parent checkpoint checksum — where they are
  useful for provenance and cost nothing in renames.

---

## 9. The unmerged dependency

sep26 branches from `e3sm/exps/hist-v2026.8.0`, not from `main`, and this is not
a convenience. That branch carries 22 files of `fme/` changes over main, and at
least one is load-bearing for the atmosphere config: **`time_buffer` does not
exist on `main`** (`git grep time_buffer origin/main -- fme/core/dataset/xarray.py`
is empty). `time_buffer: 10` with `time_buffer_pool_size: 2` is what took the
atmosphere from 3.155 s/batch to 0.925 s/batch — a 3.4x difference that decides
whether a run fits any window at all.

So the whole campaign, aug26 and sep26 both, rests on ~1,660 lines of
experiment-branch code. That is a risk worth naming in its own right and it is
independent of everything else here: the data-loader work should be PR'd to
`main` on a `feature/` branch regardless of what this campaign decides.

---

## 10. Open decisions

Ordered by how much they change what gets built.

1. **Run-id convention** — content-addressed sparse delta (`sep26.atm.crps-pure.s01`,
   recommended) or keep a leading `A##` field. §3.
2. **Tier 0.1 first, or run at 30 epochs regardless.** The epoch-ranking read
   costs nothing and could halve the campaign. Recommend gating on it.
3. **Seeds against arms.** 10 arms × 1 seed, or ~6 arms × 2–3 seeds for the
   same charge. For an ablation study I would take the replication. §5.
4. ~~**40 GB or 80 GB**~~ — **answered: 40 GB.** 48% memory headroom and 9.4%
   slower per step, against 5.5x the node pool and no reservation. §2. What is
   left is a preference, not a question: whether to keep a small hbm80g
   allocation for the arms whose critical path matters most.
5. **Who fixes `get_energy_score`, and when.** aug26's E25/E26 are queued and
   will crash. §4.
6. **Does REF-D (aug26 E21) actually run?** Half of Tier 1 differences against
   it, and it is behind a window that has not opened.

---

## Measurements

*(2026-09-03, this branch)*

**`get_energy_score` member support** — CPU, direct call, complex64 inputs:
raises `NotImplementedError` at 1 and 3 members, returns at 2. `get_crps` at one
member equals MAE exactly (`torch.equal` → `True`).

**Epoch stability, Tier 0.1** — a read of aug26's per-epoch diagnostics, no GPU
time. Scripts: `analysis/epoch_stability.py` (one-step `val/`) and
`analysis/rollout_stability.py` (held-out `5yr_test/`). The comparable arm
family is `A3_B16_C1_L0_O5` — E07 (`W1`), E08 (`W2`), E10 (`X1`), E15 (`W3`).
**E09 (`W4`, which zeroes `STW_0` from the loss) is excluded**: its rollout score
is 1.26–1.58 against ~0.6 for the rest, a mechanical consequence of leaving a
channel unconstrained, and it would dominate any range statistic.

*One-step `val/`, `mean_norm_diagnostics.nc`, `weighted_rmse-channel_mean`:*

| epoch | REF-S mean | 3-seed spread | spread/mean | 4-arm ρ vs deepest |
|---|---|---|---|---|
| 1 | 0.22574 | 0.00223 | 0.99% | 1.000 |
| 5 | 0.14226 | 0.00110 | 0.77% | 1.000 |
| 9 | 0.12362 | 0.00085 | 0.69% | 1.000 |
| 10 | 0.12073 | 0.00044 | 0.36% | — |

*Held-out `5yr_test/`, `mean_step_20_norm_diagnostics.nc`, same metric:*

| epoch | REF-S mean | 3-seed spread | spread/mean | 4-arm ρ | 4-arm range | range/spread |
|---|---|---|---|---|---|---|
| 3 | 0.63441 | 0.02219 | **3.50%** | 0.800 | 0.0401 | **1.8** |
| 6 | 0.57418 | 0.01873 | **3.26%** | 0.800 | 0.0502 | **2.7** |
| 9 | 0.53797 | 0.00537 | 1.00% | 1.000 | 0.1577 | 29.4 |
| 12 | 0.51207\* | 0.01627\* | **3.18%** | — | — | — |
| 15 | 0.49538\* | 0.00818\* | 1.65% | — | — | — |

\* two seeds only; S01 is behind S02/S03.

The time-mean bias maps say the same thing more bluntly — at epoch 3 the
three-seed spread of the area-weighted `PS` bias RMSE *equals its mean*
(124 ± 124 Pa), narrowing only to 98.6 ± 23.2 Pa by epoch 9; `Tat2m` goes
0.454 ± 0.21 K → 0.298 ± 0.12 K.

**Card sweep** — the 12-variant sweep on one node × 4 GPUs of each card type,
run concurrently. First point in and it settles §2: A100-40GB
(`nid001185`, 40960 MiB, 251 GB host) at 21,155 MiB peak and 0.903 s/batch
median; A100-80GB (`nid008649`, 81920 MiB) at 21,151 MiB and 0.826 s/batch.
**Ratio 1.094.** Remaining variants still running; they fill in the `rel`
estimates for `roll-c2`, `fdcrps-1` and `ntype-gauss`, which are arithmetic
rather than measurement until they land. Scripts:
`analysis/card-sweep.sh` and `analysis/steprate.py`.

For reference, aug26 measured 0.925 s/batch clean on an 80 GB card at
production scale (4 nodes / 16 ranks). This probe is 1 node / 4 ranks on a
3-year subset, so the absolute numbers are not the production ones — the
card-to-card ratio is what it is for.

Two operational notes bought the hard way and worth writing down:

* **`/tmp` is node-local on Perlmutter.** A script or config staged there is
  invisible to any other node, and a batch job that reads one fails with exit
  127 nine seconds in. Stage to `$PSCRATCH`. Three submissions died this way.
* **`log_train_every_n_batches` is a `TrainConfig` field, not a `LoggingConfig`
  one.** `--override logging.log_train_every_n_batches=10` is rejected by dacite
  (`UnexpectedDataError`) at config parse, before any GPU work, which reads as a
  silent no-op in a sweep harness that only greps for step lines.

---

## 11. Six things measured on 2026-09-03, and what each one changed

An external review of the branch at `6b74fa8` questioned what several arms
actually isolate. Every claim below was checked by running code, not by
reading it. Scripts are in `analysis/`, one per finding.

### 11.1 Every data-parallel rank draws identical noise — CONFIRMED

`set_seed` (`fme/core/rand.py:20`) gives every rank the same `torch.manual_seed`
and `torch.cuda.manual_seed_all`. The `RandomState` machinery that would
decorrelate them is reached only through `apply_config_seed`, which is called
from `inference.py` and `evaluator.py` and **never from training**. So the
conditioning noise comes from the process-global CUDA RNG, seeded identically
everywhere.

Measured at 2 ranks (`analysis/rank_noise.py`): the per-sample noise hashes are
byte-identical across ranks, for both `isotropic` and `gaussian`.

    gaussian: distinct within rank=True  identical across ranks=True  unique fields globally=2/4
    isotropic: distinct within rank=True identical across ranks=True  unique fields globally=2/4

At campaign scale this is worse than it looks. Global batch 16 over 16 ranks is
local batch 1, so a `Mn` update contains **n unique noise fields, not 16n**.
The estimator stays unbiased — noise is independent of the data — but batch size
buys no noise averaging at all, and the CRPS dispersion gradient is computed
from the same one or two latent fields for every sample in the batch.

**The fix is one line, and it was tested** (`analysis/rank_noise_fix.py`):
offset only the CUDA seed by rank. At 4 ranks, noise decorrelates and model
init stays bit-identical across ranks, which is what DDP requires.

    variant                  init identical    noise identical
    current                  True              True
    rank_offset_cuda_seed    True              False      <- wanted

**What changed:** nothing in the run list. sep26 differences five arms against
RF01, which is aug26's E01 trained under the current behaviour, so silently
fixing this would break the one control the campaign inherits. It is recorded
as upstream item B4 and as a *proposed* arm, not a taken one.

### 11.2 A seed label does not pair arms across the `Z` axis — CONFIRMED, worse than suspected

Building the same model at `noise_embed_dim` 0 and 32 under one seed
(`analysis/seed_pairing.py`): the conditioning modules are constructed inline
inside every block, so they advance the init RNG before the layers after them.

    control (32 vs 32, same seed) identical: True
    Z0 vs Z1: 22 shared tensors, 5 identical, 17 DIFFERENT   PAIRED: False
    Z1 vs Z2: 30 shared tensors, 5 identical, 25 DIFFERENT   PAIRED: False

Only 5 of 22 shared tensors survive. A different init is just another draw from
the same distribution, so the precise consequence is:

> **A contrast that changes `Z` carries a full seed's worth of noise. A contrast
> that does not is paired and carries none.**

`Z` is the only axis that touches the architecture — `D`, `G`, `M`, `N`, `Q`,
`R`, `Y` all leave the module identical. So in the LG 2×2 the **rows are paired
and the columns are not**.

**What changed:** seeds 2 and 3 on LG01–LG03 (+1,134 node-h), and NC02 (`Z2`
against `Z1`, a one-seed unpaired contrast) parked at priority 6.

### 11.3 At `Z0` the ensemble members are bit-identical — CONFIRMED

With zero noise channels the model is a deterministic function of its input and
`broadcast_ensemble` hands every member the same input. Measured on CPU, where
kernels are deterministic (`analysis/z0_degeneracy.py`):

    max|member0 - member1| = 0.000e+00     CRPS(M=2) == MAE, exactly
    energy-score dispersion term max = 0.000e+00

**What changed:** the review proposed two `M2`/`Z0` controls. This measurement
says one of them is worth running and the other is not:

* `D0_G1_…M2…Z0` (pure CRPS) — **refused.** With identical members the pairwise
  term is exactly zero, so this optimises bit-for-bit LG01's objective at twice
  the cost. Now a hard blocker in `validate()` with no opt-in.
* `D0_G0_…M2…Z0` (the 0.9/0.1 mix) — **added as LG04.** The energy score's
  target term survives even with its dispersion term zeroed, so the objective is
  genuinely distinct: MAE + 0.1 × a spectral L1 distance. It is the only arm
  that asks whether the noise helps under a loss that *can* reward dispersion.
  The wasted second member is forced by upstream: `get_energy_score` demands
  exactly two.

### 11.4 The stochastic arms start deterministic, and E01 has learned to use its noise — NEW

`ConditionalLayerNorm.reset_parameters` zeroes `W_scale_2d` and `W_bias_2d` and
sets `W_scale.bias = 1`, so at step 0 the noise pathway is an exact identity and
**every stochastic run begins as a deterministic one.** Whether it stays that
way is an empirical question nobody had asked.

Read straight out of aug26 E01's epoch checkpoints
(`analysis/noise_amplitude.py`). The 1σ modulation of a layer-norm scale is the
L2 norm of that output channel's row of noise weights:

| epoch | 1 | 3 | 5 | 8 | 11 | 13 |
|---|---|---|---|---|---|---|
| scale 1σ (mean) | 0.0277 | 0.0352 | 0.0408 | 0.0459 | 0.0488 | 0.0498 |
| bias 1σ (mean) | 0.0288 | 0.0369 | 0.0407 | 0.0428 | 0.0435 | 0.0437 |

Monotone from zero, saturating: the scale gains 41 × 10⁻⁴ between epochs 1 and
2 but 5 × 10⁻⁴ between 12 and 13, and the bias term is flat from epoch 8. The
typical channel ends at **±5.0%** and the most strongly conditioned channel at
±30%, across all 16 blocks.

So the noise pathway is real, it is used, and it is essentially converged well
inside 30 epochs. That is reassuring for the epoch budget and it is a cheap
per-epoch telemetry worth logging for every `Z1` arm: an arm whose noise weights
stay at zero has quietly become a deterministic model.

### 11.5 The energy score is per-coefficient marginal, not joint — CONFIRMED

`get_energy_score` applies a complex modulus at each spherical-harmonic
coefficient independently; nothing couples modes or channels in a norm. Measured
(`analysis/loss_semantics.py`) by permuting which member holds which value
independently at each (channel, mode) — an operation that preserves every
per-coefficient marginal and destroys the cross-coefficient dependence:

    score(original) = 305.4745483398
    score(swapped)  = 305.4745483398     <- bit-identical, elementwise
    a true joint ES with an L2 norm over all modes: 9.1198 vs 9.1367 (differs)

The score is invariant to arbitrary per-coefficient member relabelling, so it
cannot see cross-mode phase organisation or cross-channel dependence at all.

**What changed:** the `G` axis is documented as "how much weight on a per-mode
spectral score", not "joint energy scoring", and OI01's note says the two terms
are not on a common scale. No arm was dropped: the question `G` asks is still a
real one, it just is not the one the name implied.

### 11.6 Almost-fair CRPS is only almost-fair at two members — CONFIRMED

`get_crps` hard-codes `epsilon = (1 - alpha) / 2`. AIFS-CRPS (arXiv:2412.15832)
defines it as `(1 - alpha) / M`. Against the analytic definition:

| M | 2 | 3 | 4 |
|---|---|---|---|
| relative error at α=0.95 | 0 (exact) | 0.89% | 1.16% |

OI04 is `M2`, so it is valid as written. **What changed:** `validate()` and
`check_campaign.py` now refuse `Y1` anywhere but `M2`, so the arm cannot be
extended onto the member sweep without someone noticing.

### What the review got right that is now fixed in the docs, not the runs

* `LG02−LG01` and `LG03−RF02` are the noise **simple effects** under MAE and
  under MSE. Their difference is the loss-by-noise interaction. Calling them
  "two independent estimates of the noise main effect" assumed that interaction
  away; the note now says so.
* `Q` is **multiscale finite-difference CRPS** on array-index increments, not
  the spatially pooled CRPS of Alet et al. It also adds 0.1 on top of a 1.0
  objective, so its weights sum to 1.1. Renamed in every table; the Alet
  attribution is gone.
* `R1` is a **detached** two-step rollout with only the terminal state scored;
  `R2` scores both and **sums** them, raising the objective scale. Neither is
  backpropagation through time. Both notes now say this, and RO02's says to
  normalise before reading it.
* The template's "12-year trajectory" comment is wrong: `n_forward_steps: 7300`
  at 6-hourly is 5 years.

### What the review got wrong

* The second `M2`/`Z0` control is degenerate, not merely wasteful — see 11.3.
* `Q3` does not "triple the auxiliary weight" and the review's own text
  half-concedes this: `FiniteDifferenceCRPSLoss.forward` returns
  `result / self.levels`, so Q3 spreads one 0.1 across three scales.

---

## 12. The two blockers, lifted on evidence

Both upstream blockers refused configurations that raise on the **first
training batch** -- after config validation, after dataset construction, after
the model is built. That is precisely the class of fault a unit test does not
see, so passing unit tests was never going to be the standard for lifting
them. Each was re-run for real on a GPU node after its fix landed on this
branch.

| was refused | fault it used to hit | re-run result |
|---|---|---|
| `M != 2` with an energy weight | `get_energy_score` raised `NotImplementedError` | **209 steps, loss 4.0444 → 0.3705**, zero `NotImplementedError` |
| `crps_weight: 0` (`G2`) | `get_channel_losses` raised "Per-channel loss has 1 elements but 50 channel names were provided" | **250 steps, loss 1.1952 → 0.2952**, zero such errors |

Both guards are gone from `validate()` and `check_campaign.py`, and the five
tests that asserted them are now permission tests: they assert the same
configurations *build*, and carry the measured evidence in their docstrings so
the reason survives.

The `Y1`-only-at-`M2` restriction is lifted too, on **weaker evidence, labelled
as such**: epsilon now scales with the ensemble size and is checked against the
analytic AIFS definition at M = 1, 2, 3 and 5, but it was not re-run on a node.
The distinction is real — that one is a scalar coefficient inside the CRPS
term, which every `D0` arm already exercises on every batch, so there is no
unexercised branch for a first-batch fault to hide in. The other two gated
whole limbs of the loss.

### What is still refused, and why

Nothing upstream. What remains is degeneracy and truth-in-labelling:

* `Z0` + pure CRPS + more than one member — the members are **bit-identical**,
  so this is the `M1` objective at M times the cost.
* `N1` at `Z0` — no noise of either type is drawn, so the token names nothing.
* `Y1` at `crps_weight: 0` — alpha only parameterises the CRPS module, which
  `EnsembleLoss.forward` gates off entirely there.
* `D1` with any `G`/`Q`/`Y` level — `LossConfig.build` discards every
  `EnsembleLoss` kwarg for MSE, so the id would claim a setting the run
  does not have.

### What this unlocks, not yet taken

`G2` is now a runnable level, and `M1`/`M3` are runnable with an energy weight.
No arm was added: the run list is a separate decision. The one that looks
clearly worth making is **EN02**, which sits at `D0_G1_M3` — pure CRPS — only
because `G0` at `M3` used to be impossible. At `G0` it would difference against
RF01 on a single factor (member count) rather than against EN01 on two. It
renames the run, so it is the user's call.
