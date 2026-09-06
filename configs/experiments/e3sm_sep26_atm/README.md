# sep26 — E3SMv3 historical, atmosphere-only ablation

An ablation of the **training objective** for the ACE2S atmosphere. Everything
else — channels, batch size, learning rate, loss weighting, data — is held at
the aug26 E01 tuning set.

**26 runs · 104 nodes concurrent · 6,230 node-hours to P5 · 137 h critical
path.** Three more arms are parked at P6 (967 node-h) with a stated
precondition each. Built, checked, validated. Nothing queued.

W&B project **`ACE2S-sep26-atm`** (entity `e3sm-aig`) — its own, not
`SamudrACE-E3SMv3`.

---

## Run ids

    <exp>.<campaign>.<realm>.<factor word>.S<seed>
    LG01 .sep26     .atm    .D0_G1_I0_M1_N0_Q0_R0_Y0_Z0 .S01

**Experiment id** = two letters (study family) + two digits.

| | study | asks |
|---|---|---|
| `RF` | references | the two poles everything differences against |
| `LG` | loss geometry | is it the loss, the noise, or the members? |
| `NC` | noise conditioning | does the noise's shape and width matter? |
| `EN` | ensemble size | do more training members help? |
| `OI` | objective internals | what do the CRPS knobs buy? |
| `RO` | rollout | does training on longer rollouts help? |
| `CU` | curriculum | does deterministic-then-stochastic beat either? |

**Factor word** = fixed order, always written in full, alphabetical by position.
Every level is defined up front — including unused ones — so adding a level
never renames a run.

| pos | axis | levels |
|---|---|---|
| `D` | objective | `D0` EnsembleLoss · `D1` MSE |
| `G` | crps / **per-mode** spectral score | `G0` .9/.1 · `G1` 1/0 · `G2` 0/1 · `G3` .5/.5 |
| `I` | init | `I0` scratch · `I1` warm start |
| `M` | members | `M1` `M2` `M3` |
| `N` | noise type | `N0` isotropic · `N1` gaussian |
| `Q` | multiscale FD-CRPS levels | `Q0` off · `Q1` · `Q3` (weight 0.1, split across levels) |
| `R` | rollout | `R0` 1 step · `R1` 2 detached, last scored · `R2` 2 detached, both scored & summed · `R3` ≤4 sampled · `R4` ≤20 sampled |
| `Y` | almost-fair alpha | `Y0` 1.00 · `Y1` 0.95 (**`M2` only**) |
| `Z` | noise dim | `Z0` 0 · `Z1` 32 · `Z2` 64 |

Template word: `D0_G0_I0_M2_N0_Q0_R0_Y0_Z1`.

### W&B mirror

| W&B field | value |
|---|---|
| `WANDB_NAME` | the run id |
| `WANDB_RUN_GROUP` | `sep26.atm.<exp>` — seeds collapse |
| `WANDB_JOB_TYPE` | the factor word — arms group |
| `WANDB_TAGS` | campaign, realm, `<exp>`, study letters, `S##`, `P#`, **and every factor token** (`D0`, `G1`, … `Z1`) |

So "every `M1` run" is a tag filter, not a regex.

---

## The run list

| exp | word | rel | node-h | pri | what it isolates |
|---|---|---|---|---|---|
| **RF01** | `D0_G0_I0_M2_N0_Q0_R0_Y0_Z1` | — | **0** | — | stochastic pole = **aug26 E01, inherited, 3 seeds** |
| **RF02** ×3 | `D1_G0_I0_M1_N0_Q0_R0_Y0_Z0` | 0.48 | 567 | 1 | deterministic pole — MSE, 1 member, no noise |
| LG01 ×3 | `D0_G1_…M1…Z0` | 0.48 | 567 | 2 | MAE vs MSE, no noise either side. **Paired** with RF02 |
| LG02 ×3 | `D0_G1_…M1…Z1` | 0.48 | 567 | 2 | noise under MAE; M1 of the member sweep |
| LG03 ×3 | `D1_…M1…Z1` | 0.48 | 567 | 2 | noise under MSE |
| LG04 | `…M2…Z0` | 1.00 | 322 | 2 | RF01's objective, noise pathway removed |
| EN01 | `D0_G1_…M2…` | 1.00 | 322 | 2 | pure CRPS at 2 members; what the 0.1 spectral term buys |
| NC01 | `…N1…` | 1.00 | 322 | 3 | gaussian vs isotropic noise |
| OI02 | `…Q1…` | 1.00 | 322 | 3 | multiscale increment CRPS |
| RO01 | `…R1…` | 1.21 | 376 | 3 | rollout length alone |
| EN02 | `D0_G1_…M3…` | 1.44 | 433 | 4 | three members |
| OI01 | `…G3…` | 1.00 | 322 | 4 | 0.5/0.5 split — third point on the trade-off |
| OI04 | `…Y1…` | 1.00 | 322 | 4 | almost-fair CRPS |
| RO02 | `…R2…` | 1.89 | 549 | 4 | the second *scored* step |
| RO03 | `D1_…M1…R1…Z0` | 0.58 | 215 | 4 | rollout on the deterministic row |
| RO04 | `…R4…` | 1.52 | 455 | 5 | sampled rollout to 20 steps |
| ~~CU01~~ | `…I1…` | 1.00 | 322 | 6 | warm start from RF02 — **parked**, compute-confounded |
| ~~NC02~~ | `…Z2…` | 1.00 | 322 | 6 | noise dim 64 — **parked**, unpaired at one seed |
| ~~OI03~~ | `…Q3…` | 1.00 | 322 | 6 | three levels — **parked**, weak contrast by construction |

P1 567, P2 2,347, P3 1,021, P4 1,841, P5 455, **P6 967 (parked)**.
`submit-campaign.sh` caps at P3 by default; P6 needs an explicit
`--max-priority 6` and each arm's `.env` carries the reason it is parked.

### The LG block is a 2×2, and only its rows are paired

RF01−RF02 moves loss family, noise conditioning and member count at once.
Holding members at one and crossing the other two gives:

| | no noise (`Z0`) | noise wired (`Z1`) |
|---|---|---|
| **MSE** | RF02 | LG03 |
| **MAE** (`G1` at `M1`) | LG01 | LG02 |

At one member the CRPS pairwise term is exactly zero, so the `D0` row is MAE
(verified to the bit). Row = loss geometry, column = noise conditioning.

Two things to read carefully:

* **LG03−RF02 and LG02−LG01 are the noise *simple effects*** under MSE and
  under MAE. Their difference is the loss-by-noise interaction. They are not
  two draws of one number.
* **The columns are not paired.** Changing `Z` reshuffles the whole
  initialisation stream — only 5 of 22 shared tensors survive a `Z` change at a
  fixed seed (`analysis/seed_pairing.py`). So a `Z` contrast carries a full
  seed's worth of noise and a loss contrast carries none. That is why LG01–LG03
  have three seeds and why NC02 is parked.
* **Nothing in this 2×2 rewards ensemble spread.** At `M1` MAE and MSE both
  want a point estimate, so the block tests whether the noise pathway *harms*
  point skill or acts as a regulariser — not whether it produces useful
  uncertainty. **LG04 is the arm that asks that**, at `M2` under the 0.9/0.1
  objective.

## Running it

```bash
cd configs/experiments/e3sm_sep26_atm/sbatch-scripts
./generate-campaign.sh --list        # run list + budget
./generate-campaign.sh               # write ../runs, then check
./submit-campaign.sh --dry-run
./submit-campaign.sh --preflight     # stage + validate, queue nothing
./submit-campaign.sh                 # queue P1..P3
./submit-campaign.sh --max-priority 5
```

Regenerating is a no-op against a committed `runs/`. Output lands in
`$CAMPAIGN_ROOT/<runid>`, default `$PSCRATCH/sep26`.

---

## Evaluating it

Training's inline rollouts run **one member per initial condition**, so they
score a single realisation and nothing else. Calibration, spread and every
finite-ensemble score come from the offline passes.

```bash
sbatch-scripts/stage-data.sh                       # once per campaign; see below
export EVAL_DATA_ROOT=$PSCRATCH/sep26-data
./make_eval_config.py RF01.S01 --pass scores       # 4 members/IC, no files
./make_eval_config.py RF01.S01 --pass traj         # 1 member, files written
sbatch-scripts/submit-eval.sh --all --pass scores --dry-run
sbatch-scripts/submit-eval.sh RF01.S01 --noise-ladder
analysis/eval_table.py $EVAL_ROOT --seeds          # the noise floor
analysis/eval_table.py $EVAL_ROOT --ladder         # overrides against it
```

**Stage the data first.** The training template reads CFS, which compute nodes
see through DVS, and the evaluator's access pattern is the one DVS handles
worst. Two concurrent 8-IC evaluations ran at 84 s per window against CFS with
ranks parked in `dvsipc_wait_for_response`, and at **13.5 s per window** against
a staged copy on Lustre. The copy is 300 GB and takes 77 s; it repays itself
before the first run finishes. `--data-root` refuses a staged root that is
missing years the rollout reaches, since a short glob gives a short dataset
rather than an error.

| pass | shape | what it is for |
|---|---|---|
| `scores` | members per IC, no trajectory files | CRPS, spread–skill ratio, ensemble-mean RMSE and rank-histogram calibration at 6 h / 1 d / 5 d / 30 d / 90 d / 1 y |
| `traj` | one member per IC, three fields written | per-trajectory variance, persistence, quantiles, wet-day frequency and intensity |

The two passes are not interchangeable and the generator says so: asking for an
ensemble on the trajectory pass is an error, because per-trajectory statistics
must be computed **inside** a trajectory. Averaging four members first gives the
lowest RMSE in the study and 8–41% too little variance.

`--noise-ladder` adds four more inferences on the same weights — noise off, the
iterated conditional mean, one held latent field, half amplitude. Each is one
inference on a checkpoint that already exists and together they are the cheapest
mechanism probe available; what they showed on RF01 is in
`analysis/noise_decomp/REVIEW.md`. **`--noise off` is not a deterministic
control** — it is 14–16% worse than the model's own conditional mean at one
step. The deterministic control is RF02.

The scores pass stops at its last scored lead — one year — because rolling on
buys no ensemble metric, only a better-sampled climatology, which is what the
trajectory pass is for. `--years` overrides either.

`ssr_bias` says whether the spread is the right *size*; `rank_bias` and
`rank_dispersion` say whether the ensemble is the right *shape*. An ensemble too
narrow in the core and too wide in the tails passes the first and fails the
second, and since the arms here differ in exactly how their loss shapes a
distribution, that is the distinction the campaign is built to see.

Output lands in `$EVAL_ROOT/<exp>.S<seed>.eval-<pass>[-<noise>]`, default
`$PSCRATCH/sep26-eval`. `analysis/eval_table.py` reads it. Start with `--seeds`:
the seed-to-seed spread of one arm is the floor every comparison is measured
against, and a difference smaller than it is not a result.

---

## Cards: 40 GB is enough (measured)

| arm | peak MiB / 40,960 | s/batch 40 GB | s/batch 80 GB |
|---|---|---|---|
| 1 member, 1 step | 15,383 | 0.433 | 0.390 |
| 2 members, 1 step | 21,155 | 0.903 | 0.826 |
| 3 members, 1 step | 24,651 | 1.304 | 1.177 |
| 2 steps, both scored | 21,921 | 1.713 | 1.551 |
| 20 steps, last-only | 23,233 | 5.355 | — |

40 GB is **9.4–11% slower** for 5.5× the node pool (1408 vs 256 in `gpu_ss11`)
and no reservation. Memory is driven by **members**, with a ~2 GB rollout-depth
term on top; worst arm is `M3` at 40% headroom.

`Q`, `Y` and `N` cost nothing resolvable: the baseline itself varies **4%
between nodes** (0.903 vs 0.868), and every loss-axis variant sits inside that.
Treat them as 1.00 ± 0.04.

---

## Upstream: four bugs, all fixed

Found by running arms, not validating them. All four now have tested fixes on
branches off `main` (pushed to the fork), and three are cherry-picked onto this
branch.

| | bug | status |
|---|---|---|
| B1 | `EnergyScoreLoss` sized `mode_weights` against `x_hat`, which still has the ensemble dim | fixed, ported, `G2` re-run on a node |
| B2 | `get_energy_score` supported exactly two members | fixed, ported, `M3` re-run on a node |
| B5 | `get_crps` epsilon hard-coded to `(1-alpha)/2` | fixed, ported, unit-tested at M = 1, 2, 3, 5 |
| B4 | every data-parallel rank draws identical noise | fix tested, **deliberately not taken** — see `PLAN.md` §11.1 |

Both first-batch blockers are lifted from the generator on measured evidence
(`PLAN.md` §12): `M3` with an energy weight trains for 209 steps (loss 4.0444 → 0.3705)
and pure energy for 250 steps (1.1952 → 0.2952). What the generator still refuses is
degeneracy and truth-in-labelling, not upstream breakage.

**Consequence for aug26, running now:** its E01 predates B1, so the energy
term's contribution to the *per-channel* breakdown is a constant across all 50
channels. The scalar total is unaffected — bit-identical across the fix — so
rankings survive, but don't present per-channel loss comparisons from a `D0`
run trained before it.

## Testing

```bash
uv run --extra dev python -m pytest configs/experiments/e3sm_sep26_atm/test_campaign.py
```

94 tests: the generator and checker, the ported loss fixes, and the offline
evaluation generator. Half are mutation tests: each breaks one thing in a
generated config and asserts the checker notices. `check_campaign.py` duplicates the generator's
tables on purpose — a checker that imports them can only prove the generator is
self-consistent.

`check_campaign.py` also asserts **RF01 is still aug26's E01**, since the
template was copied from it and five arms difference against those three seeds.

---

## Gotchas

* **`/tmp` is node-local.** Stage scripts and configs to `$PSCRATCH` or a batch
  job dies with exit 127.
* **`log_train_every_n_batches` is a `TrainConfig` field**, not `LoggingConfig`.
* **A sampled rollout is an `outcomes` list**, not a `{steps: probability}`
  mapping. `validate_config` catches it with an opaque `UnionMatchError`.
* **`uvx` and `pre-commit` both hit flock/errno 524.** Set `UV_TOOL_DIR` and
  `UV_CACHE_DIR` to node-local storage first.
* **Judge a run by `REAL_EXIT=0` and `DONE ---- rank 0`**, never the log tail.
* **The data loader is bimodal.** Step timing must be a median over windows.
* **A config that parses is not a config that runs.** Smoke-test any new axis
  with a real forward *and backward* pass.
* **Evaluations read CFS through DVS and it is the bottleneck**, not the GPUs.
  A rank in uninterruptible `D` while other GPUs sit at 100% is this, not an
  arm that needs more memory. Stage to Lustre first.

## Files

| | |
|---|---|
| `PLAN.md` | design argument, decision rules, measurements |
| `TODO.md` | what is left |
| `AGENTS.md` | history |
| `make_campaign.py` | run list, axis tables, guards, generator |
| `make_eval_config.py` | offline evaluation configs, both passes |
| `sbatch-scripts/stage-data.sh` | copy the dataset off DVS before evaluating |
| `analysis/eval_table.py` | read the scores pass: seed floor, noise ladder |
| `check_campaign.py` | asserts each config agrees with its run id |
| `test_campaign.py` | unit + mutation tests |
| `analysis/` | Tier 0 reads, card sweep, upstream-fix verification |
| `analysis/noise_decomp/` | what the conditioning noise does to a trajectory |
| `runs/` | generated; do not hand-edit |
