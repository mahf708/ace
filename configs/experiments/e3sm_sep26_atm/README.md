# sep26 — E3SMv3 historical, atmosphere-only ablation

An ablation of the **training objective** for the ACE2S atmosphere. Everything
else — channels, batch size, learning rate, loss weighting, data — is held at
the aug26 E01 tuning set.

**19 runs · 76 nodes concurrent · 5,741 node-hours · 137 h critical path.**
Built, checked, validated. Nothing queued.

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
| `G` | crps/energy split | `G0` .9/.1 · `G1` 1/0 · `G2` 0/1 · `G3` .5/.5 |
| `I` | init | `I0` scratch · `I1` warm start |
| `M` | members | `M1` `M2` `M3` |
| `N` | noise type | `N0` isotropic · `N1` gaussian |
| `Q` | pooled CRPS levels | `Q0` off · `Q1` · `Q3` (weight 0.1) |
| `R` | rollout | `R0` 1 step · `R1` 2 last-only · `R2` 2 both · `R3` ≤4 sampled · `R4` ≤20 sampled |
| `Y` | almost-fair alpha | `Y0` 1.00 · `Y1` 0.95 |
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
| LG01 | `D0_G1_…M1…Z0` | 0.48 | 189 | 2 | MAE vs MSE, no noise either side |
| LG02 | `D0_G1_…M1…Z1` | 0.48 | 189 | 2 | noise under MAE; M1 of the member sweep |
| LG03 | `D1_…M1…Z1` | 0.48 | 189 | 2 | noise under MSE |
| EN01 | `D0_G1_…M2…` | 1.00 | 322 | 2 | pure CRPS at 2 members; what the 0.1 energy term buys |
| NC01 | `…N1…` | 1.00 | 322 | 3 | gaussian vs isotropic noise |
| OI02 | `…Q1…` | 1.00 | 322 | 3 | spatially pooled CRPS |
| RO01 | `…R1…` | 1.21 | 376 | 3 | rollout length alone |
| EN02 | `D0_G1_…M3…` | 1.44 | 433 | 4 | three members |
| OI01 | `…G3…` | 1.00 | 322 | 4 | 0.5/0.5 split — third point on the trade-off |
| OI04 | `…Y1…` | 1.00 | 322 | 4 | almost-fair CRPS |
| RO02 | `…R2…` | 1.89 | 549 | 4 | the second *scored* step |
| RO03 | `D1_…M1…R1…Z0` | 0.58 | 215 | 4 | rollout on the deterministic row |
| CU01 | `…I1…` | 1.00 | 322 | 5 | warm start from RF02 |
| NC02 | `…Z2…` | 1.00 | 322 | 5 | noise dim 64 |
| OI03 | `…Q3…` | 1.00 | 322 | 5 | three coarsening levels |
| RO04 | `…R4…` | 1.52 | 455 | 5 | sampled rollout to 20 steps |

Priorities drain reference → mechanism → single-factor → tail:
P1 567, P2 890, P3 1,021, P4 1,841, P5 1,422 node-hours.

**The LG block is a 2×2.** RF01−RF02 moves loss family, noise conditioning and
member count at once. Holding members at one and crossing the other two gives:

| | no noise (`Z0`) | noise wired (`Z1`) |
|---|---|---|
| **MSE** | RF02 | LG03 |
| **MAE** (`G1` at `M1`) | LG01 | LG02 |

At one member the CRPS pairwise term is zero, so the `D0` row is MAE. Row =
loss geometry, column = noise conditioning with nothing rewarding it. LG03−RF02
and LG02−LG01 are two independent estimates of the noise main effect.

---

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

## Blocked on upstream

Both found by running arms, not validating them. The generator refuses both.

1. **`M1`/`M3` with any energy-score weight.** `get_energy_score`
   (`fme/core/ensemble.py:80`) supports exactly two members. *Workaround in
   use:* the member sweep runs on `G1`, which sets `energy_score_weight: 0` and
   makes `EnsembleLoss.forward` skip it entirely.
2. **`G2` (pure energy).** `EnergyScoreLoss` sizes `mode_weights` against
   `x_hat`, which still has the ensemble dim, so the component carries two
   spurious leading dims and `get_channel_losses` raises on the first batch.
   One-line fix verified in `analysis/verify_mode_weights_fix.py`; E01's total
   is bit-identical across it.

**Consequence for aug26, running now:** E01's total loss is correct, but the
energy term's contribution to the *per-channel* breakdown is a constant across
all 50 channels. Don't present per-channel loss comparisons from a `D0` run
until the fix lands.

---

## Testing

```bash
uv run --extra dev python -m pytest configs/experiments/e3sm_sep26_atm/test_campaign.py
```

61 tests. Half are mutation tests: each breaks one thing in a generated config
and asserts the checker notices. `check_campaign.py` duplicates the generator's
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

## Files

| | |
|---|---|
| `PLAN.md` | design argument, decision rules, measurements |
| `TODO.md` | what is left |
| `AGENTS.md` | history |
| `make_campaign.py` | run list, axis tables, guards, generator |
| `check_campaign.py` | asserts each config agrees with its run id |
| `test_campaign.py` | unit + mutation tests |
| `analysis/` | Tier 0 reads, card sweep, upstream-fix verification |
| `runs/` | generated; do not hand-edit |
