# sep26 — E3SMv3 historical, atmosphere-only ablation

An ablation of the **training objective** for the ACE2S atmosphere: the loss
family, its internals, the member count, the noise conditioning and the
rollout. Nothing else varies — the channels, batch size, learning rate, loss
weighting and data are held at the aug26 E01 tuning set.

`PLAN.md` is the design argument and the measurements. This file is what a
colleague needs to run it.

## The two things to know first

**The references are aug26's, and they are not re-run here.**

| | aug26 run id | what it is |
|---|---|---|
| **REF-S** | `E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S{01,02,03}` | stochastic pole: EnsembleLoss crps 0.9 / energy 0.1, 2 members, noise 32, 1 step |
| **REF-D** | `E21.aug26.atm.….D1_I0_M1_RF1_Z00.S{01,02,03}` | deterministic pole: MSE, 1 member, no noise, 1 step |

Every sep26 arm is one factor from one of them. That is what keeps the campaign
to ~2,700 node-hours instead of needing its own baselines. **REF-D does not
exist yet** — it is queued in aug26 — and half the run list differences against
it.

**Two arms that look fine cannot train, and both passed `validate_config`.**
The generator refuses them; see "Blocked on upstream" below. The lesson
generalises: a config that parses is not a config that runs, so smoke-test any
new axis with a real forward *and backward* pass before queueing it.

## Run ids

    <campaign>.<realm>.<delta>.s<seed>

    sep26.atm.base.s01
    sep26.atm.crps-pure.s01
    sep26.atm.crps-pure_mem-1_noise-0.s01

The delta is **sparse and canonically sorted**: only axes that differ from the
template appear, rendered `key-value` and joined by `_`, and the empty delta is
the literal `base`. Adding an axis therefore cannot rename an existing run —
that is a property of the encoding, not a rule anyone has to remember.

The delta is also the identity; there is no experiment number, so inserting an
arm renumbers nothing. Short handles for slides (`M01`, `L02`) live in
`MANIFEST.tsv` and the W&B tags, never in the id or the directory name.

| axis | template | levels |
|---|---|---|
| `obj` | `crps` | `mse` |
| `crps` | 0.9 / 0.1 | `pure` 1.0/0.0 · `energy` 0.0/1.0 · `half` 0.5/0.5 |
| `mem` | `2` | `1` · `3` |
| `noise` | `32` | `0` · `64` |
| `ntype` | `iso` | `gauss` |
| `roll` | `f1` | `c2` 2 steps last-only · `f2` 2 both scored · `s04` · `s20` |
| `fdcrps` | `0` | `1` · `3` levels, at weight 0.1 |
| `alpha` | `100` | `095` |

## Running it

```bash
cd configs/experiments/e3sm_sep26_atm/sbatch-scripts

./generate-campaign.sh --list        # the run list and the budget
./generate-campaign.sh               # write ../runs, then check it
./submit-campaign.sh --dry-run       # what would be queued
./submit-campaign.sh --preflight     # stage + validate everything, queue nothing
./submit-campaign.sh                 # queue P9 (the arms that carry the claims)
./submit-campaign.sh --max-priority 11   # ...including the tail
```

Regenerating is a no-op against a committed `runs/` — the output has no
username, scratch path or timestamp in it, which is what lets several people
share one campaign. `run-train.sh` refuses a dirty worktree, so without that
property only the generator's author could launch anything.

Output lands in `$CAMPAIGN_ROOT/<runid>`, default `$PSCRATCH/sep26`.

## Cards: 40 GB, measured

The atmosphere fits an A100-40GB with room, and this campaign is sized for the
1408-node `hbm40g` pool rather than the 256-node `hbm80g` one, so it needs no
reservation. Measured 2026-09-03 (`analysis/card-sweep.sh`):

| arm | peak MiB of 40,960 | headroom | s/batch, 40 GB | s/batch, 80 GB |
|---|---|---|---|---|
| 1 member, 1 step | 15,383 | 62% | 0.433 | 0.390 |
| 2 members, 1 step | 21,155 | 48% | 0.903 | 0.826 |
| 3 members, 1 step | 24,651 | 40% | 1.304 | 1.177 |
| 2 members, 2 steps both scored | 21,921 | 46% | 1.713 | 1.551 |

The 40 GB card is **9.4–11% slower** — stable across all four variants — which
buys 5.5x the node pool. Note that **rollout length costs time, not memory**:
`use_gradient_accumulation: true` means each scored step's backward runs before
the next forward, so the sampled-rollout arms are free on memory however long
they get.

## Blocked on upstream

Both were found by running them, not by validating them.

1. **`mem-1` and `mem-3` under any energy-score weight.**
   `get_energy_score` (`fme/core/ensemble.py:80`) supports exactly two members
   and raises on the first training batch otherwise. This is aug26's E25 and
   E26. **Workaround in use:** the member sweep runs on `crps-pure`, which sets
   `energy_score_weight: 0.0`, and `EnsembleLoss.forward` then skips the energy
   score entirely. The cost is that those arms are two factors from REF-S rather
   than one, so they answer the narrower "does the member count matter to a pure
   CRPS objective".
2. **`crps-energy`.** `EnergyScoreLoss` builds `mode_weights` with
   `x_hat.ndim - 1` leading singleton dims while `get_energy_score` has already
   consumed the ensemble dim, so the energy component is shaped
   `(1, 1, B, C, n_l, n_m)`. With a CRPS component present the correctly-shaped
   one carries the channel breakdown and nothing fails; alone, it raises
   `Per-channel loss has 1 elements but 50 channel names were provided`.
   The arm is commented out in `RUNLIST` rather than deleted — restore it when
   the fix lands.

**A consequence for aug26, which is running now:** E01 is not broken — its total
loss is exactly `0.9 × CRPS + 0.1 × energy`, verified — but the energy term's
contribution to the *per-channel* breakdown is a constant across all channels.
Per-channel loss plots therefore show `0.9 × CRPS_channel + constant`, and the
spectral term contributes nothing channel-specific to them.

## Testing

```bash
uv run --extra dev python -m pytest configs/experiments/e3sm_sep26_atm/test_campaign.py
```

Half of it is mutation tests: each breaks one thing about a generated config and
asserts `check_campaign.py` notices. A checker that passes a clean campaign
proves nothing — aug26's passed E25 and E26.

`check_campaign.py` duplicates the generator's level-to-value tables **on
purpose**. A checker that imports them can only prove the generator is
self-consistent; one that re-derives the expected config from the run id
catches a typo in that mapping, which is what it exists for.

## Gotchas

* **`/tmp` is node-local on Perlmutter.** A script or config staged there is
  invisible to any other node and a batch job that reads one dies with exit 127
  about nine seconds in. Stage to `$PSCRATCH`.
* **`log_train_every_n_batches` is a `TrainConfig` field**, not a
  `LoggingConfig` one. `--override logging.log_train_every_n_batches=N` is
  rejected by dacite at config parse, before any GPU work.
* **`pre-commit` cannot run on Perlmutter** (flock, errno 524) — and neither can
  a bare `uvx`, for the same reason. Point uv at node-local storage first:
  `export UV_TOOL_DIR=/tmp/$USER/uvtools UV_CACHE_DIR=/tmp/$USER/uvcache`, then
  the pinned `uvx ruff@0.8.1` / `uvx --with types-PyYaml==5.4.3 mypy@1.15.0`.
* **Judge a run by `REAL_EXIT=0` and `DONE ---- rank 0`,** never by the log
  tail: successful runs print alarming teardown tracebacks.
* **The atmosphere data loader is bimodal, not noisy.** One interval in twenty
  is a `time_buffer` window refill several times longer than the rest. Any step
  timing has to be a median over windows, not an end-to-end mean, or it is a
  coin flip on whether the refill was caught.

## The unmerged dependency

This branch sits on `e3sm/exps/hist-v2026.8.0`, not on `main`, and that is not a
convenience: `time_buffer` does not exist on `main`, and `time_buffer: 10` with
`time_buffer_pool_size: 2` is what took the atmosphere from 3.155 to
0.925 s/batch. The whole campaign rests on ~1,660 lines of experiment-branch
code that should be PR'd to `main` on its own `feature/` branch.

## Files

| | |
|---|---|
| `PLAN.md` | the design argument, the decision rules and the measurements |
| `config-train-atm.template.yaml` | the template — a template, not a run |
| `make_campaign.py` | run list, axis tables, guards, generator |
| `check_campaign.py` | asserts each config agrees with its own run id |
| `test_campaign.py` | unit and mutation tests |
| `analysis/` | the Tier 0 reads and the card sweep |
| `sbatch-scripts/` | generate, submit, stage-and-launch |
| `runs/` | generated; do not hand-edit |
