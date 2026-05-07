# Parallelism Roadmap

**Goal**: enable the noise-conditioned ("stochastic") SFNO to fit and train at higher resolution on multi-node HPC by adding PyTorch-native fully-sharded parallelism (FSDP2 / DTensor / sharded checkpoints / tensor parallelism) and composing it with the existing spatial-parallelism backend.

**Audience**: future agents (and humans) picking up one or more tasks from this roadmap. Each task file is self-contained — it states its own preconditions, files to touch, acceptance criteria, and verification, so an agent can work it without reading the rest of the roadmap.

**Status**: design-stage. No code lands until a task is explicitly approved.

Read [`CONTEXT.md`](./CONTEXT.md) first — it distills *why* this work exists and the memory-budget framing that motivates the task ordering.

---

## Task index

| ID | Title | Status | Depends on | Estimated size |
|----|---|---|---|---|
| [00](./00-instrumentation.md) | Memory instrumentation: validate the premise | NOT STARTED | — | S |
| [01](./01-dcp-checkpoints.md) | Replace rank-0 checkpointing with DCP | NOT STARTED | — | M |
| [02](./02-config-yaml.md) | Promote parallelism config from env vars to YAML | NOT STARTED | — | S–M |
| [03](./03-fsdp2-backend.md) | Add an FSDP2 (`fully_shard`) backend | NOT STARTED | 01 | L |
| [04](./04-fsdp-spatial.md) | Compose FSDP2 with the existing spatial backend | NOT STARTED | 03 | L |
| [05](./05-dtensor-migration.md) | Express spatial sharding via DTensor on a unified DeviceMesh | NOT STARTED | 04 | XL |
| [06](./06-tensor-parallel.md) | Tensor parallelism on AdaLN + MLP + spectral conv channels | NOT STARTED | 05 | L |
| [07](./07-hybrid-shard.md) | `HYBRID_SHARD` as multi-node default | NOT STARTED | 04 | S |
| [08](./08-multinode-docs.md) | Multi-node deployment & NCCL/IB tuning guide | NOT STARTED | (any of 03+) | S |

Sizing key: S ≈ <1 day, M ≈ 1-3 days, L ≈ 3-7 days, XL ≈ >1 week of focused work.

---

## Dependency graph

```
            00 (instrumentation, validates premise — run anytime)

            01 (DCP)
              │
              ▼
            03 (FSDP2 backend)
              │
              ▼
            04 (FSDP × spatial composition)
             / \
            /   \
           ▼     ▼
          05    07 (HYBRID_SHARD default)
           │
           ▼
          06 (tensor parallelism)

  02 (YAML config)  ───── independent; lands cleanly any time after 03 is shaped
  08 (deployment docs) ── independent; useful as soon as 03 is shipping
```

**Critical path**: 01 → 03 → 04. Everything else either parallelizes with the critical path (00, 02, 08) or extends it (05, 06, 07).

---

## Sequencing strategies

### Strategy A — Sequential, single-agent
01 → 02 → 03 → 04 → 07 → 08 → 05 → 06.
This is the lowest-risk path: each task lands and is verified before the next starts. Best when only one agent is working on this.

### Strategy B — Parallel, two agents
- **Track 1 (memory)**: 01 → 03 → 04 → 07 → 05 → 06
- **Track 2 (ergonomics & validation)**: 00, 02, 08 in parallel; merges into Track 1 at any point

Strategy B halves wall-clock at the cost of more rebase / coordination work. Track 2 tasks have minimal collisions with Track 1 (different files), so the merge cost is small.

### Strategy C — Maximum parallel, three+ agents
- **Track 1**: 01 → 03 → 04 → 05 → 06 (one task at a time, blocking)
- **Track 2**: 00 (one-shot), then 07, then 08 (drafted ongoing as Track 1 lands features)
- **Track 3**: 02 (one-shot, can begin as soon as 03 has a stable interface)

Diminishing returns past three agents because of the strict 01 → 03 → 04 chain.

---

## Conventions for picking up a task

1. **Read `CONTEXT.md`** for the high-level "why" and memory-budget framing.
2. **Read the task file end-to-end** before opening any code.
3. **Re-verify the "Files touched" list** — line numbers may have drifted; use `grep` to re-locate the symbols named in the task before editing.
4. **Implement on the branch named in the task**, or a fresh branch named `claude/parallelism-<task-id>-<slug>` if none is named.
5. **Run the verification stanza locally** before opening a PR. Most tasks call `make cpu_test_all_parallel` plus a task-specific test addition.
6. **Update the task file's Status** field (NOT STARTED / IN PROGRESS / DONE / BLOCKED) and append a brief log entry under "Notes / handoff log" so the next agent has continuity.
7. **Do not silently change scope.** If a task as written is wrong, edit the task file with a rationale before starting code.

---

## Out-of-scope items, recorded for completeness

- Diffusion sampler upgrades (DPM++, Restart, parallel sampling). Independent of this roadmap; track separately.
- Pipeline parallelism for the FNO block stack. Not needed at current model depths.
- DeepSpeed / Megatron-Core integration. The PyTorch-native stack covers what's needed.
- Inference-time spatial parallelism. Out of scope per current direction (training-memory focus).

---

## Source materials

- The original meta-review that motivated this roadmap: `~/.claude/plans/we-have-spatial-parallelism-vectorized-ripple.md` (local-only; not checked in).
- Existing parallelism docs in this repo: `AGENTS.md` (test infrastructure), `Makefile` (`make cpu_test_all_parallel`), inline docstrings in `fme/core/distributed/`.
