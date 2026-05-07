# 08 — Multi-node deployment & NCCL/IB tuning guide

**Status**: NOT STARTED
**Depends on**: any of 03+ (the doc gets richer as features land)
**Blocks**: —
**Estimated size**: S (~1 day)

## Goal

Add a deployment guide covering Slurm launch templates, NCCL/IB environment knobs, and recommended `(fsdp × spatial × tp)` topology for common GPU/node counts on the noise-conditioned SFNO. Today the repo has zero multi-node deployment documentation — `AGENTS.md` describes the test infrastructure, but nothing addresses production deployment.

## Deliverable

A new Sphinx page: `docs/distributed.rst`, linked from `docs/index.rst`.

## Outline

1. **Quick start: single-node, multi-GPU**
   - `torchrun --standalone --nproc-per-node 8 -m fme.ace.train <config>`
   - Reference YAML with `parallelism: { spatial_h: 2, spatial_w: 2 }` (4 spatial ranks, 2 data replicas) and `parallelism: { fsdp_size: 8 }` (pure FSDP) — document when to pick which.

2. **Slurm template** (existing path: `fme/core/distributed/torch_distributed.py:50-66`)
   - Sample `sbatch` script: `--nodes`, `--ntasks-per-node=8`, `--gpus-per-task=1`, `--gpu-bind=closest`.
   - Required env vars: `SRUN_DIST_FILE_PATH=<shared filesystem path>`.
   - Optional env vars listed in §3.

3. **NCCL / IB tuning**
   Tabulate the env knobs that matter, with guidance:
   - `NCCL_IB_HCA` — restrict to active HCAs (e.g., `mlx5_0,mlx5_1`)
   - `NCCL_SOCKET_IFNAME` — interface for bootstrap (e.g., `^docker0,lo`)
   - `NCCL_NET_GDR_LEVEL=PHB` (or `SYS`) — GPUDirect RDMA threshold
   - `NCCL_IB_QPS_PER_CONNECTION=4` — improves bandwidth at high node count
   - `NCCL_BUFFSIZE=8388608` — 8 MB; helps at high IB BW
   - `NCCL_ASYNC_ERROR_HANDLING=1` — fail-fast on collective hang
   - `NCCL_DEBUG=INFO` for first-run sanity, off in production
   - `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
   - Note that the right values are cluster-specific; the guide gives a starting point, not a one-size-fits-all.

4. **Recommended topology by scale**
   This section gets richer as later tasks land. Initial table:

   | World size | Recommended `(fsdp, spatial_h, spatial_w)` | Notes |
   |---|---|---|
   | 1-8 (1 node) | `(8, 1, 1)` pure FSDP, or `(2, 2, 2)` mixed | FSDP if model is large; mixed if activations dominate |
   | 16 (2 nodes) | `(16, 1, 1)` HSDP — sharded intra-node, replicated inter-node | Task 07 |
   | 32 (4 nodes) | `(32, 1, 1)` HSDP, or `(8, 2, 2)` HSDP × spatial | At higher resolution, add spatial |
   | 64+ | `(8, 4, 2)` HSDP × spatial; consider TP=2 for huge models | Task 06 |

5. **Checkpointing**
   - DCP layout (from task 01): directory of files, atomic via `<path>.tmp/` rename.
   - Resume across topology change: example commands.

6. **Troubleshooting**
   - "Hang at first step": almost always NCCL bootstrap; check `NCCL_DEBUG=INFO` and IB visibility.
   - "OOM at high resolution": follow the bucket framing in `CONTEXT.md`; profile with the task 00 instrumentation script.
   - "Loss diverges under FSDP": mixed-precision policy; check `reduce_dtype=fp32`.

7. **Validation runbook**
   - Recipe: small noise-conditioned SFNO config + 100 training steps + cross-topology save/load + verify loss matches single-rank baseline within tolerance. This is how a user should validate their deployment before committing real wall-clock to a long run.

## Steps

1. Draft the page using the outline above.
2. Wire it into `docs/index.rst`.
3. Cross-link from `README.md` and `AGENTS.md` (single-line "for production deployment, see ...").
4. Add a smoke-test script `scripts/validate_multinode.sh` that runs the validation runbook from §7.

## Acceptance criteria

- `make -C docs html` builds without errors, page renders correctly.
- Validation runbook script runs end-to-end on a 2-rank torchrun (CPU is fine for the runbook itself).
- All NCCL env knobs in the table have a citation to PyTorch / NCCL docs.

## Verification

1. Build the docs: `make -C docs html`.
2. Run the validation runbook on a small cluster (2 ranks CPU is sufficient for correctness; real cluster validation is up to the user).
3. Cross-check the topology table against task 00's instrumentation results once those land.

## Notes / handoff log

- This task can begin as a stub immediately after task 03 lands; flesh out as 04, 06, 07 land.
- Keep the doc short and prescriptive. Users reaching this page typically need a working command, not a survey of options. The "Recommended topology" table is the most-read section; protect its quality.
