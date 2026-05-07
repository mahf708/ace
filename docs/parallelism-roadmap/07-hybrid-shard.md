# 07 — `HYBRID_SHARD` as multi-node default

**Status**: NOT STARTED
**Depends on**: 04 (FSDP × spatial composition)
**Blocks**: —
**Estimated size**: S (~1 day)

## Goal

Configure FSDP2 to **shard within a node and replicate across nodes** (`HYBRID_SHARD` semantics) by default for multi-node training. This avoids the `all_gather` traffic crossing IB/RoCE on every forward, keeping the heavy FSDP communication on intra-node NVLink/NVSwitch.

## Why this is the right multi-node default

Pure FSDP (`FULL_SHARD`) shards weights across the entire data-parallel mesh. At multi-node scale, each forward triggers an `all_gather` of weights along that whole mesh — meaning weights move across IB on every block. This is bandwidth-bound and dominates iteration time at >1 node.

`HYBRID_SHARD`:
- Within each node: full FSDP sharding (intra-node `all_gather` on NVLink).
- Across nodes: DDP replication (inter-node `all_reduce` once per backward).

For models that fit `1/intra_node_size` of params on a single GPU, this is the right answer. For larger models that don't fit even sharded within a node, you fall back to `FULL_SHARD` and accept the IB cost.

## Files touched

Primary:
- `fme/core/distributed/fsdp_distributed.py` (from task 03) — accept a `hybrid_shard: bool` config and build a 2-D FSDP mesh `(replicate, shard)` instead of 1-D when enabled.
- `fme/core/distributed/config.py` (from task 02) — add `hybrid_shard: bool = False` to `ParallelismConfig`. Default to `True` once this task lands? Open question, see below.

Tests:
- Extend `parallel_tests/test_fsdp.py` from task 03 with HSDP cells.

## Mesh shape

```python
# Single-node FSDP: 1-D mesh
mesh = init_device_mesh("cuda", (fsdp_size,), mesh_dim_names=("fsdp",))

# HSDP: 2-D mesh, "shard" group is intra-node, "replicate" group is inter-node
n_intra = torch.cuda.device_count()                # 8 on H100/A100 nodes
n_replicate = world_size // n_intra
mesh = init_device_mesh(
    "cuda",
    (n_replicate, n_intra),
    mesh_dim_names=("replicate", "shard"),
)
fully_shard(module, mesh=mesh)  # FSDP2 reads multi-dim mesh as HSDP automatically
```

## Steps

1. **Detect intra-node device count**. Use `torch.cuda.device_count()` or `LOCAL_WORLD_SIZE` env var. Document behavior under heterogeneous nodes (probably reject — single-node-size HSDP only).
2. **Build the 2-D mesh** when `hybrid_shard=True` and `world_size > intra_node_size`.
3. **Rank-to-mesh mapping**. Default torchrun rank ordering is intra-node-first. PyTorch's `init_device_mesh` honors this. Validate with a small test that prints `mesh.get_local_rank("shard")` and `mesh.get_local_rank("replicate")` on each rank.
4. **Compose with task 04's spatial mesh**. Final mesh becomes 4-D: `(replicate, shard, h, w)`. The `(replicate, shard)` slice is FSDP's HSDP mesh; `(h, w)` is spatial. Document the rank ordering convention.
5. **Default behavior**. Leave `hybrid_shard=False` as the default initially — flip to `True` once task 09 (`08-multinode-docs.md`) covers the user-facing implications.

## Acceptance criteria

- `parallelism: { fsdp_size: <world>, hybrid_shard: true }` produces an HSDP mesh.
- On a 2-node × 8-GPU cluster, intra-node `all_gather` traffic is on NVLink and inter-node `all_reduce` traffic is on IB (verify with `NCCL_DEBUG=INFO`).
- Throughput at 16 ranks under HSDP is meaningfully better than `FULL_SHARD` at the same world size (expect at least 1.3× on a typical IB cluster; document actual measurement).

## Verification

1. Numerics: HSDP at world_size=4 with intra-node-size=2 produces same parameter values as FSDP at world_size=4 (same math, different mesh — should be bit-identical with the same seed).
2. Comm pattern: NCCL log shows expected groups.
3. Throughput regression: log iter/sec for FULL_SHARD vs HSDP across 8 / 16 / 32 ranks.

## Notes / handoff log

- HSDP only makes sense at >1 node. On a single 8-GPU node, HSDP degenerates to FSDP and adds a tiny mesh overhead. The backend should silently fall back to FSDP when `n_replicate == 1`.
- For very large models that exceed intra-node memory, `FULL_SHARD` is required. Document the rule of thumb: if `param_bytes_per_rank < gpu_mem_minus_activations` under HSDP, use HSDP; otherwise use `FULL_SHARD`.
- This task is small but high-value at the user's deployment target (multi-node HPC). Recommend pairing with task 08 (deployment docs) so users get a working "first multi-node run" experience.
