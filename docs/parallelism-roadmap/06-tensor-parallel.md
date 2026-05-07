# 06 — Tensor parallelism on AdaLN + MLP + spectral conv channels

**Status**: NOT STARTED
**Depends on**: 05 (DTensor migration)
**Blocks**: —
**Estimated size**: L (3-7 days)

## Goal

Add a tensor-parallelism (TP) axis that shards model parameters along the channel dimension within an FNO block. This cuts per-rank channel-axis activation memory and the all-gather cost of large `SpectralConvS2` weights at high `lmax`. TP composes with FSDP and spatial parallelism on the unified `DeviceMesh` from task 05.

## Why this is worthwhile (and why it's last)

- At very high resolution, the dominant memory cost shifts from optimizer state to **`SpectralConvS2` weight tensors**, which scale with `lmax`. These are the largest single tensors in the model. FSDP's `all_gather` of these tensors to assemble the un-sharded weight at use-time becomes the comm bottleneck.
- TP shards the same weights along channel axes and **avoids the all-gather entirely** — replaces it with smaller activation collectives (`all_reduce` across the TP group on the row-parallel layer).
- It's last because: (a) the absolute memory win from TP is smaller than from FSDP for the parameter sizes the user is targeting today, and (b) TP requires DTensor (task 05) to be ergonomic.

## Files touched

Primary:
- `fme/core/models/conditional_sfno/layers.py:141-214,360-416` — annotate `ConditionalLayerNorm` AdaLN projections and `MLP`'s two `Conv2d` layers as `ColwiseParallel` / `RowwiseParallel`.
- `fme/core/models/conditional_sfno/s2convolutions.py:137-144` — `SpectralConvS2` weight is sharded along `out_ch/g` (column-parallel) or `in_ch/g` (row-parallel); pick to align with the surrounding MLP/AdaLN layout to avoid extra collectives.
- `fme/core/distributed/fsdp_distributed.py` — extend `wrap_module` to also call `parallelize_module(...)` on the per-block layers.

Tests:
- New: `fme/core/distributed/parallel_tests/test_tensor_parallel.py`.
- Extend matrix in `cpu_test_all_parallel`.

## Sharding scheme

Per FNO block (left → right inside the block's forward):

| Module | Parallelism | Notes |
|---|---|---|
| `ConditionalLayerNorm` (norm + AdaLN projections) | `SequenceParallel` on the norm; `ColwiseParallel` on the projection Conv2d | Gathers across TP only at the AdaLN scale/bias broadcast. |
| `SpectralConvS2` (forward SHT → spectral mul → inverse SHT) | column-parallel on the spectral weight along `out_ch/g` | Spectral mul becomes per-rank; iSHT input is `Shard(out_ch)` and gets row-reduced. |
| Inner skip + activation | local | No comm. |
| `ConditionalLayerNorm` (second one) | as above | |
| `MLP` (`Conv2d` → activation → `Conv2d`) | `ColwiseParallel(W1)` + `RowwiseParallel(W2)` | Classic megatron pattern. One `all_reduce` at the end of the block. |
| Outer skip | local | |

This produces **one `all_reduce` per FNO block per forward and per backward**, on the TP mesh group — comparable comm volume to a single FSDP `all_gather` of the same weights, but on the TP group (typically a smaller, faster group like NVLink within a node).

## Steps

1. **Define the TP mesh dim**. After task 05 the unified mesh has shape `(fsdp, h, w, tp)`. TP collectives use the `tp` sub-mesh.
2. **Annotate layer parallelism**. Use `parallelize_module(model.blocks[i], device_mesh=tp_mesh, parallelize_plan={...})`. The plan dict maps submodule names to `ColwiseParallel` / `RowwiseParallel` / `SequenceParallel`.
3. **Tune partitioning of `SpectralConvS2`**. Its weight has shape `[num_groups, modes_lat, out_ch/g, in_ch/g, 2]` complex. Pick `Shard(2)` (out_ch within a group) for column-parallel; pair with `Shard(3)` (in_ch within a group) for the next layer's row-parallel input. Document the convention.
4. **Compose with FSDP**. After `parallelize_module`, the layer's parameters are TP-sharded DTensors. `fully_shard(block, ...)` on top further shards along `fsdp` axis. PyTorch composes these via DTensor's multi-mesh placements; verify by inspecting `param.placements` after both wraps.
5. **Tests**: numerics regression vs. single-rank baseline under `(fsdp=1, spatial=(1,1), tp=2)`, then `(fsdp=2, spatial=(1,1), tp=2)`, then `(fsdp=2, spatial=(2,2), tp=2)`.

## Acceptance criteria

- Single training step under `(fsdp=2, spatial=(1,1), tp=2)` produces parameter values within `atol=1e-5, rtol=1e-4` of the single-rank baseline.
- Per-rank parameter memory for `SpectralConvS2` weights is reduced by `1/tp_size` relative to FSDP-only.
- Communication volume per FSDP `all_gather` is reduced (measure with NCCL_DEBUG=INFO or PyTorch profiler trace).
- New test file passes in the `cpu_test_all_parallel` matrix.

## Verification

1. Numerics test (above).
2. Memory test: instrumentation script reports expected `SpectralConvS2` weight reduction.
3. Profiler trace shows one `all_reduce` per block per forward on the TP group, no extra collectives on FSDP/spatial groups.

## Notes / handoff log

- TP is most useful when the TP group is high-bandwidth (NVLink within a node). Avoid placing the TP group across IB/RoCE — it will dominate iteration time. The mesh ordering in task 05 should put `tp` as the innermost dim so it lands on intra-node ranks first under standard rank-to-device mapping.
- `SequenceParallel` on the norm is an optimization; if it complicates AdaLN's Conv2d-broadcast pattern, drop it and use plain replicated norm. The wins are small relative to MLP and SpectralConv TP.
- The user's stated focus (memory at higher res) is largely solved by 03+04. TP's incremental memory win is real but smaller. Prioritize accordingly.
