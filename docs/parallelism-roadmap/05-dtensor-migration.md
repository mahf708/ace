# 05 — Express spatial sharding via DTensor on a unified DeviceMesh

**Status**: NOT STARTED
**Depends on**: 04
**Blocks**: 06 (tensor parallelism cleanly requires DTensor)
**Estimated size**: XL (>1 week of focused work). Largest refactor in the roadmap.

## Goal

Replace the bespoke spatial-sharding plumbing in `ModelTorchDistributed` with PyTorch's `DTensor` abstraction on a unified `DeviceMesh`. Keep torch-harmonics distributed SHT as the one cross-mesh primitive that crosses sharding dims (it has to — it's the SHT). Everything else — local slicing, gradient hooks, all-reduce — becomes DTensor's standard `redistribute` / collective semantics.

## Why this matters

Two practical reasons:

1. **Tensor parallelism (task 06)** is dramatically simpler if model parameters are already DTensors. `parallelize_module` from `torch.distributed.tensor.parallel` is a thin layer over DTensor placements; with bespoke spatial sharding it's nearly impossible.
2. **Composition robustness**. The custom autograd-aware all-reduce, custom gradient hooks, and explicit `get_local_slices` calls scattered through the SFNO layers (`fme/core/models/conditional_sfno/layers.py:128`, `fme/core/models/conditional_sfno/s2convolutions.py:126`) all become "what placement does this DTensor have?" — a much cleaner contract.

## Files touched

Primary:
- `fme/core/distributed/model_torch_distributed.py:46-435` — gut and replace with DTensor placements. Keep the SHT helper functions; move them to a new `fme/core/distributed/sht.py`.
- `fme/core/models/conditional_sfno/layers.py:128,141-214,360-416` — `ConditionalLayerNorm` and `MLP` consume DTensors instead of explicit slices.
- `fme/core/models/conditional_sfno/s2convolutions.py:126,137-144` — `SpectralConvS2` declares parameter placements; forward uses DTensor `redistribute` to convert to/from sharded activations around the SHT call.
- `fme/core/distributed/distributed.py` — `Distributed` becomes a thin wrapper around a single `DeviceMesh`; backend-specific code shrinks.

Tests:
- All of `fme/core/distributed/parallel_tests/`. Numerics tests should pass unchanged because the math is identical; only the plumbing changes.

## Design sketch

**Unified mesh**:
```python
mesh = init_device_mesh(
    "cuda",
    (fsdp_size, spatial_h, spatial_w, tensor_parallel_size),
    mesh_dim_names=("fsdp", "h", "w", "tp"),
)
# tp dimension is 1 until task 06 enables it
```

**Activation placement**:
- Inside an FNO block, activations are `DTensor` with `Shard(-2)` (height axis) on the `h` mesh dim and `Shard(-1)` on the `w` mesh dim.
- Around the distributed SHT call, redistribute to whatever placement the SHT expects — typically `Replicate()` after gathering, then re-shard after the inverse SHT. Torch-harmonics' distributed SHT may already operate on rank-local tiles; confirm and adapt.

**Parameter placement**:
- Pre-FSDP: parameters are `DTensor` with `Replicate()` on `(h, w)` mesh dims (they're replicated across spatial ranks).
- After `fully_shard(...)`: `Shard(0)` on `fsdp` mesh dim, `Replicate()` on others.
- After `parallelize_module(...)` in task 06: additional `Shard(out_ch_dim)` or `Shard(in_ch_dim)` on the `tp` mesh dim.

## Steps

1. **Build the unified mesh** in `Distributed.__init__`. Drop the separate "data group" / "spatial group" / "fsdp group" fields; everything is a sub-mesh of the unified mesh.
2. **Convert `_AutogradAllReduce`** (`model_torch_distributed.py:46-70`) into a DTensor `redistribute` call where used. In most cases the call site should change from `spatial_reduce_sum(x)` to `x.redistribute(placements=[Replicate()])` (or `.full_tensor()` for the gathered result).
3. **Layer-by-layer migration**. Convert `ConditionalLayerNorm`, `SpectralConvS2`, `MLP` to consume DTensor inputs and produce DTensor outputs. Each layer migration is independently testable: flip the layer, run the regression test for that layer (`test_layers.py`, `test_s2convolutions.py`), confirm pass, move on.
4. **Distributed SHT integration**. Wrap the torch-harmonics distributed SHT in a small adapter that takes DTensors in, calls the underlying op on local tiles, and returns DTensors with the appropriate placements. This is the one place where the abstraction leaks slightly — document it.
5. **Drop the per-param spatial gradient hook** entirely. With DTensor, the gradient is automatically reduced through `redistribute(Partial -> Replicate)` semantics.
6. **DCP under DTensor**. DCP natively handles DTensor state dicts. Task 04's checkpoint code should largely Just Work after this migration; verify.

## Acceptance criteria

- All existing parallel tests in `fme/core/distributed/parallel_tests/` pass with no numerical change vs. pre-migration.
- The custom `_AutogradAllReduce` autograd Function is removed from the codebase (or marked deprecated and unused).
- The custom `_register_spatial_grad_hooks` method is removed or its body is empty (no-op).
- `ModelTorchDistributed` is renamed or replaced; the `Distributed` class no longer holds backend-specific spatial state — it holds a `DeviceMesh`.
- Stepper-side `wrap_module` calls (`fme/core/step/single_module.py:284`, etc.) are unchanged externally.

## Verification

1. **Bit-identical numerics** before vs. after migration on the existing `parallel_tests/test_step.py` regression. (DTensor uses the same underlying NCCL collectives, so the numerics should be identical, not merely "within tolerance".)
2. `make cpu_test_all_parallel` passes in roughly the same wall-clock as before.
3. Gradients on parameters under `(fsdp=2, spatial=(2,2))` match the pre-migration values to within bitwise equality on a fixed seed.

## Risks

- **Torch-harmonics distributed SHT API stability**. The adapter in step 4 is an integration point with an external library; if torch-harmonics changes its distributed API, this adapter is the one place that needs to follow.
- **Performance regression**. DTensor adds a small dispatch overhead per op. For the SFNO this is negligible (the SHT and Conv2d dominate), but worth measuring. If the regression is >5% on the test matrix, profile and consider explicit `local_tensor()` escapes in hot paths.
- **PyTorch version dependency**. DTensor APIs have been stabilizing through PyTorch 2.4-2.7. Pin the version in this task's PR and document any patches needed for older versions.

## Notes / handoff log

- This task is the right time to also drop the env-var-only configuration of spatial parallelism (the YAML config from task 02 should be the only path).
- The migration is risky enough that it should land in **two separate PRs**: (a) introduce the unified `DeviceMesh` in `Distributed` while keeping the old code paths, (b) flip layers one-at-a-time. This task file describes the end state; an executing agent should propose the PR split as part of starting work.
- Coordinate with task 06's author: TP's correctness depends on this migration being complete on the affected layers.
