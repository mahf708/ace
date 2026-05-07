# 04 — Compose FSDP2 with the existing spatial backend

**Status**: NOT STARTED
**Depends on**: 03
**Blocks**: 05, 06, 07
**Estimated size**: L (3-7 days). The hardest task in the roadmap.

## Goal

Run FSDP2 (sharded params/grads/opt-state along the **data-parallel** axis) and the existing spatial parallelism (sharded activations along the `H × W` model-parallel axis) **together** under one `DeviceMesh`. This is the configuration that actually addresses the user's stated need: train the noise-conditioned SFNO at higher resolution with the model fitting in memory.

## Why this is non-trivial

The existing `ModelTorchDistributed` does two things that conflict with naive FSDP composition:

1. **Per-parameter spatial gradient hooks** (`fme/core/distributed/model_torch_distributed.py:361-388`). These call `all_reduce` over the spatial group on each parameter's `.grad` after backward. FSDP2 instead calls `reduce_scatter` on the unsharded gradient inside its own communication hook, then frees the unsharded grad. If the spatial hook fires **after** FSDP's reduce-scatter, it sees a sharded grad and produces wrong values. The spatial reduction must fire **on the unsharded gradient, before FSDP reduce-scatters**.

2. **Custom autograd-aware all-reduce inside layers** (`_AutogradAllReduce`, `model_torch_distributed.py:46-70`, used by `spatial_reduce_sum`). This is fine — it operates on activations, which FSDP doesn't touch. No conflict here.

The composition design is therefore: keep FSDP2's reduce-scatter for the data axis, but interpose a spatial all-reduce on the unsharded gradient before FSDP performs its reduce-scatter.

## Files touched

Primary:
- `fme/core/distributed/model_torch_distributed.py:46-70,361-388` — adapt gradient-hook plumbing.
- `fme/core/distributed/fsdp_distributed.py` (from task 03) — add a "compose with spatial group" mode.
- `fme/core/distributed/distributed.py:21-131` — backend selection becomes a 2-D choice (fsdp_size, spatial_h, spatial_w).
- `fme/core/distributed/config.py` (from task 02) — add `fsdp_size` and `hybrid_shard` fields with cross-axis validation.

Tests:
- `fme/core/distributed/parallel_tests/test_step.py` — extend with FSDP×spatial cells.
- New: `fme/core/distributed/parallel_tests/test_fsdp_spatial.py`.

## Mesh shape

```python
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    "cuda",
    (fsdp_size, spatial_h, spatial_w),
    mesh_dim_names=("fsdp", "h", "w"),
)
```

The product `fsdp_size × spatial_h × spatial_w` must equal world size. The "data parallel group" used by `Distributed.get_sampler()` is the `fsdp` axis. The "spatial group" used by torch-harmonics distributed SHT is the (`h`, `w`) sub-mesh.

## Design — the gradient-hook composition

FSDP2 exposes a `register_fsdp_forward_method` and per-module communication hooks via `module.set_unshard_in_backward()` and friends. The cleanest insertion point is a **pre-reduce-scatter callback on each FSDP module** that:

1. All-reduces the unsharded `.grad` over the spatial mesh (replaces the existing per-param hook in `_register_spatial_grad_hooks`).
2. Returns control to FSDP, which then `reduce_scatter`s along the FSDP axis.

Pseudocode:
```python
from torch.distributed.fsdp import fully_shard

def wrap_module(self, module):
    fsdp_mesh = self._mesh["fsdp"]
    spatial_mesh = self._mesh["h", "w"]   # (h × w) sub-mesh

    for block in module.blocks:
        fully_shard(block, mesh=fsdp_mesh, mp_policy=mp_policy)
        # After fully_shard, block is an FSDPModule. Register a pre-comm hook:
        block.register_pre_backward_comm_hook(
            lambda grads, mesh=spatial_mesh: _all_reduce_over(grads, mesh)
        )
    fully_shard(module, mesh=fsdp_mesh, mp_policy=mp_policy)
    return module
```

The exact API for "pre reduce-scatter hook" in FSDP2 is `module.set_post_optim_event` / `module.unshard_in_backward` — confirm against the PyTorch version pinned in the repo. If FSDP2 does not yet expose a clean pre-comm hook, fall back to:

- Use FSDP2's `set_reduce_scatter_divide_factor` to compensate for the spatial rank count.
- Implement the spatial all-reduce as a `torch.autograd.Function` (subclass of `_AutogradAllReduce`) inserted into the model graph, so the gradient is summed across spatial ranks **as part of backward**, before FSDP sees it.

Pick whichever works; document the choice in the task file.

## Steps

1. **Build the 3-D mesh** `(fsdp, h, w)` and expose it on `Distributed`. Update `data_parallel_rank`, `total_data_parallel_ranks`, `get_sampler`, and `get_local_slices` to consult the right sub-mesh axes. Most existing call sites should keep working unchanged because they go through `Distributed`.
2. **Replace the existing `_register_spatial_grad_hooks`** with the composition described above. Verify the math: with `spatial_h=H, spatial_w=W, fsdp=N`, each parameter's gradient should be `sum_over_spatial / N` after both reductions. (Equivalent to the existing single-rank gradient when applied to the same input batch.)
3. **DCP under composition**. The state dict has DTensor parameters with two sharding dimensions: `Shard("fsdp")` and `Replicate()` over the spatial mesh (because spatial replicates weights). DCP handles this; verify by saving and reloading.
4. **Spatial buffers**. Distributed SHT keeps Legendre buffers per-spatial-rank. These must NOT be sharded by FSDP. Confirm by inspecting the FSDP-wrapped module: buffers should remain plain tensors, not DTensors.
5. **Tests**:
   - Numerics: single-rank `.pt` baseline vs. `(fsdp=2, spatial=(2,1))`, `(fsdp=2, spatial=(1,2))`, `(fsdp=4, spatial=(1,1))`, `(fsdp=2, spatial=(2,2))`. All within tolerance.
   - Memory: report params/grads/opt-state per-rank under each cell; should be `1/fsdp_size` of single-rank.
   - Activations: report peak activation memory under each cell; should be `1/(spatial_h × spatial_w)` of pre-spatial baseline.

## Acceptance criteria

- A 4-rank cluster can run `(fsdp=2, spatial_h=1, spatial_w=2)` and produce single-rank-equivalent numerics within tolerance.
- An 8-rank cluster can run `(fsdp=2, spatial_h=2, spatial_w=2)`. Both axes' memory wins compose: param memory = ½ baseline, activation memory = ¼ baseline.
- The matrix `make cpu_test_all_parallel` is extended with at least 4 new cells covering FSDP×spatial combinations and continues to pass in <10 minutes.
- DCP checkpoints save/resume across topology changes including changes to `fsdp_size` independent of changes to spatial.

## Verification

1. End-to-end training step under `(fsdp=2, spatial=(1,2))` on a small noise-conditioned SFNO config: loss curve matches the spatial-only baseline within `atol=1e-4, rtol=1e-3` over 100 steps.
2. Memory measurement: instrumentation script from task 00 confirms expected per-bucket reductions.
3. Save under `(fsdp=2, spatial=(2,1))`, restore under `(fsdp=4, spatial=(1,1))`, continue training: validation loss matches the never-saved baseline.
4. Throughput sanity: under `(fsdp=4, spatial=(1,1))` vs DDP×4, FSDP throughput is no worse than 1.5× slower (modulo any tuning, which is task 07's concern).

## Open design questions

- **Order of composition with EMA**. Today EMA stores a CPU-side shadow of model parameters. Under FSDP each rank's shadow only mirrors its own shard. Pick: (a) per-rank shadow, no gather (cheapest, EMA is implicitly sharded — fine for resume but `gather`-required for inference handoff); (b) full gather to root rank (memory-heavy on root). Recommend (a) plus a DCP integration so EMA shadows are also sharded and resumable.
- **Inference handoff**. After training, the user typically runs inference under DDP or single-rank. Adding a `state_dict(full_state_dict=True)` extraction utility (`torch.distributed.checkpoint.state_dict.get_model_state_dict(..., options=StateDictOptions(full_state_dict=True))`) gives a portable inference checkpoint. Document.

## Notes / handoff log

- This is the task most likely to need PyTorch-version-specific tweaks. FSDP2's hook API has been moving in 2.4 → 2.5 → 2.6. Pin the version in the task PR's commit message and link to the relevant PyTorch RFC.
- The existing `ModelTorchDistributed.wrap_module` is conceptually replaced by the composed wrap. Keep the file alive for the SHT helpers (`get_sht`, `get_isht`, `get_disco_conv_s2`) which task 05 will eventually move.
- Once 04 lands, the user's stated production goal is met for current model sizes. 05/06/07 are scaling refinements, not foundational gaps.
