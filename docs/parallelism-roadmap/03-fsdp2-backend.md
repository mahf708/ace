# 03 — Add an FSDP2 (`fully_shard`) backend

**Status**: NOT STARTED
**Depends on**: 01 (DCP checkpoints)
**Blocks**: 04, 05, 06, 07
**Estimated size**: L (3-7 days)

## Goal

Add a fourth distributed backend, `fsdp`, that uses PyTorch's modern FSDP2 (`torch.distributed.fsdp.fully_shard`) to shard parameters, gradients, and optimizer state across the data-parallel group. This is the headline missing capability identified in `CONTEXT.md`. It directly cuts the per-rank cost of buckets 1-3 (params + grads + opt-state), which today are fully replicated on every rank.

For this task: **FSDP-only mode**, no spatial parallelism. Composition with the existing spatial backend is task 04.

## Why FSDP2 specifically (not FSDP1)

- `fully_shard` is the per-module API that composes naturally with `DeviceMesh` and `DTensor` — required for tasks 05 and 06.
- FSDP2's communication hooks are accessible per-module, which makes the spatial-gradient-hook composition in task 04 tractable. FSDP1's monolithic `FullyShardedDataParallel` wrapper makes it nearly impossible.
- FSDP2 has cleaner mixed-precision semantics (`MixedPrecisionPolicy(param_dtype, reduce_dtype, output_dtype)`) and supports CPU param offload as a per-module policy.
- The PyTorch ecosystem (TorchTitan, the Modulus refactor, Llama recipes) is converging on FSDP2; FSDP1 is on a soft-deprecation path.

## Files touched

Primary:
- New file: `fme/core/distributed/fsdp_distributed.py` — `FSDPDistributed(DistributedBackend)`.
- `fme/core/distributed/distributed.py:21-131` — backend selection (`FME_DISTRIBUTED_BACKEND=fsdp`) and YAML routing (from task 02).
- `fme/core/distributed/base.py:107-...` — confirm `DistributedBackend.wrap_module` signature is sufficient; FSDP wrap is per-module shape-dependent, not a single outer wrap.

Stepper-side wrappers (these all call `dist.wrap_module(module)`):
- `fme/core/step/single_module.py:284`
- `fme/core/step/radiation.py:295`
- `fme/core/step/secondary_module.py:327`
- `fme/core/step/secondary_decoder.py:87`
- `fme/ace/step/fcn3.py:386`
- `fme/downscaling/_deterministic_models.py:104`
- `fme/downscaling/models.py:374`

Optimizer / training:
- `fme/core/optimization.py` — `GradScaler` is a no-op for bf16; confirm clip-grad-norm uses `torch.nn.utils.clip_grad_norm_` which needs `unscale_grads` semantics aware of FSDP. Use `torch.distributed.fsdp.fully_shard`'s native `clip_grad_norm_` helper, or upgrade to `torch.nn.utils.clip_grad_norm_` which works on DTensor in PyTorch ≥ 2.4.
- `fme/core/generics/trainer.py` — checkpoint integration with task 01's DCP plumbing.

Tests:
- `fme/core/distributed/parallel_tests/test_step.py` — extend with FSDP cells in the matrix.
- New: `fme/core/distributed/parallel_tests/test_fsdp.py`.

## Sharding policy

The natural FSDP unit for the noise-conditioned SFNO is **one wrap per FNO block**. Each block (`SphericalFourierNeuralOperatorBlock` in `fme/core/models/conditional_sfno/sfnonet.py:232-401`) is the smallest module whose forward is a meaningful compute window — large enough for `all_gather` overlap, small enough to keep per-block memory bounded.

Pseudocode:
```python
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=None,  # honor caller dtype
)

# inside FSDPDistributed.wrap_module:
def wrap_module(self, module):
    # Wrap each FNO block first (innermost), then the outer module.
    for block in module.blocks:        # iterate the .blocks attribute of SFNO
        fully_shard(block, mesh=self._fsdp_mesh, mp_policy=mp_policy)
    fully_shard(module, mesh=self._fsdp_mesh, mp_policy=mp_policy)
    return module
```

Critical: FSDP2 wraps **in-place** and does not produce a wrapper module. The returned object is the same `nn.Module` you passed in, mutated. This means existing `wrap_module` callers that bind to the return value (which they do) keep working.

## Steps

1. **Author `FSDPDistributed`** as a sibling of `TorchDistributed` and `ModelTorchDistributed`. Inherit from `DistributedBackend`. Reuse `TorchDistributed`'s init logic (`__init__`, lines 29-70 of `torch_distributed.py`) for process-group setup; differentiate only `wrap_module`.
2. **Build a 1-D `DeviceMesh`** named `"fsdp"`. (Multi-dim mesh comes in task 04.)
3. **Implement `wrap_module`** with the per-block + outer wrap pattern above. Detect the `.blocks` attribute by duck-typing — fall back to wrapping just the outer module if `.blocks` is absent (covers non-SFNO models).
4. **Implement `clip_grad_norm`** explicitly. With FSDP2, `torch.nn.utils.clip_grad_norm_` works on the model's parameters as DTensors (PyTorch ≥ 2.4) and reduces across the FSDP mesh internally. Validate the value matches single-rank baseline.
5. **DCP integration**: the saved checkpoint is sharded across the FSDP mesh. Use `torch.distributed.checkpoint.state_dict.get_state_dict(model, optimizer, options=StateDictOptions(full_state_dict=False))` to obtain a sharded state-dict. Task 01 sets up the directory layout; this task just plumbs the FSDP-aware extraction.
6. **Backend selection**:
   - YAML (task 02): `parallelism: { fsdp_size: <N> }` activates this backend. Validate `fsdp_size > 1` and `fsdp_size == world_size` (pure FSDP requires no other axes; mixed configurations are task 04).
   - Env: `FME_DISTRIBUTED_BACKEND=fsdp` plus `FME_FSDP_SIZE` for parity.
7. **Buffer handling**: FSDP2 leaves buffers replicated across the mesh by default — fine for SFNO's Legendre buffers as long as they're identical across data ranks. (They are: spatial parallelism would split them, but FSDP-only does not.)
8. **EMA + FSDP**: confirm the `EMAConfig` plumbing (used as a shadow set of params) handles DTensor parameters. Most likely fix: switch EMA storage to use `module.parameters()` directly so it sees DTensors.
9. **Tests**:
   - Add a baseline regression test under `parallel_tests/test_fsdp.py` following the AGENTS.md pattern: generate `.pt` baseline with single-rank `python -m pytest`, then verify under `torchrun --nproc-per-node 4 ... -m parallel`.
   - Extend the matrix in `Makefile`'s `cpu_test_all_parallel` target with FSDP cells. Watch for CPU FSDP support — `gloo` works for FSDP2 but is slow; document the constraint.

## Acceptance criteria

- `FME_DISTRIBUTED_BACKEND=fsdp torchrun --nproc-per-node 4 -m pytest -m parallel fme/core/distributed/parallel_tests/test_fsdp.py` passes on CPU and (if available) GPU.
- Single-step parameter values after one optimizer update match the single-rank baseline within `atol=1e-5, rtol=1e-4`.
- Per-rank parameter memory (measured via `torch.cuda.memory_allocated()` after model construction) is ~1/`fsdp_size` of the single-rank baseline.
- Checkpoints save and resume across `fsdp_size` topology changes (1 → 4 → 2).
- bf16 mixed precision is honored: forward activations are bf16, gradient reductions are fp32.

## Verification

1. Numerics: cross-backend comparison test in `test_fsdp.py` (single-rank `.pt` vs FSDP).
2. Memory: instrumentation harness from task 00 reports `params + grads + opt_state` is reduced by `~1/fsdp_size`.
3. Throughput: log iterations/sec for `fsdp_size=1,2,4` and confirm FSDP doesn't catastrophically degrade — expect ~10-20% slowdown vs DDP at 4 ranks before any tuning.
4. `make cpu_test_all_parallel` continues to pass for the `model` and `torch` backends (no regression).

## Open design questions

- **Per-block vs. per-layer wrap granularity**. Per-block is the natural compromise. If profiling shows `all_gather` of `SpectralConvS2` weights dominating, the next refinement is to wrap `SpectralConvS2` and `MLP` separately within each block. Defer until measured.
- **`reshard_after_forward`**. Default is `True` for non-root modules. For inference (no backward), set to `False` on the outer wrap to keep weights gathered across multiple forward passes. Document; do not change defaults.
- **Activation checkpointing under FSDP2**. `checkpoint_wrapper(use_reentrant=False)` composes correctly with `fully_shard`. Confirm by adding a cell to `test_fsdp.py` that combines AC + FSDP.

## Notes / handoff log

- FSDP2 lands in PyTorch ≥ 2.4. Check `requirements.txt` / `pyproject.toml` for the pinned PyTorch version; if it's older, this task includes bumping it. The Dockerfile uses `pytorch/pytorch:2.7.1-cuda12.8` so the runtime is fine.
- Mixed precision under FSDP2 is per-module via `MixedPrecisionPolicy`, not via `torch.amp.autocast`. The existing `autocast` in `fme/core/optimization.py:118-119` should still wrap the forward for any non-FSDP-managed compute (loss, optimizer steps under FSDP are fp32 by virtue of `reduce_dtype=fp32` + fp32 master).
- For the user's stated focus, **memory wins matter more than throughput wins**. Optimize the design for clarity and correctness; defer comm/compute overlap tuning until task 04 lands.
