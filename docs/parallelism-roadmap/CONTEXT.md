# Context: Why This Roadmap Exists

## The problem

The user wants to **train the noise-conditioned ("stochastic") SFNO at higher resolution and have it fit in memory** on multi-node HPC (Slurm + NCCL/IB).

Today the repo has two distributed backends that matter:
- `torch` — vanilla DDP (`fme/core/distributed/torch_distributed.py:176-187`). Every rank holds a full copy of params, grads, and optimizer state.
- `model` — bespoke spatial parallelism, sharding **activations** over `H × W` via PhysicsNeMo `DistributedManager` + torch-harmonics distributed SHT (`fme/core/distributed/model_torch_distributed.py:89-435`). Inside each spatial group, DDP layers on top — so weights are still replicated.

Neither shards weights. The repo has **no FSDP, FSDP2 (`fully_shard`), DTensor, `ZeroRedundancyOptimizer`, `torch.distributed.checkpoint` (DCP), or `torch.distributed.tensor.parallel` anywhere** — verified by grep over `fme/`.

## The memory framing

Per-rank GPU memory for SFNO training breaks into four buckets:

1. **Parameters** (`P` bytes). Dominated by `SpectralConvS2` weights of shape `[num_groups, modes_lat, out_ch/g, in_ch/g, 2]` complex (`fme/core/models/conditional_sfno/s2convolutions.py:137-144`). `modes_lat` scales with `lmax`, which scales with resolution. **Doubling resolution roughly doubles spectral parameters per block.**
2. **Gradients** (`P` bytes).
3. **Optimizer state** — Adam carries `2 × P` fp32 bytes plus a fp32 master copy: `~3 × P` fp32. Total bucket-1+2+3 cost is roughly `~16-20 bytes/param` at common precisions. **Replicated on every rank under both existing backends.**
4. **Activations** — scale with `B × T × C × H × W × num_layers`. Doubling resolution **quadruples** activations.

What each parallelism axis cuts:

| Tool | Cuts buckets | Where it lives |
|---|---|---|
| Activation checkpointing | 4 | Already present (`fme/core/optimization.py:18-72`) |
| bf16 autocast | 4 (½) | Already present |
| Spatial parallelism | 4 (by H×W factor) | Already present (`model` backend) |
| DDP | nothing | Already present |
| **FSDP / FSDP2** | **1 + 2 + 3** | **Absent** |
| Tensor parallelism | 1 + 2 + 3 + per-rank channel-axis activations | Absent |
| CPU/NVMe param offload | 1 + 2 + 3 (trades for PCIe) | Absent |

At higher resolution, spatial parallelism handles the activation explosion well, but **buckets 1–3 are growing with `lmax` and there is currently no way to shard them**. That is the bind. FSDP2 is the headline missing capability.

## Why FSDP2 *and* spatial parallelism (not "instead of")

They shard orthogonal things — FSDP shards weights along the data-parallel group, spatial parallelism shards activations along the model-parallel group. They compose under one `DeviceMesh`:

```
mesh = (fsdp_dp = N,  spatial_h = H,  spatial_w = W)
```

The composition is the actual production answer. FSDP alone leaves activations exploding at high resolution. Spatial alone leaves params/grads/opt-state replicated. Both together, with a shared mesh, give a bounded per-rank memory footprint at arbitrary resolution within hardware limits.

## Why the model is ready, the framework isn't

The stochastic SFNO architecture is FSDP-friendly:
- 12 default FNO blocks, each an `nn.Module` — natural per-block `fully_shard` unit.
- `MLP` is two 1×1 `Conv2d` (`fme/core/models/conditional_sfno/layers.py:360-416`) — classic megatron-style TP target.
- `ConditionalLayerNorm` AdaLN projections are 1×1 `Conv2d` (`layers.py:141-214`) — channel-local, FSDP-clean.
- `SpectralConvS2` weights are a single large complex tensor per block — cleanly shardable along channel axes.

The work is all in the distributed framework, not in the SFNO. That is what this roadmap covers.

## Production-readiness honest read

- **Single-node training of stochastic SFNO at moderate resolution**: production-ready today.
- **Multi-node training at moderate resolution**: production-capable for current model sizes; rank-0-only checkpointing (`fme/core/generics/trainer.py:633-671`) is the latent risk.
- **Higher-resolution stochastic SFNO that doesn't fit per-GPU**: **not production-ready**. There is no path to reduce parameter / gradient / optimizer-state memory; spatial parallelism alone cannot fix it.

This roadmap closes that gap.

## What "good" looks like at the end

A user can write something like:

```yaml
parallelism:
  data_parallel_size: 8
  fsdp_size: 8                  # equal to dp → full-shard within node
  hybrid_shard: true            # replicate across nodes
  spatial_h: 2
  spatial_w: 2
  tensor_parallel_size: 1       # opt-in once 06 lands
  mixed_precision:
    param_dtype: bf16
    reduce_dtype: fp32
  activation_checkpoint:
    after_n_forward_steps: 1
```

…launch via `torchrun` or `srun`, and have the framework allocate a single 4-D `DeviceMesh`, shard weights via FSDP2, shard activations via spatial parallelism, save & resume sharded checkpoints via DCP, and produce numerics matching a single-rank baseline within tolerance. Tasks 01–07 build that picture.
