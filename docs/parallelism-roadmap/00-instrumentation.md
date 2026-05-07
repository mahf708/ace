# 00 — Memory instrumentation: validate the premise

**Status**: NOT STARTED
**Depends on**: —
**Blocks**: nothing strictly, but its result determines whether 03 is the right next step
**Estimated size**: S (~½ – 1 day)

## Goal

Empirically confirm that, at the user's target resolution for the noise-conditioned ("stochastic") SFNO, the residual per-GPU memory constraint after spatial parallelism + activation checkpointing + bf16 is **parameters + gradients + optimizer state** (buckets 1-3 in `CONTEXT.md`), not activations.

If the data show otherwise — e.g., activations still dominate even under spatial parallelism — the roadmap's task ordering should be revisited.

## Deliverable

A short markdown report (`docs/parallelism-roadmap/00-instrumentation-results.md`) plus the script that produced it (`scripts/profile_memory.py` or similar). The report contains a per-bucket memory breakdown for **at least three** resolutions and **at least two** spatial-parallelism configurations.

## Steps

1. **Pick configurations**:
   - Model: noise-conditioned SFNO from `fme/ace/registry/stochastic_sfno.py`. Use a config under `configs/experiments/2025-01-17-diffusion/` as the starting point.
   - Resolutions: current target, 2× current target, 4× current target (or whatever maps to the user's roadmap; document the choice).
   - Parallelism: single-rank, spatial=`(2,2)`, spatial=`(4,4)`. All under bf16 + activation checkpointing.

2. **Instrument** one training step using `torch.cuda.memory._record_memory_history(...)` and `torch.cuda.memory._snapshot()`. Group allocations into the four buckets:
   - Parameters: walk `model.named_parameters()` and sum `.numel() * .element_size()`.
   - Gradients: same, after a backward.
   - Optimizer state: walk `optimizer.state` and sum allocated tensors.
   - Activations: residual = peak memory − the three above − reserved framework overhead.

3. **Report** as a table:

   | Resolution | Parallelism | Params (GB) | Grads (GB) | Opt state (GB) | Activations (GB) | Peak (GB) |
   |---|---|---|---|---|---|---|

4. **Conclusion paragraph** addressing two questions:
   - At which resolution does spatial-parallel(4×4) cease to fit within an H100 (80 GB) or A100 (40 GB)?
   - Of that excess, what fraction is buckets 1–3 vs. bucket 4? This decides whether FSDP2 (cuts 1–3) or more aggressive activation handling (cuts 4) is the correct next investment.

## Acceptance criteria

- Script is reproducible: `python scripts/profile_memory.py --config <path> --resolution <res> --spatial <h>x<w>` produces the table line for that row.
- Results table covers at least 3×2 = 6 cells.
- Conclusion paragraph explicitly answers the two questions above.

## Verification

Re-run the script for one cell and confirm the numbers reproduce within 1% (CUDA caching allocator can introduce small variance — set `PYTORCH_NO_CUDA_MEMORY_CACHING=1` for the measurement run).

## Notes / handoff log

- This task is independent of all others and can run any time. It validates the premise behind the rest of the roadmap.
- The bucket-1-2-3 formula `~16-20 bytes/param` (`CONTEXT.md`, "memory framing") is a rule of thumb; the empirical number is what the report should publish.
- The activation-bucket residual will include CUDA workspace memory (cuDNN, NCCL buffers); call this out in the methodology section so it isn't read as "real" activation memory.
