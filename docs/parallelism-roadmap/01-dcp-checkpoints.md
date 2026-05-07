# 01 — Replace rank-0 checkpointing with DCP

**Status**: IN PROGRESS — helper module landed; trainer integration deferred to 01a
**Depends on**: —
**Blocks**: 03 (FSDP2 backend cannot save/load full state dicts cleanly without this)
**Estimated size**: M (1-3 days)

## Landed in this iteration

- `fme/core/generics/_checkpointing.py` — DCP save/load helpers with
  collective and non-collective modes, atomic writes, legacy-file detection,
  and a deprecation-warning legacy reader.
- `fme/core/generics/test_checkpointing.py` — round-trip, atomicity,
  partial-load, and rank-handling unit tests.

The trainer continues to write legacy single-file checkpoints. `_restore_checkpoint`
also continues to use the legacy path. The DCP helpers are ready for use
but not yet wired into the trainer.

## Why the trainer wiring was deferred

DCP's `dcp.load(template, ...)` requires the template to have all keys
present, including per-parameter optimizer-state entries
(``state[i]["exp_avg"]`` etc.). A freshly-built optimizer has an empty
``state`` dict, so a naive DCP load into the fresh template loses the
saved optimizer state.

The correct pattern uses PyTorch's
``torch.distributed.checkpoint.state_dict.set_state_dict`` /
``set_optimizer_state_dict`` helpers, which know how to apply a loaded
optimizer state dict to a fresh optimizer (mirroring
``Optimizer.load_state_dict`` semantics). That refactor naturally lands
alongside FSDP (task 03) since FSDP-sharded optimizer state requires
those same helpers anyway.

A separate **task 01a** (sub-task) tracks the trainer wiring:

1. Use ``get_state_dict`` / ``set_state_dict`` from
   ``torch.distributed.checkpoint.state_dict`` for model + optimizer.
2. Add a ``save_dcp_checkpoint: bool = False`` flag to ``TrainConfig``
   (default off until inference loaders are also updated).
3. Update ``load_stepper`` and ``load_stepper_config`` in
   ``fme/ace/stepper/single_module.py`` (and downscaling counterparts)
   to auto-detect DCP directories and load via the same helpers.
4. Switch the default to DCP once inference loaders work.

The remainder of this file describes the end-state of task 01 (helper
module + trainer integration + inference loader updates).

## Goal

Replace the existing rank-0-only full-state-dict checkpoint format with `torch.distributed.checkpoint` (DCP). DCP supports saving and loading sharded tensors, resumes into a different mesh / world size, and avoids rank-0 RAM and save-time bottlenecks at multi-node scale. This is a **prerequisite for FSDP** (a sharded model produces a sharded ckpt) and a real production-readiness improvement on its own.

## Why this is foundational

Today's checkpoint code (`fme/core/generics/trainer.py:633-671`) does:
- Save: `if dist.is_root(): torch.save(stepper.get_state(), tmp_path); os.replace(tmp_path, final_path)`
- Restore: `_restore_checkpoint()` loads a single dict.

That format **assumes one logical state dict** and is incompatible with FSDP-sharded parameters. It also creates a real bottleneck under multi-node training — rank-0 RAM, host I/O, slow save, no resume from a different topology.

## Files touched

Primary:
- `fme/core/generics/trainer.py:633-671` (`save_checkpoint`, `restore_checkpoint`).

Secondary (state plumbing):
- `fme/ace/stepper/parameter_init.py:224` and any `get_state` / `load_state` on the steppers — confirm they expose `state_dict()` / `load_state_dict()` semantics compatible with DCP's planner.
- `fme/core/optimization.py` — optimizer state dict handling.
- `fme/core/distributed/distributed.py` — may need a `Distributed.is_dcp_available()` helper.

Tests:
- `fme/core/distributed/parallel_tests/test_regression.py` already does a save→load loop; extend or fork into a DCP-specific regression test.

## Steps

1. **Read the current contract**. `Trainer.save_checkpoint` writes a dict containing `stepper`, `ema`, `optimizer`, and metadata. Catalogue every key.
2. **Define the DCP layout**. Two complementary patterns — pick one and document it:
   - **Single shared planner**: one `DCP.save(state, storage_writer=FileSystemWriter(path))` call where `state` is the same dict; DCP's default planner shards tensors across ranks. This is the simplest migration.
   - **Per-component DCP files**: separate `model/`, `optimizer/`, `meta/` subdirectories under `path`. More verbose, easier to debug.
   Recommend the first.
3. **Implement**:
   ```python
   import torch.distributed.checkpoint as dcp
   from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
   from torch.distributed.checkpoint.state_dict import (
       get_state_dict, set_state_dict, StateDictOptions,
   )
   ```
   Use `get_state_dict(model, optimizers, options=StateDictOptions(full_state_dict=False, cpu_offload=False))` to produce a sharded state dict that DCP can write.
4. **Backwards compatibility for restoring older single-file ckpts**: detect format by presence of `.metadata` (DCP) vs. a single `.tar` / `.pt` (legacy). Keep the legacy path alive for at least one release; emit a deprecation warning.
5. **Atomic save semantics**: DCP writes a directory of files. Replicate the existing `os.replace`-based atomicity by writing to `<path>.tmp/` then `os.rename(<path>.tmp, <path>)` once `dcp.save` returns.
6. **Metadata & EMA**: non-tensor metadata (epoch, step, EMA decay state, RNG state) goes in a small companion JSON / pickle file under the DCP directory. Document the schema in a comment block.
7. **Resume into a different mesh**: DCP handles this natively when both saver and loader use the same model definition. Add a unit test that saves under spatial=(1,1) and restores under spatial=(2,1) (and vice versa).

## Acceptance criteria

- New checkpoints are DCP directories (`<name>/.metadata`, `<name>/__0_0.distcp`, …) instead of `.tar` files.
- Legacy `.tar` checkpoints still load with a deprecation warning.
- Saving on a 4-rank cluster takes O(P/4) wall-clock, not O(P), at the I/O layer (validate with `time` on a synthetic 1 GB-param model).
- Resume across topology change (1→2 ranks, 2→4 ranks) reproduces validation loss within numerical tolerance.
- `make cpu_test_all_parallel TEST_PATH=fme/core/distributed/parallel_tests/test_regression.py` passes.

## Verification

1. Single-rank save/load round-trip produces bitwise-identical params after one optimizer step.
2. Multi-rank save/load round-trip (spatial=(2,1)) produces bitwise-identical params.
3. Cross-topology resume (save 1-rank, restore 2-rank) produces validation loss within `atol=1e-5, rtol=1e-4` of the never-saved baseline.
4. Legacy `.tar` ckpts in `parallel_tests/testdata/` continue to load.

## Open design questions

- Does the trainer need an explicit `checkpoint_format: dcp | legacy` knob, or is auto-detection on load + always-DCP on save sufficient? Recommend the latter unless there's a real downstream tool that consumes legacy `.tar`.
- Where does EMA state live? EMA is held by `EMAConfig.resume_ema_ckpt_path` (`fme/core/...`); confirm whether EMA params should be in the same DCP directory or a sibling.

## Notes / handoff log

- The existing test pattern in `parallel_tests/test_regression.py:115` (load module → wrap → ...) is a good starting point for a DCP regression test.
- Atomicity: DCP does *not* guarantee partial writes are recoverable if the process is killed mid-save. The `<path>.tmp/` → `os.rename` wrapper preserves the existing trainer guarantee.
- DCP requires `torch.distributed` to be initialized even for a "single-rank save". For the `NonDistributed` path, fall back to the legacy single-file save.
