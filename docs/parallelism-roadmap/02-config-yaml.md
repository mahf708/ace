# 02 — Promote parallelism config from env vars to YAML

**Status**: NOT STARTED
**Depends on**: —
**Blocks**: nothing strictly, but every later task gains a clean place to opt in
**Estimated size**: S–M (~1 day, plus migration of existing configs)

## Goal

Today, parallelism choice is **env-var-only**: `FME_DISTRIBUTED_BACKEND`, `FME_DISTRIBUTED_H`, `FME_DISTRIBUTED_W` (`fme/core/distributed/distributed.py:107-122`). That's fine for a switch in a Makefile, but not for reproducible experiments — the parallelism choice is not part of the YAML config, so an experiment archive doesn't carry the topology it ran on.

Add a `ParallelismConfig` dataclass that lives in the YAML, validated in `__post_init__` per the project's config conventions (`AGENTS.md` → "Config design"). Backwards compatibility: when the YAML block is absent, fall back to env vars (the current behavior).

## Files touched

Primary:
- New file: `fme/core/distributed/config.py` (or fold into `fme/core/distributed/distributed.py`).
- `fme/core/distributed/distributed.py:107-131` — `Distributed.get_instance()` consults YAML config first, env vars second.
- Wherever the top-level YAML config is composed (look in `fme/ace/train/`, `fme/core/cli.py`) — add an optional `parallelism: ParallelismConfig` field.

Tests:
- `fme/core/distributed/test_distributed.py` — new tests for the config dataclass and its precedence over env vars.

## Proposed shape

```python
@dataclass
class ParallelismConfig:
    """Distributed parallelism configuration.

    All sizes default to 1 (no parallelism in that axis). The product of
    all axes must equal the world size. When the entire block is absent,
    backend selection falls back to the FME_DISTRIBUTED_* env vars.
    """
    data_parallel_size: int = 1     # DDP / data-parallel replica count
    fsdp_size: int = 1              # added when task 03 lands
    spatial_h: int = 1              # height-axis spatial parallelism
    spatial_w: int = 1              # width-axis spatial parallelism
    tensor_parallel_size: int = 1   # added when task 06 lands
    hybrid_shard: bool = False      # added when task 07 lands

    def __post_init__(self):
        if self.spatial_h < 1 or self.spatial_w < 1:
            raise ValueError("spatial dims must be >= 1")
        if self.fsdp_size < 1:
            raise ValueError("fsdp_size must be >= 1")
        # cross-axis validation lives here when more axes land
```

YAML usage:
```yaml
parallelism:
  spatial_h: 2
  spatial_w: 2
```

## Steps

1. **Define `ParallelismConfig`** with only the fields that make sense today (`data_parallel_size`, `spatial_h`, `spatial_w`). Other fields (`fsdp_size`, `tensor_parallel_size`, `hybrid_shard`) get added by tasks 03/06/07 — leave a comment marking the extension point.
2. **Plumb it through `Distributed.get_instance()`**. The classmethod becomes:
   ```python
   def get_instance(cls, config: ParallelismConfig | None = None) -> "Distributed":
       if config is not None:
           return cls(spatial_parallelism=(config.spatial_h, config.spatial_w))
       # ... existing env-var fallback ...
   ```
   Threading the config to `get_instance()` cleanly is the main design question; one option is a setter (`Distributed.set_config(config)`) called early in `Trainer.__init__`, before the first `get_instance()`.
3. **Migrate one example YAML** under `configs/experiments/2025-01-17-diffusion/` to use the new block (with `spatial_h=spatial_w=1`, i.e. preserving current default behavior). Confirm it round-trips through the loader.
4. **Add deprecation warning** when env vars are used and YAML block is absent: `"FME_DISTRIBUTED_* env vars are deprecated; use parallelism: in YAML"`. Keep working for one release.
5. **Document in `docs/training_config.rst`** (the existing Sphinx config docs).

## Acceptance criteria

- New YAML block round-trips through the config loader.
- When YAML block is absent, env vars work as before.
- When both are set, YAML wins; env vars emit a warning.
- All existing experiment YAMLs continue to work without modification.
- Unit tests cover validation errors (negative size, etc.).

## Verification

1. `make cpu_test_all_parallel` (with the new config block in the test YAML) produces the same results as without.
2. `python -m pytest fme/core/distributed/test_distributed.py` passes.
3. Run one short training job under both env-var configuration and YAML configuration; loss curves should match.

## Notes / handoff log

- The project's general config-design rule (`AGENTS.md`): validate in `__post_init__`, not at runtime. Cross-axis checks (`product == world_size`) belong in `__post_init__`.
- Convention from `AGENTS.md`: "Config classes loaded from user-specified yaml: append `Config` to the built type" — hence `ParallelismConfig`.
- For backward compatibility on training configs, this can break (per `AGENTS.md`); for *inference* configs, must keep working with deprecation warnings. Confirm there are no inference configs that touch parallelism today (probably none, since spatial parallelism is currently blocked at inference).
