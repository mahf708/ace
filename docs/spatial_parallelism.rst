Spatial parallelism
===================

Spatial parallelism splits a single sample's grid across several ranks, so a
model whose activations do not fit on one device can still be trained. It is a
*memory*-enabling technique, not automatically a throughput one: every
spectral block exchanges data between the ranks sharing a sample, so when a
full globe already fits, plain data parallelism will usually be faster.

This page describes what is supported today and, just as importantly, what is
refused. A configuration outside the supported set raises before the model
runs rather than producing a plausible but different answer.

Layout
------

Ranks are arranged as a three-dimensional ``data x h x w`` mesh. ``h`` splits
latitude and ``w`` splits longitude; the remaining ranks become data-parallel
replicas. Distributed spherical harmonic transforms run over ``h`` and ``w``,
DDP runs over ``data``, and the two are orthogonal.

.. code-block:: bash

    FME_DISTRIBUTED_BACKEND=model FME_DISTRIBUTED_H=2 FME_DISTRIBUTED_W=2 \
      torchrun --nproc-per-node 8 -m fme.ace.train config-train.yaml

Eight ranks with ``h=2, w=2`` gives four ranks per sample and two data-parallel
replicas. ``FME_DISTRIBUTED_BACKEND=torch`` (the default) is pure data
parallelism; ``none`` forces a single-process backend.

What is supported
-----------------

* Latitude-longitude grids whose height and width each divide evenly by the
  corresponding mesh dimension.
* Training the noise-conditioned SFNO with ``filter_type: linear``.
* Both ``noise_type`` settings. The conditioning noise is drawn over the
  global grid and sliced, so a seeded run reproduces the single-rank run.
* Area-weighted global means, sums, zonal means and the conservation
  correctors built on them.
* Aggregator-based evaluation.

What is refused, and why
------------------------

Each of these raises where the offending object is constructed, before the
first collective.

``global_layer_norm: true``
    ``nn.LayerNorm`` normalizes over an explicitly-shaped trailing block, so
    it is built over the global ``(C, H, W)`` while receiving this rank's
    tile. A distributed implementation needs local sums and sums-of-squares
    reduced over the spatial group, plus a placement for the affine
    parameters.

``spectral_lora_rank > 0`` with ``h > 1``
    The LoRA factors are indexed by spectral latitude mode and span the global
    set, while the input holds only this rank's modes. Localizing them makes
    them genuinely sharded parameters, which the gradient reduction and the
    checkpoint format do not yet represent. A ``w``-only decomposition is
    fine: the factors are then fully replicated.

``filter_type: makani-linear``
    Its spectral weight is allocated at the global mode extent with no local
    slicing in the forward pass.

HEALPix
    ``HEALPixOperations``' area-weighted reductions are local to each rank's
    tile, so tile sums and means would be reported as global quantities.

Inference data writers
    Every rank opens the same output path and serializes the tensor it is
    handed, which under spatial parallelism is a tile. Runs that write nothing
    and report through the aggregator are allowed. Note that multi-rank
    inference *without* spatial parallelism races on the same paths; that is a
    pre-existing property of the writers and currently only warns.

Uneven splits
    ``torch_harmonics`` will return ragged tiles, but ``gather`` and any
    collective sized from the local tile assume equal shapes, so a ragged
    split fails as a hang rather than an error. The rule applies to the data
    grid only -- the spectral ``(l, m)`` axes are split by torch-harmonics'
    own variable-size machinery, and ``mmax = W // 2 + 1`` is usually odd.

Ownership contracts
-------------------

Knowing where a tile lives is not enough; each piece of state also needs an
owner.

Activations
    Sharded over ``(h, w)``. Batches are sliced before the copy to device, so
    a rank moves only the bytes it uses.

Parameters
    Replicated. Every rank stores the global extent and slices in the forward
    pass, so its gradient is a partial sum over its tile and summing the
    partials over the spatial group reconstructs the single-rank gradient.
    ``fme.core.distributed.parameter_placement`` enforces this: a parameter
    sized differently across spatial co-ranks is refused, and replicated
    parameters are broadcast from the spatial-group root at wrap time, which
    is the counterpart to the broadcast DDP already performs over the data
    group.

Randomness
    Two scopes are implemented. Spatially-indexed draws (the conditioning
    noise) are generated over the global grid and sliced, so the realization
    depends on the seed alone. Spatially-shared draws (the input-dropout mask,
    which is per-sample rather than per-point) are broadcast over the spatial
    group. Data-parallel replicas currently share a seed and therefore draw
    identical noise for corresponding samples; see the known limitations
    below.

Checkpoints
    Replicated, saved by rank 0 only. Coherent precisely because no parameter
    is sharded -- which is why sharding one has to be refused rather than
    merely discouraged.

Known limitations
-----------------

* **Data-parallel replicas draw identical conditioning noise.** Every rank is
  seeded identically and draws the same shape, so only
  ``batch_size / n_data_ranks`` of the realizations in a global batch are
  independent. This predates spatial parallelism and affects plain data
  parallelism too. Fixing it means keying the draw on the global sample index,
  which changes existing multi-rank results and every committed baseline.
  Pinned by a strict xfail in ``parallel_tests/test_noise.py``.
* **Parameter memory does not fall with the decomposition.** Activations are
  sharded; parameters, gradients, optimizer state and EMA are replicated on
  every rank. Reducing those needs FSDP2 on a separate mesh axis, which in
  turn needs a distributed checkpoint format.
* **The conditioning noise field is materialized globally on every rank.**
  This is what buys decomposition invariance, and it cannot be chunked without
  changing the numbers -- torch's normal fill is not sequentially composable.
  A counter-based generator keyed by global position would give both.
* **Direct ``srun`` is untested.** The code path exists and is reachable, but
  CI has no Slurm, and the model backend initializes over TCP where the
  data-parallel backend uses a shared file. Use ``torchrun``.
* **Restarts cannot change the mesh.** A checkpoint written under one layout
  loads under another only because everything is replicated; nothing verifies
  it, and that stops being true as soon as any state is sharded.

Testing
-------

Tests marked ``parallel`` run under ``torchrun``:

.. code-block:: bash

    make cpu_test_all_parallel

Regression baselines under ``parallel_tests/testdata`` must be generated by a
single-rank ``python -m pytest`` run and committed. Generating them under the
same decomposition they are then checked against compares a layout with itself
and proves nothing.
