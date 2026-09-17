#!/bin/bash
# Copy the evaluation dataset from CFS to Lustre before running evaluations.
#
#     ./stage-data.sh                    # the 2040s, the held-out block
#     ./stage-data.sh '204[0-5]'         # just the years a 1-year pass reaches
#     ./stage-data.sh '19[4-9]' '20[0-3]'   # several year globs at once
#     ./stage-data.sh --training         # everything training reads, to the
#                                        # root run-train.sh defaults to
#
# Why this exists.  The training template points at the project tree under
# /global/cfs, which compute nodes reach through DVS.  The evaluator's read
# pattern -- every variable of a 20-step window, per initial condition, once
# per window -- is the one DVS handles worst.  MEASURED on 2026-09-05, two
# concurrent 8-IC evaluations on four nodes:
#
#     CFS through DVS   84.0 s per window, ranks in `dvsipc_wait_for_response`
#                       while the GPUs that had data sat at 100%
#     staged on Lustre  13.5 s per window
#
# A single run against CFS managed ~25 s per window, so the filesystem was the
# bottleneck and concurrency made it worse rather than better.  Staged, two at
# once each beat one run on CFS by a factor of two.
#
# The whole decade is 120 monthly files, about 300 GB, and copies in 77 s at
# 3.3 GB/s with twelve streams -- it repays itself inside the first run.
#
# Then point the generator at it:
#     export EVAL_DATA_ROOT=$PSCRATCH/sep26v2-data
# or pass --data-root.  The generator refuses a staged root that is missing
# files the rollout needs, since a short glob gives a short dataset and not an
# error.
#
# PSCRATCH is purged, so expect to run this again on a cold campaign.
set -euo pipefail

SRC=${EVAL_DATA_SOURCE:-/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run}
DST=${EVAL_DATA_ROOT:-${PSCRATCH:?PSCRATCH must be set}/sep26v2-data}
STREAMS=${STAGE_STREAMS:-12}

# Training reads more than the eval decade: 1940-1990 and 2000-2040 for the two
# training subsets, 1990-1995 for validation, and the inference dataset carries
# no `subset:` at all, so it globs whatever is there. Staging the whole record
# is the only version of this that cannot silently come up short -- a short
# glob gives a short dataset, not an error.
if [ "${1:-}" = "--training" ]; then
    shift
    DST=${FME_DATA_ROOT:-/pscratch/sd/m/mahf708/v3.LR.historical_0101.aigo/run}
    set -- '19[4-9]' '20[0-9]'
fi
[ $# -gt 0 ] || set -- 204

PATTERNS=()
for y in "$@"; do PATTERNS+=("v3.LR.historical_0101.aigo.eam.h0.${y}*.nc"); done
mkdir -p "$DST"

cd "$SRC"
mapfile -t FILES < <(ls "${PATTERNS[@]}" 2>/dev/null)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "no files matching ${PATTERNS[*]} under $SRC" >&2
    exit 1
fi
echo "staging ${#FILES[@]} files matching ${PATTERNS[*]}"
echo "  from $SRC"
echo "  to   $DST"

# -n so a re-run only fetches what is missing: staging is idempotent and a
# partial copy is resumable.
printf '%s\n' "${FILES[@]}" | xargs -P "$STREAMS" -I{} cp -n {} "$DST/"

echo "done -- $(ls "$DST"/*.nc | wc -l) files in $DST"
echo "now:  export EVAL_DATA_ROOT=$DST"
echo "      (run-train.sh already defaults FME_DATA_ROOT to the training root)"
