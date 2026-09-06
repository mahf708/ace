#!/bin/bash
# Copy the evaluation dataset from CFS to Lustre before running evaluations.
#
#     ./stage-data.sh                    # the 2040s, the held-out block
#     ./stage-data.sh '204[0-5]'         # just the years a 1-year pass reaches
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
#     export EVAL_DATA_ROOT=$PSCRATCH/sep26-data
# or pass --data-root.  The generator refuses a staged root that is missing
# files the rollout needs, since a short glob gives a short dataset and not an
# error.
#
# PSCRATCH is purged, so expect to run this again on a cold campaign.
set -euo pipefail

SRC=${EVAL_DATA_SOURCE:-/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run}
DST=${EVAL_DATA_ROOT:-${PSCRATCH:?PSCRATCH must be set}/sep26-data}
YEARS=${1:-204}
STREAMS=${STAGE_STREAMS:-12}

PATTERN="v3.LR.historical_0101.aigo.eam.h0.${YEARS}*.nc"
mkdir -p "$DST"

cd "$SRC"
mapfile -t FILES < <(ls $PATTERN)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "no files matching $PATTERN under $SRC" >&2
    exit 1
fi
echo "staging ${#FILES[@]} files matching $PATTERN"
echo "  from $SRC"
echo "  to   $DST"

# -n so a re-run only fetches what is missing: staging is idempotent and a
# partial copy is resumable.
printf '%s\n' "${FILES[@]}" | xargs -P "$STREAMS" -I{} cp -n {} "$DST/"

echo "done -- $(ls "$DST"/*.nc | wc -l) files in $DST"
echo "now:  export EVAL_DATA_ROOT=$DST"
