#!/bin/bash
# Launch a smoke config on an IDLE node of the current allocation.
#
#     ./run_smoke.sh <smoke-dir> [--jobid ID] [--minutes 45] [--node nidXXXXXX]
#
# Every smoke failure in this campaign so far has been the launcher, not the
# arm. Three distinct ways, each of which looked like a result:
#
#   1. Bare `srun -n4` does not set LOCAL_RANK, so all four ranks pick cuda:0
#      and OOM. It reads as "this arm needs more memory". The giveaway is
#      several processes listed against device 0.
#   2. Two steps that each ask for all four GPUs on one node collide, and the
#      loser dies with NO traceback -- it just stops logging. Picking a
#      different node NAME is not enough; the node has to actually be free,
#      and a previous smoke run may still hold it.
#   3. A timeout shorter than dataset setup kills the run before the first
#      batch, which looks identical to a hang. Setup is ~10-15 min on the
#      3-year subset, so nothing below ~30 min is a real test.
#
# This script does all three correctly: finds a node with no GPU memory in
# use, launches through torchrun with absolute paths, and defaults to a
# 45-minute deadline. It refuses rather than guesses when no node is free.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"

SMOKE_DIR="${1:?usage: $0 <smoke-dir> [--jobid ID] [--minutes N] [--node nid]}"
shift
JOBID=""; MINUTES=45; NODE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --jobid)   JOBID="${2:?}"; shift 2 ;;
        --minutes) MINUTES="${2:?}"; shift 2 ;;
        --node)    NODE="${2:?}"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[ -x "$TORCHRUN" ] || { echo "no torchrun at $TORCHRUN; run 'uv sync'" >&2; exit 1; }
[ -f "$SMOKE_DIR/config.yaml" ] || { echo "no config.yaml in $SMOKE_DIR" >&2; exit 1; }

if [ -z "$JOBID" ]; then
    JOBID=$(squeue -u "$USER" -h -o "%i %P" | awk '$2!="resv"{print $1; exit}')
    [ -n "$JOBID" ] || { echo "no interactive allocation found; pass --jobid" >&2; exit 1; }
fi

NODES=$(scontrol show hostnames "$(squeue -j "$JOBID" -h -o '%N')")

# An idle node is one whose GPUs report zero memory in use. A node still
# holding a previous smoke run reports hundreds of MiB at minimum.
find_idle_node() {
    for n in $NODES; do
        used=$(srun --jobid="$JOBID" -N1 -n1 --overlap -w "$n" \
                 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 999999)
        if [ "${used:-999999}" -eq 0 ] 2>/dev/null; then echo "$n"; return 0; fi
    done
    return 1
}

if [ -z "$NODE" ]; then
    NODE=$(find_idle_node) || {
        echo "no idle node in $JOBID -- every node still has GPU memory in use." >&2
        echo "Wait for the running smoke tests, or pass --node to override." >&2
        exit 1
    }
fi

PORT=$((29000 + RANDOM % 2000))
LOG="$SMOKE_DIR/out.log"

# Never destroy a previous result. A log that already records logged steps is
# evidence; move it aside rather than deleting it, so a careless rerun cannot
# erase the run it is about to be compared against.
if [ -f "$LOG" ]; then
    if grep -aq "batch_loss" "$LOG" 2>/dev/null; then
        KEEP="$LOG.$(date -r "$LOG" +%Y%m%dT%H%M%S)"
        mv "$LOG" "$KEEP"
        echo "  kept previous passing log at $(basename "$KEEP")"
    else
        rm -f "$LOG"
    fi
fi

echo "smoke   $SMOKE_DIR"
echo "  node    $NODE (idle)   job $JOBID   port $PORT"
echo "  deadline ${MINUTES} min   log $LOG"

set +e
timeout "$((MINUTES * 60))" srun --jobid="$JOBID" -N1 -n1 --gpus-per-node=4 --overlap -w "$NODE" \
    --export=ALL,FME_DISTRIBUTED_BACKEND=torch \
    "$TORCHRUN" --nproc-per-node 4 --master-port "$PORT" \
    -m fme.ace.train "$SMOKE_DIR/config.yaml" > "$LOG" 2>&1
rc=$?
set -e

steps=$(grep -ac "batch_loss" "$LOG" 2>/dev/null || echo 0)
tb=$(grep -ac "Traceback" "$LOG" 2>/dev/null || echo 0)
first=$(grep -aoE "batch_loss': tensor\([0-9.]+" "$LOG" 2>/dev/null | head -1 | grep -oE "[0-9.]+$" || true)
last=$(grep -aoE "batch_loss': tensor\([0-9.]+" "$LOG" 2>/dev/null | tail -1 | grep -oE "[0-9.]+$" || true)

# rc 124 is our own deadline, which is a PASS when steps were logged: the
# point is that the arm trains, not that it finishes an epoch.
if [ "$steps" -gt 0 ] && [ "$tb" -eq 0 ]; then
    echo "SMOKE PASS  steps=$steps  loss ${first} -> ${last}  (exit $rc)"
    exit 0
fi
echo "SMOKE FAIL  steps=$steps  tracebacks=$tb  exit=$rc"
echo "  Before reading this as a result, rule out the launcher:"
grep -aoE "DUE to SIGNAL [A-Za-z]+|OutOfMemoryError|No such file or directory" "$LOG" 2>/dev/null | sort -u | sed 's/^/    /'
exit 1
