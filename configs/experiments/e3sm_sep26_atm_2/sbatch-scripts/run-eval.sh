#!/bin/bash
# Run one generated evaluation config inside the current allocation.
#
#     ./run-eval.sh <eval-dir> [--nodes N] [--node nidXXXXXX] [--minutes M]
#
# <eval-dir> is what make_eval_config.py printed the config.yaml into.  The
# script is the payload, not the allocation: it expects to be inside one
# already, which is true both under sbatch-eval.sh and in an interactive
# `salloc`.  That is deliberate -- the same line that a batch job runs is the
# line you can run by hand while working out why an arm behaves oddly, so the
# two cannot drift.
#
# Three launcher traps, each paid for once in this campaign:
#
#   * Launch through torchrun, not bare `srun -n4`. FME takes its device from
#     LOCAL_RANK, which plain srun does not set, so all four ranks land on
#     cuda:0 and OOM. It reads as the arm needing more memory; the giveaway is
#     several processes listed against device 0.
#   * Pin the node with -w when sharing an allocation. Two overlapping steps
#     each asking for all four GPUs of one node collide and the loser dies
#     with no traceback -- it simply stops logging.
#   * Give it a deadline. A stalled evaluator holds the allocation silently:
#     the 16-IC shape stalls in an uninterruptible DVS wait and never returns.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../../../.." && pwd)
TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"

EVAL_DIR="${1:?usage: run-eval.sh <eval-dir> [--nodes N] [--node nid] [--minutes M]}"
shift
NODES=""
NODE=""
MINUTES=240

while [ $# -gt 0 ]; do
    case "$1" in
        --nodes)   NODES="${2:?--nodes needs a count}"; shift 2 ;;
        --node)    NODE="${2:?--node needs a nodename}"; shift 2 ;;
        --minutes) MINUTES="${2:?--minutes needs a count}"; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done

CONFIG="$EVAL_DIR/config.yaml"
[ -f "$CONFIG" ] || { echo "no config.yaml in $EVAL_DIR" >&2; exit 2; }
[ -x "$TORCHRUN" ] || { echo "no torchrun at $TORCHRUN" >&2; exit 2; }
[ -n "${SLURM_JOB_ID:-}" ] || {
    echo "run-eval.sh runs inside an allocation; use sbatch-eval.sh to make one" >&2
    exit 2
}

# The .env carries the node count the config was generated for, and the WandB
# identity.  Sourcing it here rather than baking the names into the config
# keeps run identity out of the file that describes the science.
if [ -f "$EVAL_DIR/eval.env" ]; then
    set -a; . "$EVAL_DIR/eval.env"; set +a
fi
NODES="${NODES:-${FME_NODES:-1}}"

PORT=$((29000 + RANDOM % 2000))

# Multi-node needs a rendezvous both torchruns agree on. srun starts one
# torchrun per node and each would otherwise default its master address to
# 127.0.0.1, so the two never meet: no error, no log past the startup banner,
# and the allocation is held until the deadline. Name the first node of the
# placement explicitly and let c10d do the rendezvous.
if [ -n "$NODE" ]; then
    MASTER="${NODE%%,*}"
else
    MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
fi

LOG="$EVAL_DIR/out.log"
[ -f "$LOG" ] && mv "$LOG" "$LOG.$(date +%Y%m%dT%H%M%S)"

PLACEMENT=()
[ -n "$NODE" ] && PLACEMENT=(-w "$NODE")

echo "eval  $EVAL_DIR"
echo "nodes $NODES  deadline ${MINUTES}m  log $LOG"

cd "$REPO_ROOT"
set +e
timeout "$((MINUTES * 60))" srun --jobid="$SLURM_JOB_ID" \
    -N "$NODES" -n "$NODES" --gpus-per-node=4 --overlap "${PLACEMENT[@]}" \
    --export=ALL,FME_DISTRIBUTED_BACKEND=torch \
    "$TORCHRUN" --nnodes "$NODES" --nproc-per-node 4 \
    --rdzv-backend c10d --rdzv-id "$SLURM_JOB_ID-$PORT" \
    --rdzv-endpoint "$MASTER:$PORT" \
    -m fme.ace.evaluator "$CONFIG" > "$LOG" 2>&1
RC=$?
set -e

if [ $RC -eq 124 ]; then
    echo "TIMED OUT after ${MINUTES}m -- $EVAL_DIR" >&2
    echo "check for a rank in state D (dvsipc_wait_for_resp): the 16-IC shape stalls" >&2
elif [ $RC -ne 0 ]; then
    echo "FAILED rc=$RC -- see $LOG" >&2
    tail -20 "$LOG" >&2
else
    echo "ok $EVAL_DIR"
fi
exit $RC
