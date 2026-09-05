#!/bin/bash
# Launch one evaluator config on one node of the current allocation.
#     ./run_eval.sh <run-dir-with-config.yaml> <node> [minutes]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../../.." && pwd)"
TORCHRUN="$REPO_ROOT/.venv/bin/torchrun"
RUN_DIR="${1:?run dir}"; NODE="${2:?node}"; MINUTES="${3:-120}"
PORT=$((29000 + RANDOM % 2000))
LOG="$RUN_DIR/out.log"
cd "$REPO_ROOT"
timeout "$((MINUTES * 60))" srun --jobid="$SLURM_JOB_ID" -N1 -n1 --gpus-per-node=4 --overlap -w "$NODE" \
    --export=ALL,FME_DISTRIBUTED_BACKEND=torch \
    "$TORCHRUN" --nproc-per-node 4 --master-port "$PORT" \
    -m fme.ace.evaluator "$RUN_DIR/config.yaml" > "$LOG" 2>&1
echo "exit $? $RUN_DIR"
