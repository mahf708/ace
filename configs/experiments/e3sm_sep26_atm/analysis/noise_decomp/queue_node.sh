#!/bin/bash
# Run evaluator configs one after another on one node of the current
# allocation, waiting for the node's GPUs to be idle before each launch.
#     ./queue_node.sh <node> <run-dir> [<run-dir> ...]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="${1:?node}"; shift
gpu_used() {
    srun --jobid="$SLURM_JOB_ID" -N1 -n1 --overlap -w "$NODE" \
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | paste -sd+ | bc 2>/dev/null || echo 999999
}
for RUN in "$@"; do
    until [ "$(gpu_used)" -eq 0 ] 2>/dev/null; do sleep 30; done
    echo "$(date +%H:%M:%S) launch $(basename "$RUN") on $NODE"
    "$HERE/run_eval.sh" "$RUN" "$NODE" 120
    sleep 20
done
echo "$(date +%H:%M:%S) queue on $NODE done"
