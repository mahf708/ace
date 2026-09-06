#!/bin/bash
# Generate and queue the offline evaluation of one or more trained arms.
#
#     ./submit-eval.sh --dry-run                       # print what would be queued
#     ./submit-eval.sh RF01.S01                        # one arm, scores pass
#     ./submit-eval.sh RF01.S01 --pass traj            # its trajectory pass
#     ./submit-eval.sh RF01.S01 --noise-ladder         # off/mean/fixed/half too
#     ./submit-eval.sh --all --pass scores             # every trained arm
#
# One job per eval directory. They are independent, so they queue as separate
# jobs rather than one array: an arm whose checkpoint is missing should not
# take the other twenty-five down with it, and the passes have different
# shapes (the scores pass is members-heavy and writes no trajectories, the
# trajectory pass is the reverse).
#
# The generator refuses a noise mode on a deterministic arm and refuses an
# initial-condition count that the loader cannot deal out evenly, so a bad
# shape fails here at submit time rather than minutes into an allocation.
#
# Arms whose checkpoint does not exist yet are skipped with a line saying so.
# That is the normal state of this campaign until RF02 finishes: nothing but
# RF01 has weights.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
REPO_ROOT=$(cd "$EXP/../../.." && pwd)
PYTHON="$REPO_ROOT/.venv/bin/python"
GENERATE="$EXP/make_eval_config.py"

: "${PSCRATCH:?PSCRATCH must be set}"
EVAL_ROOT="${EVAL_ROOT:-$PSCRATCH/sep26-eval}"

# Evaluations are I/O bound against CFS through DVS and six times faster on a
# staged copy (see stage-data.sh). Use one if it is there, and say so rather
# than silently choosing a filesystem for the caller.
if [ -z "${EVAL_DATA_ROOT:-}" ] && [ -d "$PSCRATCH/sep26-data" ]; then
    export EVAL_DATA_ROOT="$PSCRATCH/sep26-data"
    echo "using staged data at $EVAL_DATA_ROOT (unset EVAL_DATA_ROOT to read CFS)"
fi

ARMS=()
PASS=scores
LADDER=0
DRY=0
ALL=0
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --all)          ALL=1; shift ;;
        --pass)         PASS="${2:?--pass needs scores or traj}"; shift 2 ;;
        --noise-ladder) LADDER=1; shift ;;
        --dry-run)      DRY=1; shift ;;
        --ics|--nodes|--years|--members|--seed|--data-root)
                        EXTRA+=("$1" "$2"); shift 2 ;;
        -*)             echo "unknown option $1" >&2; exit 2 ;;
        *)              ARMS+=("$1"); shift ;;
    esac
done

if [ "$ALL" -eq 0 ] && [ ${#ARMS[@]} -eq 0 ]; then
    echo "usage: $0 [--all | <arm> ...] [--pass scores|traj] [--noise-ladder]" >&2
    exit 2
fi

# `keep` is the trained behaviour and is always evaluated. The rest are the
# mechanism ladder: each is one extra inference on weights that already exist,
# and together they say what the noise pathway is doing to the trajectory.
MODES=(keep)
if [ "$LADDER" -eq 1 ]; then
    MODES+=(off mean fixed half)
fi

SELECT=("${ARMS[@]}")
if [ "$ALL" -eq 1 ]; then
    # Expand here rather than passing --all through: each arm is generated and
    # queued on its own, so one refusal (a deterministic arm under a noise
    # mode, a missing checkpoint) skips that arm and not the rest.
    mapfile -t SELECT < <("$PYTHON" "$GENERATE" --list)
fi

QUEUED=0
SKIPPED=0
for mode in "${MODES[@]}"; do
    for arm in "${SELECT[@]}"; do
        # The generator prints the config path, or explains its refusal.
        if ! OUT=$(EVAL_ROOT="$EVAL_ROOT" "$PYTHON" "$GENERATE" \
                   "$arm" --pass "$PASS" --noise "$mode" "${EXTRA[@]}" 2>&1); then
            case "$OUT" in
                *"noise pathway"*) SKIPPED=$((SKIPPED + 1)); continue ;;
                *) echo "$OUT" >&2; exit 2 ;;
            esac
        fi
        CONFIG=$(echo "$OUT" | tail -1)
        DIR=$(dirname "$CONFIG")
        CKPT=$(grep '^checkpoint_path:' "$CONFIG" | awk '{print $2}')
        if [ ! -f "$CKPT" ]; then
            echo "skip $(basename "$DIR") -- no checkpoint at $CKPT"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        # shellcheck disable=SC1091
        NODES=$(grep '^FME_NODES=' "$DIR/eval.env" | cut -d= -f2)
        if [ "$DRY" -eq 1 ]; then
            echo "would queue $(basename "$DIR")  ${NODES} nodes"
            QUEUED=$((QUEUED + 1))
            continue
        fi
        JOB=$(sbatch --parsable \
            -A e3sm_g -q regular -C 'gpu&hbm40g' \
            -J "eval-$(basename "$DIR")" \
            --nodes="$NODES" --ntasks-per-node=1 --gpus-per-node=4 \
            --cpus-per-task=128 -t 06:00:00 \
            --output="$EXP/joblogs/%x-%j.out" \
            --wrap "$HERE/run-eval.sh $DIR --nodes $NODES --minutes 330")
        echo "queued $JOB  $(basename "$DIR")  ${NODES} nodes"
        QUEUED=$((QUEUED + 1))
    done
done

echo "-- $QUEUED queued, $SKIPPED skipped"
