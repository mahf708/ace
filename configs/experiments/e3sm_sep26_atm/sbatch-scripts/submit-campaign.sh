#!/bin/bash
# Queue the sep26 campaign in priority order.
#
#     ./submit-campaign.sh --dry-run              # print what would be queued
#     ./submit-campaign.sh --preflight            # stage + validate, queue nothing
#     ./submit-campaign.sh                        # queue P1..P3
#     ./submit-campaign.sh --max-priority 5       # ...including the tail
#     ./submit-campaign.sh --only LG01            # one experiment, by id
#
# Priorities are 1..5 and the default cap is 3. P1 is the deterministic
# reference, which five arms difference against and which therefore has to
# finish first; P2 is the mechanism block; P3 the single-factor arms that carry
# the remaining claims. P4 and P5 are the tail, dropped first if the charge
# budget bites. sep26 has its own directory and its own submit script, so there
# is no shared priority space with aug26 to defend against.
#
# Reads MANIFEST.tsv, which generate-campaign.sh writes. Every column it uses is
# named in the header, so a column added to the manifest does not shift this
# script's parsing.
#
# run-train.sh refuses a dirty worktree, so a campaign submission fails fast
# rather than queueing half a campaign against uncommitted code. It also refuses
# to submit a run id that is already in your queue: two jobs writing ckpt.tar in
# one directory do not fail, they silently produce one corrupted run.
#
# On the node budget: at 40 concurrent nodes out of 1408 hbm40g in gpu_ss11,
# this campaign is charge-bound rather than concurrency-bound, so there is no
# reservation to overflow and no ordering constraint beyond priority.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
MANIFEST="$EXP/runs/MANIFEST.tsv"
RUN="$HERE/run-train.sh"

DRY=0
PRE=0
ONLY=""
MAXP=3

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY=1; shift ;;
        --preflight)    PRE=1; shift ;;
        --only)         ONLY="${2:?--only needs an experiment id or run id}"; shift 2 ;;
        --max-priority) MAXP="${2:?--max-priority needs a number}"; shift 2 ;;
        *) echo "usage: $0 [--dry-run|--preflight] [--only EXP] [--max-priority N]" >&2
           echo "       N is 1..3 for the arms that carry the claims, 4..5 for the tail" >&2
           exit 2 ;;
    esac
done

[ -f "$MANIFEST" ] || {
    echo "no $MANIFEST -- run ./generate-campaign.sh first" >&2; exit 1; }

# Resolve columns by NAME, from the header. The manifest carries provenance
# columns (rel, run_hours) that are for humans and may grow; positional parsing
# would break the moment one is added.
header=$(head -1 "$MANIFEST")
col() { awk -v want="$1" -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i==want){print i; exit}}' "$MANIFEST"; }
C_ID=$(col runid); C_LABEL=$(col exp); C_PRI=$(col priority)
C_NODES=$(col nodes); C_HOURS=$(col run_hours); C_NOTE=$(col note)
for c in "$C_ID" "$C_LABEL" "$C_PRI" "$C_NODES"; do
    [ -n "$c" ] || { echo "MANIFEST.tsv is missing a required column: $header" >&2; exit 1; }
done

total=0
count=0
hours=0
while IFS=$'\t' read -r -a f; do
    [ "${f[$((C_ID-1))]}" = "runid" ] && continue
    runid="${f[$((C_ID-1))]}"; label="${f[$((C_LABEL-1))]}"
    pri="${f[$((C_PRI-1))]}"; nodes="${f[$((C_NODES-1))]}"
    rh="${f[$((C_HOURS-1))]:-0}"; note="${f[$((C_NOTE-1))]:-}"
    [ "$pri" -le "$MAXP" ] || continue
    if [ -n "$ONLY" ] && [ "$label" != "$ONLY" ] && [ "$runid" != "$ONLY" ]; then
        continue
    fi
    total=$((total + nodes))
    count=$((count + 1))
    hours=$((hours + nodes * rh))
    if [ "$DRY" = 1 ]; then
        printf 'P%-2s %2s nodes %4s h  %-40s %s\n' "$pri" "$nodes" "$rh" "$runid" "$note"
        continue
    fi
    printf 'P%-2s %2s nodes  %s\n' "$pri" "$nodes" "$runid"
    # < /dev/null: the child inherits this loop's stdin, which is the manifest.
    # Anything it reads from stdin is a run that never gets submitted.
    if [ "$PRE" = 1 ]; then
        "$RUN" atm "$runid" --no-submit > /dev/null < /dev/null \
            || { echo "PREFLIGHT FAILED: $runid" >&2; exit 1; }
    else
        "$RUN" atm "$runid" > /dev/null < /dev/null
    fi
done < "$MANIFEST"

echo
if [ "$DRY" = 1 ]; then
    echo "$count runs, $total nodes concurrent, ~$hours node-hours (dry run, nothing submitted)"
elif [ "$PRE" = 1 ]; then
    echo "$count runs, $total nodes -- all staged and validated, nothing queued"
else
    echo "$count runs, $total nodes submitted, ~$hours node-hours"
fi
exit 0
