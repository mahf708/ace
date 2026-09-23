#!/bin/bash
# Queue the sep26v2 pilot campaign in priority order, as ONE bundled Slurm job
# rather than one job per run (see bundle.sh / sbatch-bundle.sh).
#
#     ./submit-campaign.sh --dry-run              # print what would be queued
#     ./submit-campaign.sh --preflight            # stage + validate, queue nothing
#     ./submit-campaign.sh                        # queue P1..P3 as one bundle
#     ./submit-campaign.sh --max-priority 5       # ...including the tail
#     ./submit-campaign.sh --only LG01            # one experiment, by id
#     ./submit-campaign.sh --only LG01 --qos regular --time 02:00:00
#
# Every matching run is staged individually through run-train.sh --no-submit
# (config validation, the dirty-worktree refusal, wandb identity, FME_NODES
# sizing, all unchanged from a single-run submission), then handed to
# bundle.sh as one bundle file. bundle.sh submits a single sbatch job sized to
# the sum of the runs' nodes; sbatch-bundle.sh carves the allocation into
# disjoint node sets and starts one srun step per run inside it, requeueing
# the whole job together on a walltime signal. A run already queued or
# running on its own (e.g. one of the runs already in flight) is refused
# rather than folded in, so re-running this script never disturbs a run that
# is already going.
#
# --qos and --time set FME_QOS / FME_TIME, which bundle.sh turns into sbatch
# overrides for the one bundle job. WALLTIME is what matters, not the QOS.
# Measured 2026-09-07 over gpu_regular jobs of 3-8 nodes: a <=2 h request
# waits a median 5.3 h, everything from 2-4 h up waits 33-59 h. Backfill is
# the only way in (priority is a per-QOS constant plus age; fairshare has
# weight 0), and backfill only takes short jobs.
#
# Do NOT reach for `--qos preempt` on the strength of its shorter pending list.
# It preempts only debug_preempt/overrun/sparewarmer -- never gpu_regular -- so
# it buys no position, and it is itself preemptible by gpu_interactive and
# resv_shared. Same shape, same window: preempt waits a median 39.2 h against
# regular's 5.5 h. See TODO E1; this campaign lost 4.4 h learning it.
#
# --reservation is not supported for a bundled submission: a reservation's
# nodes are hbm80g while this campaign's runs stage at hbm40g, and bundle.sh
# has no partition/qos/constraint override for it the way the single-run path
# did. Refuse rather than silently ignore it.
#
# Priorities are 1..5 and the default cap is 3. P1 is the deterministic
# reference, which five arms difference against and which therefore has to
# finish first; P2 is the mechanism block; P3 the single-factor arms that carry
# the remaining claims. P4 and P5 are the tail, dropped first if the charge
# budget bites. sep26v2 is this two-run pilot, everything at P1; it has its
# own directory and its own submit script, so there is no shared priority
# space with sep26 or aug26 to defend against.
#
# Reads MANIFEST.tsv, which generate-campaign.sh writes. Every column it uses is
# named in the header, so a column added to the manifest does not shift this
# script's parsing.
#
# run-train.sh refuses a dirty worktree, so a campaign submission fails fast
# rather than queueing half a campaign against uncommitted code.
#
# On the node budget: at 40 concurrent nodes out of 1408 hbm40g in gpu_ss11,
# this campaign is charge-bound rather than concurrency-bound, so there is no
# reservation to overflow and no ordering constraint beyond priority.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
MANIFEST="$EXP/runs/MANIFEST.tsv"
RUN="$HERE/run-train.sh"
BUNDLE="$HERE/bundle.sh"

DRY=0
PRE=0
ONLY=""
MAXP=3
BUNDLE_NAME_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)      DRY=1; shift ;;
        --preflight)    PRE=1; shift ;;
        --only)         ONLY="${2:?--only needs an experiment id or run id}"; shift 2 ;;
        --reservation)  echo "--reservation is not supported for a bundled submission (see the header); cancel and use ./bundle.sh directly with FME_CONSTRAINT if you need it" >&2; exit 2 ;;
        --qos)          export FME_QOS="${2:?--qos needs a name}"; shift 2 ;;
        --time)         export FME_TIME="${2:?--time needs HH:MM:SS}"; shift 2 ;;
        --max-priority) MAXP="${2:?--max-priority needs a number}"; shift 2 ;;
        --bundle-name)  BUNDLE_NAME_ARG="${2:?--bundle-name needs a name}"; shift 2 ;;
        *) echo "usage: $0 [--dry-run|--preflight] [--only EXP] [--max-priority N]" >&2
           echo "              [--qos NAME] [--time HH:MM:SS] [--bundle-name NAME]" >&2
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

# Same default as run-train.sh, so a bundled submission lands where an
# individual ./run-train.sh atm <runid> would have.
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${PSCRATCH}/sep26v3}"

BUNDLE_FILE=""
if [ "$DRY" != 1 ] && [ "$PRE" != 1 ]; then
    BUNDLE_FILE=$(mktemp "${TMPDIR:-/tmp}/sep26v2-bundle.XXXXXX")
    trap 'rm -f "$BUNDLE_FILE"' EXIT
fi

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
        # Collected here rather than submitted individually: bundle.sh stages
        # and validates every run the same way run-train.sh --no-submit does,
        # then queues the whole set as one job. A run already queued or
        # running on its own is refused by bundle.sh rather than folded in.
        printf '%s\t%s\n' "$runid" "$CAMPAIGN_ROOT" >> "$BUNDLE_FILE"
    fi
done < "$MANIFEST"

echo
if [ "$DRY" = 1 ]; then
    echo "$count runs, $total nodes concurrent, ~$hours node-hours (dry run, nothing submitted)"
elif [ "$PRE" = 1 ]; then
    echo "$count runs, $total nodes -- all staged and validated, nothing queued"
else
    [ "$count" -gt 0 ] || { echo "nothing matched; nothing to bundle"; exit 0; }
    NAME=${BUNDLE_NAME_ARG:-sep26v2-p${MAXP}-$(date +%Y%m%dT%H%M%S)}
    JOBID=$(BUNDLE_NAME="$NAME" "$BUNDLE" "$BUNDLE_FILE" --go)
    echo "$count runs, $total nodes submitted as one bundle, ~$hours node-hours"
    echo "bundle job: $JOBID"
fi
exit 0
