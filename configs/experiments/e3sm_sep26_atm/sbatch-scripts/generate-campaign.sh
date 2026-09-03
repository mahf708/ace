#!/bin/bash
# Regenerate every config for the sep26 atmosphere ablation campaign.
#
#     ./generate-campaign.sh              # write ../runs
#     ./generate-campaign.sh --list       # the run list and the budget
#     ./generate-campaign.sh /some/dir    # write elsewhere
#
# The output is identical whoever runs it -- no username, no scratch path, no
# timestamp -- which is what lets several people share one campaign:
# regenerating is a no-op against a committed runs/, so nobody dirties the
# worktree and nobody has to commit before submitting. run-train.sh refuses a
# dirty worktree, so without that property only the generator's author could
# launch anything.
#
# Sizing is per run and comes out of the config rather than the sbatch file:
# nodes = batch_size / local_batch / 4. run-train.sh reads FME_NODES from the
# generated .env and passes --nodes to sbatch.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
EXP=$(dirname "$HERE")
REPO=$(cd "$EXP/../../.." && pwd)
GEN="$EXP/make_campaign.py"

run_py() { ( cd "$REPO" && uv run --quiet python "$@" ); }

if [ "${1:-}" = "--list" ]; then
    run_py "$GEN" --list
    exit 0
fi

OUT="${1:-$EXP/runs}"

# Clear stale output first. runs/ is entirely generated, and although the sparse
# delta convention means adding an *axis* renames nothing, changing an existing
# arm's delta does rename that arm's files -- and without this the old id
# lingers beside the new one and someone can launch an orphan by hand.
if [ -d "$OUT" ]; then
    find "$OUT" -maxdepth 1 -type f \( -name '*.yaml' -o -name '*.env' \) -delete
    rm -f "$OUT/MANIFEST.tsv"
fi

run_py "$GEN" --all -o "$OUT"

# Assert every emitted config says what its run id says it says. validate_config
# proves a config parses -- it passed aug26's E25, which raises on its first
# training batch -- so it is not a substitute for this.
echo
run_py "$EXP/check_campaign.py" --dir "$OUT"

echo
echo "next: ./submit-campaign.sh --dry-run"
