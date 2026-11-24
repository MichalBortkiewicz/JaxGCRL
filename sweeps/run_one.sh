#!/bin/bash
# usage: bash sweeps/run_one.sh <set> <run_name>
# Idempotent (TORCH_JOB_SUBMISSION.md §3.4): SKIP if DONE/FAILED_NAN, resume from latest.pkl, else fresh.
set -uo pipefail
source "${SSAC_ENV_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh}"
SET=$1; NAME=$2
CODE=${CODE_DIR:-$REPO}             # frozen worktree for a set, if one is used (PLAN.md §4.B.7)
DIR=$REPO/checkpoints/$SET/$NAME
LINE=$(awk -F'\t' -v n="$NAME" '$1==n' $REPO/scripts/$SET/runs.tsv)
[ -n "$LINE" ] || { echo "unknown run $NAME in set $SET"; exit 1; }
FLAGS=$(echo "$LINE" | cut -f3)
for M in DONE FAILED_NAN; do [ -f "$DIR/$M" ] && { echo "SKIP $NAME ($M)"; exit 0; }; done
mkdir -p "$DIR"
CMD="$PY $CODE/run.py $FLAGS --exp-name $NAME --run-dir $DIR"
if [ ! -f "$DIR/README.md" ]; then
  printf "# %s\nSet %s (see scripts/%s/README_set.md). Command:\n\n    %s\n\nRun log:\n" "$NAME" "$SET" "$SET" "$CMD" > "$DIR/README.md"
fi
echo "- $(date '+%F %T') | job ${SLURM_JOB_ID:-none}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID} | $(hostname) | git $(git -C $CODE rev-parse --short HEAD) | $([ -f $DIR/latest.pkl ] && echo resume || echo fresh)" >> "$DIR/README.md"
cd $CODE
PYTHONPATH=$CODE $PY -c "import os,jaxgcrl; assert jaxgcrl.__file__.startswith('$CODE'), jaxgcrl.__file__" || { echo "WRONG jaxgcrl"; exit 1; }
echo "=== $NAME :: $(date)"; echo "$CMD"
T0=$(date +%s)
PYTHONPATH=$CODE $CMD
RC=$?
echo "=== $NAME exit $RC after $(( $(date +%s) - T0 )) s"
exit $RC
