#!/bin/bash
# Guarded submit (TORCH_JOB_SUBMISSION.md §5.7 + PLAN.md §4.B).
# usage: bash sweeps/safe_sbatch.sh <number_of_new_queue_items> <sbatch args...> <file.sbatch> [script args]
# Refuses if the user's queue (all projects) would exceed 12 items, or if our running GPU jobs would exceed 6,
# or if the sbatch file lacks the account line or a std_ job name. Appends the job to JOBS_LEDGER.md.
MAX=${SSAC_MAX_QUEUE:-12}; MAX_OURS_RUNNING=12
LEDGER=${SSAC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/JOBS_LEDGER.md
ACCOUNT=${SLURM_ACCOUNT_NAME:?set SLURM_ACCOUNT_NAME (source env.sh)}
NEW=$1; shift
NOW=$(squeue -u $USER -h -r | wc -l)
if [ $((NOW + NEW)) -gt $MAX ]; then echo "REFUSED: $NOW queued + $NEW new > $MAX. Wait or ask the user."; exit 1; fi
FILE=""
for a in "$@"; do [[ "$a" == *.sbatch ]] && { FILE=$a; break; }; done
[ -n "$FILE" ] || { echo "REFUSED: no .sbatch file in args"; exit 1; }
grep -q -- "--account=$ACCOUNT" "$FILE" || { echo "REFUSED: $FILE has no account line"; exit 1; }
grep -qE -- "--job-name=std_" "$FILE" || { echo "REFUSED: $FILE job name must start with std_"; exit 1; }
OURS=$(squeue -u $USER -h -r -o "%j %T" | awk '$1 ~ /^std_/' | wc -l)
if [ $((OURS + NEW)) -gt $MAX ]; then echo "REFUSED: too many std_ items"; exit 1; fi
# Potential concurrency: our running jobs + pending ones not held by an array throttle (they can start any time).
POT=$(squeue -u $USER -h -r -o "%j %T %r" | awk '$1 ~ /^std_/ && $1 !~ /analysis|unittest/ && ($2=="RUNNING" || ($2=="PENDING" && $3!="JobArrayTaskLimit"))' | wc -l)
echo "our running + startable-pending std_ items before this submit: $POT (limit $MAX_OURS_RUNNING; new array caps add to this)"
GPUJOB=1; grep -q -- "--partition=cpu_short" "$FILE" && GPUJOB=0
if [ $GPUJOB = 1 ] && [ "${FORCE_OVER_6:-0}" != "1" ] && [ $((POT + 1)) -gt $MAX_OURS_RUNNING ]; then
  echo "REFUSED: would allow more than $MAX_OURS_RUNNING of our GPU jobs to run at once. Use --dependency, or wait."; exit 1
fi
OUT=$(sbatch --parsable "$@") || { echo "sbatch failed"; exit 1; }
JID=${OUT%%;*}
NAME=$(grep -oE -- "--job-name=[^ ]+" "$FILE" | head -1 | cut -d= -f2)
echo "| $(date '+%F %T') | $JID | $NAME | $FILE ${*: -3} | $NEW | submitted |" >> $LEDGER
echo "submitted $JID ($NAME), queue now $((NOW + NEW))"
