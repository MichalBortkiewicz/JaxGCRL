#!/bin/bash
# In-job resume test for one run (PLAN.md Phase 3): start, SIGKILL after the 2nd saved eval, rerun (must
# resume from the saved step), let it finish (DONE), rerun again (must SKIP). Prints PASS/FAIL lines.
# usage: bash sweeps/resume_test.sh <set> <run_name>
set -u
source "${SSAC_ENV_SH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/env.sh}"
SET=$1; NAME=$2; DIR=$REPO/checkpoints/$SET/$NAME
L1=$REPO/logs/$SET/${NAME}_rt1_${SLURM_JOB_ID:-x}.out; L2=${L1/_rt1_/_rt2_}; L3=${L1/_rt1_/_rt3_}
echo "=== RESUME TEST $NAME"
setsid bash $REPO/sweeps/run_one.sh $SET $NAME > $L1 2>&1 &
PID=$!
# wait until a checkpoint with >= 2 epochs exists (or the run ends)
until [ -f $DIR/latest.pkl ] && $PY -c "import pickle,sys; d=pickle.load(open('$DIR/latest.pkl','rb')); sys.exit(0 if d['meta']['epochs_done']>=2 else 1)" 2>/dev/null; do
  kill -0 $PID 2>/dev/null || break; sleep 5
done
if ! kill -0 $PID 2>/dev/null; then echo "FAIL $NAME: run ended before the 2nd checkpoint (see $L1)"; tail -30 $L1; exit 1; fi
SAVED=$($PY -c "import pickle; d=pickle.load(open('$DIR/latest.pkl','rb')); print(d['meta']['env_steps'], d['meta']['epochs_done'])")
kill -9 -- -$PID 2>/dev/null; sleep 3
echo "killed after checkpoint: env_steps epochs = $SAVED"
bash $REPO/sweeps/run_one.sh $SET $NAME > $L2 2>&1; RC2=$?
STEP=$(echo $SAVED | cut -d' ' -f1)
grep -q "resumed from step $STEP " $L2 && echo "PASS resumed from step $STEP" || { echo "FAIL no 'resumed from step $STEP' in $L2"; grep -n "resumed\|Error\|error" $L2 | head; }
[ -f $DIR/DONE ] && echo "PASS DONE written (exit $RC2)" || { echo "FAIL no DONE (exit $RC2)"; tail -40 $L2; }
$PY - <<PY
import pandas as pd
m = pd.read_csv("$DIR/metrics.csv")
steps = m["step"].tolist()
print("metrics steps:", steps)
d = m.drop_duplicates("step", keep="last")
ok = d["step"].is_monotonic_increasing and d["step"].iloc[-1] > $STEP
print(("PASS" if ok else "FAIL"), "step counter continues after resume; last eval/episode_success:",
      d["eval/episode_success"].round(2).tolist())
PY
bash $REPO/sweeps/run_one.sh $SET $NAME > $L3 2>&1
grep -q "SKIP $NAME" $L3 && echo "PASS SKIP on rerun" || { echo "FAIL rerun did not skip"; tail $L3; }
grep -E "^=== $NAME exit|W&B run" $L2 | head -3
