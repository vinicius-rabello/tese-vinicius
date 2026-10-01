#!/bin/bash
# Pilot, second pass: the 7 configs the clean-tree guard refused in the first
# pass (the main working tree had uncommitted edits). Runs from a detached git
# worktree at HEAD, so edits in the main tree can't affect it; results still go
# to /home/lcadame/Vinicius_Thesis/Vinicius/results_pilot. Then evaluates all 10 pilot runs (finished ones are
# re-evaluated identically).
cd /tmp/claude-1000/-home-lcadame-Vinicius-Thesis-Vinicius/6f0b8e26-24a1-42bb-8726-4e0f50ee2043/scratchpad/pilot_wt
source /home/lcadame/Vinicius_Thesis/Vinicius/venv/bin/activate
echo "pilot2 start $(date '+%F %T') at commit $(git rev-parse --short HEAD)"
for f in pilot_configs/*.yaml; do
  n=$(basename $f .yaml)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > /home/lcadame/Vinicius_Thesis/Vinicius/results_pilot/smi_$n.log &
  SMI=$!
  python train.py --config $f 2>&1 | tr '\r' '\n' | grep -E "completed_|Traceback|Error|out of memory" | grep -v "^  "
  kill $SMI
  echo "$n peak_MiB=$(sort -n /home/lcadame/Vinicius_Thesis/Vinicius/results_pilot/smi_$n.log | tail -1)"
done
echo "training done $(date '+%F %T')"
python evaluate_run.py /home/lcadame/Vinicius_Thesis/Vinicius/results_pilot/P_*_seed0 2>&1 | grep -vE "hycom_|Loading|Loaded|Dataset shape|Warning|warn"
echo "pilot2 done $(date '+%F %T')"
