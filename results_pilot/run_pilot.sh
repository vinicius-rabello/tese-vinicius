#!/bin/bash
# Short pilot: 2 full-data epochs per config through train.py, nvidia-smi
# sampled every 0.5 s, then a test-set evaluation of each run (spectra).
cd /home/lcadame/Vinicius_Thesis/Vinicius
source venv/bin/activate
echo "pilot start $(date '+%F %T')"
for f in results_pilot/configs/*.yaml; do
  n=$(basename $f .yaml)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > results_pilot/smi_$n.log &
  SMI=$!
  python train.py --config $f 2>&1 | tr '\r' '\n' | grep -E "completed_|Traceback|Error|out of memory" | grep -v "^  "
  kill $SMI
  echo "$n peak_MiB=$(sort -n results_pilot/smi_$n.log | tail -1)"
done
echo "training done $(date '+%F %T')"
python evaluate_run.py results_pilot/P_*_seed0 2>&1 | grep -vE "hycom_|Loading|Loaded|Dataset shape|Warning|warn"
echo "pilot done $(date '+%F %T')"
