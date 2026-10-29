#!/bin/bash
# Hann re-evaluation on CPU: baselines (all noise levels) and the convolutional runs.
cd /home/lcadame/Vinicius_Thesis/Vinicius
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=10
echo "inicio $(date '+%F %T')"
for s in 0 0.0001 0.001 0.003 0.01 0.03 0.1; do
  venv/bin/python evaluate_run.py --window hann --device cpu --baseline identity wiener --noise-std $s 2>&1 | grep -E "^baseline|Error|Traceback"
done
venv/bin/python evaluate_run.py --window hann --device cpu results/A_srcnn_res_lam0_seed0 results/A_dscms_res_lam0_seed0 \
  results/A_prusr_res_lam0_seed0 results/B_srcnn_res_lam0.001_seed0 results/B_srcnn_res_lam0.01_seed0 \
  results/B_srcnn_res_lam0.1_seed0 results/B_srcnn_res_lam1_seed0 2>&1 | grep -E "\[test|SKIPPED|Error|Traceback"
echo "fim $(date '+%F %T')"
