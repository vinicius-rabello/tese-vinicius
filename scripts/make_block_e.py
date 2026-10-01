"""Block E: multistage (4 stages) on one architecture plus the
continuous-training control (section 4.7).

    python scripts/make_block_e.py --arch <name> [--stage1-run <run_id>] [--lambda <value>]

Stage 1 is an existing run, by default A_<arch>_res_lam0_seed0 (block A,
seed 0). Stages 2..4 chain on the best.pt of the previous stage
(STAGE1_CHECKPOINT), with RESIDUAL false and LAMBDA 0: f_n is trained with
MSE on the normalized residual. The control is the stage-1 configuration
(residual, --lambda = stage 1's lambda) trained for exactly the epochs that
stage 1 and stage 2 took (FIXED_EPOCHS_FROM), so it must be queued after
stage 2. Queue order: the files in the order printed.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, lam_tag, write_config  # noqa: E402

N_STAGES = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=ARCHS)
    parser.add_argument("--stage1-run", help="run_id of stage 1 (default: A_<arch>_res_lam0_seed0)")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0,
                        help="lambda of stage 1, used by the control (default 0)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    stage1 = args.stage1_run or f"A_{args.arch}_res_lam{lam_tag(args.lam)}_seed0"
    if f"_lam{lam_tag(args.lam)}_" not in stage1:
        parser.error(f"--lambda {args.lam:g} does not match the stage-1 run {stage1}")

    paths, previous, chain = [], stage1, [stage1]
    for n in range(2, N_STAGES + 1):
        name = f"E_{args.arch}_stage{n}"
        paths.append(write_config("E", args.arch, 0.0, False, 0, run_name=name, extra={
            "STAGE": n, "STAGE1_CHECKPOINT": f"{args.results_dir}/{previous}/best.pt"}))
        previous = f"{name}_nores_lam0_seed0"
        chain.append(previous)
    paths.append(write_config("E", args.arch, args.lam, True, 0, run_name=f"E_{args.arch}_control", extra={
        "STAGE": "control", "FIXED_EPOCHS_FROM": [f"{args.results_dir}/{chain[0]}", f"{args.results_dir}/{chain[1]}"]}))
    print("\n".join(paths))
    print(f"{len(paths)} configs; stage 1 = {stage1}; the control trains the epochs of {chain[0]} + {chain[1]}")
