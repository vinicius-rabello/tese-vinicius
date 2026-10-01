"""Block C: the best lambda from block B on the 5 architectures, residual,
seeds 0-2. Seed 0 of each sweep architecture is skipped: block B already
trained it with that lambda (14 runs with one sweep architecture, 13 with two).

    python scripts/make_block_c.py --lambda <value> --sweep-arch <name> [<name> ...]
    python scripts/make_block_c.py --lambda-cnn <v> --lambda-op <v> --sweep-arch srno srcnn

--lambda-cnn / --lambda-op: when the SRCNN and SRNO sweeps choose different
lambdas (DECISIONS.md, selection rules), SRCNN's lambda for SRCNN, DSC/MS,
PRUSR and SRNO's for SRNO, SoFoNO. Seed 0 of a sweep architecture is still
the block-B run with its family's lambda.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, add_lambda_arguments, lam_tag, lambdas_per_arch, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_lambda_arguments(parser)
    parser.add_argument("--sweep-arch", required=True, choices=ARCHS, nargs="+")
    args = parser.parse_args()
    lam = lambdas_per_arch(args, parser)
    paths = [write_config("C", arch, lam[arch], True, seed)
             for seed in (0, 1, 2) for arch in ARCHS
             if not (arch in args.sweep_arch and seed == 0)]
    print("\n".join(paths))
    print(f"{len(paths)} configs; seed 0 of " + ", ".join(
        f"{a} is B_{a}_res_lam{lam_tag(lam[a])}_seed0" for a in args.sweep_arch) + " from block B")
