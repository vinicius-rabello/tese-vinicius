"""Block C: the best lambda from block B on the 5 architectures, residual,
seeds 0-2. Seed 0 of each sweep architecture is skipped: block B already
trained it with that lambda (14 runs with one sweep architecture, 13 with two).

    python scripts/make_block_c.py --lambda <value> --sweep-arch <name> [<name> ...]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, lam_tag, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", required=True, type=float)
    parser.add_argument("--sweep-arch", required=True, choices=ARCHS, nargs="+")
    args = parser.parse_args()
    if args.lam <= 0:
        parser.error("block C uses the chosen lambda > 0")
    paths = [write_config("C", arch, args.lam, True, seed)
             for seed in (0, 1, 2) for arch in ARCHS
             if not (arch in args.sweep_arch and seed == 0)]
    print("\n".join(paths))
    print(f"{len(paths)} configs; seed 0 of " + ", ".join(
        f"{a} is B_{a}_res_lam{lam_tag(args.lam)}_seed0" for a in args.sweep_arch) + " from block B")
