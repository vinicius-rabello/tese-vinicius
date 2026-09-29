"""Block B: lambda sweep on one architecture, residual, seed 0 (4 runs).
Both the architecture and the 4 lambda values are chosen after the one-epoch
magnitude check (tables/lpips_magnitude.csv), so there are no defaults.

    python scripts/make_block_b.py --arch <name> --lambdas <l1> <l2> <l3> <l4>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=ARCHS)
    parser.add_argument("--lambdas", required=True, nargs=4, type=float)
    args = parser.parse_args()
    if any(lam <= 0 for lam in args.lambdas):
        parser.error("lambda = 0 is block A; the sweep values must be > 0")
    paths = [write_config("B", args.arch, lam, True, 0) for lam in args.lambdas]
    print("\n".join(paths))
