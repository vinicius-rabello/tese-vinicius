"""Block C: the best lambda from block B on the 5 architectures, residual,
seeds 0-2. Seed 0 of the sweep architecture is skipped: block B already
trained it with that lambda (14 runs).

    python scripts/make_block_c.py --lambda <value> --sweep-arch <name>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, lam_tag, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda", dest="lam", required=True, type=float)
    parser.add_argument("--sweep-arch", required=True, choices=ARCHS)
    args = parser.parse_args()
    if args.lam <= 0:
        parser.error("block C uses the chosen lambda > 0")
    paths = [write_config("C", arch, args.lam, True, seed)
             for seed in (0, 1, 2) for arch in ARCHS
             if not (arch == args.sweep_arch and seed == 0)]
    print("\n".join(paths))
    print(f"{len(paths)} configs; {args.sweep_arch} seed 0 is "
          f"B_{args.sweep_arch}_res_lam{lam_tag(args.lam)}_seed0 from block B")
