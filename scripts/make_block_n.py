"""Block N (noisy regime B): the 5 architectures with MSE only and the 5 with
the chosen lambda, residual, seed 0, observation noise sigma on the LR input
(10 runs). Sigma and lambda are chosen by the user, so both are required.

    python scripts/make_block_n.py --noise <sigma> --lambda <value>
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", required=True, type=float, help="NOISE_STD in m/s (> 0)")
    parser.add_argument("--lambda", dest="lam", required=True, type=float, help="chosen lambda (> 0)")
    args = parser.parse_args()
    if args.noise <= 0:
        parser.error("the noisy regime needs --noise > 0 (sigma = 0 is blocks A-C)")
    if args.lam <= 0:
        parser.error("--lambda is the chosen lambda > 0; the MSE runs (lambda = 0) are generated automatically")
    paths = [write_config("N", arch, lam, True, 0, noise=args.noise) for lam in (0.0, args.lam) for arch in ARCHS]
    print("\n".join(paths))
    print(f"{len(paths)} configs")
