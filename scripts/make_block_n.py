"""Block N (noisy regime B): the 5 architectures with MSE only and the 5 with
the chosen lambda, residual, seed 0, observation noise sigma on the LR input
(10 runs). Sigma and lambda are chosen by the user, so both are required.

    python scripts/make_block_n.py --noise <sigma> --lambda <value> [--sweep-arch srcnn [--sweep-lambdas ...]]

--sweep-arch adds the lambda sweep with noise on that architecture (lambda
grid of block B by default). The campaign (user decision, 2026-10-01) uses
`--noise 0.01 --lambda <best> --sweep-arch srcnn`: 10 + 4 runs, of which the
SRCNN run with the best lambda is the same file in both lists, so 13 configs.
Only after the advisor's approval.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, write_config  # noqa: E402

LAMBDA_GRID = (1e-3, 1e-2, 1e-1, 1.0)  # block B grid (user decision)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", required=True, type=float, help="NOISE_STD in m/s (> 0)")
    parser.add_argument("--lambda", dest="lam", required=True, type=float, help="chosen lambda (> 0)")
    parser.add_argument("--sweep-arch", choices=ARCHS, help="also run the lambda sweep with noise on this architecture")
    parser.add_argument("--sweep-lambdas", nargs="+", type=float, default=list(LAMBDA_GRID))
    args = parser.parse_args()
    if args.noise <= 0:
        parser.error("the noisy regime needs --noise > 0 (sigma = 0 is blocks A-C)")
    if args.lam <= 0:
        parser.error("--lambda is the chosen lambda > 0; the MSE runs (lambda = 0) are generated automatically")
    paths = [write_config("N", arch, lam, True, 0, noise=args.noise) for lam in (0.0, args.lam) for arch in ARCHS]
    if args.sweep_arch:
        for lam in args.sweep_lambdas:
            path = write_config("N", args.sweep_arch, lam, True, 0, noise=args.noise)
            if path not in paths:  # the best lambda is already in the list above
                paths.append(path)
    print("\n".join(paths))
    print(f"{len(paths)} configs")
