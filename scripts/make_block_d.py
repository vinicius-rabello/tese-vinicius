"""Block D: ablation of the global skip -- residual = false, MSE (lambda = 0),
seed 0, on the 5 architectures (5 runs). Each run is compared with seed 0 of
block A (same architecture, residual = true).

    python scripts/make_block_d.py [--arch <name> ...]

--arch restricts the ablation to some architectures; the default is all 5.
Campaign (user decision, 2026-10-01): configs/D holds SRCNN, DSC/MS and SRNO
(`--arch srcnn dscms srno`); PRUSR and SoFoNO were moved to configs/D_optional/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, write_config  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", nargs="+", choices=ARCHS, default=list(ARCHS))
    args = parser.parse_args()
    paths = [write_config("D", arch, 0.0, False, 0) for arch in ARCHS if arch in args.arch]
    print("\n".join(paths))
    print(f"{len(paths)} configs; compare with A_<arch>_res_lam0_seed0")
