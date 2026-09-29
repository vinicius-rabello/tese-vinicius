"""Block A: the 5 architectures, MSE only (lambda = 0), residual, seeds 0-2 (15 runs).

    python scripts/make_block_a.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocks import ARCHS, write_config  # noqa: E402

if __name__ == "__main__":
    paths = [write_config("A", arch, 0.0, True, seed) for seed in (0, 1, 2) for arch in ARCHS]
    print("\n".join(paths))
    print(f"{len(paths)} configs")
