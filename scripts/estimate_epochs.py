"""Estimates how many epochs runs will take under the new stopping rule, by
replaying it on the validation-MSE sequences of the old stage-1 logs.

    python scripts/estimate_epochs.py

Rule: stop when the validation MSE has not improved by more than 0.1%
(relative) for 15 epochs, never before epoch 50, at most 300. The old runs
stopped by a different rule, so when an old log ends before the new rule
fires, the estimate is only a lower bound (marked 'censored'). Old runs used
0.6 MSE + 0.4 LPIPS for their LPIPS strategy, not MSE + lambda*LPIPS; they
only serve as a rough guide for lambda > 0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks'))
from multistage_report_lib import MODELS, find_stage1_log, parse_stage1_log  # noqa: E402

REL, PATIENCE, MIN_EPOCHS, MAX_EPOCHS = 1e-3, 15, 50, 300


def replay(val):
    best, counter = float('inf'), 0
    for epoch, v in enumerate(val, start=1):
        if v < best * (1 - REL):
            best, counter = v, 0
        else:
            counter += 1
        if counter >= PATIENCE and epoch >= MIN_EPOCHS:
            return epoch, False
        if epoch >= MAX_EPOCHS:
            return epoch, False
    return len(val), True  # the old run ended before the new rule fired


def main():
    print(f"{'arch':8s} {'strategy':10s} {'old epochs':>10s} {'new rule':>10s}")
    for model in MODELS:
        for strategy in ('mse', 'mse_lpips'):
            log = find_stage1_log(model, strategy)
            if log is None:
                continue
            _, val = parse_stage1_log(log)
            epochs, censored = replay(val)
            shown = f">= {max(epochs, MIN_EPOCHS)}" if censored else str(epochs)
            print(f"{model:8s} {strategy:10s} {len(val):>10d} {shown:>10s}{'  (censored)' if censored else ''}")


if __name__ == "__main__":
    main()
