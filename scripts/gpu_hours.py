"""GPU-hours per block, from measured seconds per epoch (training + validation
with LPIPS logging, as train.py runs it) and three epoch-count scenarios.

    python scripts/gpu_hours.py --pilot <pilot results dir> [--extra arch:lambda:sec ...]

Seconds per epoch come from the pilot's train_log.csv (the last logged epoch,
i.e. after warm-up), keyed by (architecture, lambda > 0); --extra adds
measurements taken elsewhere. Epochs per run: 50 (the minimum), 150 (central;
the new rule replayed on the old LPIPS logs gives 139-202 for the CNNs) and
300 (the maximum). Multistage (block E) stage n also runs the n-1 frozen
earlier stages forward: cost factor 1 + 0.38 (n - 1) per epoch, from the
measured PRUSR stage-2 epoch (368 s) against its stage 1 (266 s); the
continuous-training control runs as many epochs as the 4 stages together.
"""

import argparse
import csv
import glob
import json
import os

SCENARIOS = (50, 150, 300)
ARCHS = ("srcnn", "dscms", "prusr", "srno", "sofono")
STAGE_FACTOR = 0.38


def measured(pilot_dir, extras):
    sec = {}
    for log in glob.glob(os.path.join(pilot_dir, '*', 'train_log.csv')):
        with open(os.path.join(os.path.dirname(log), 'config.json')) as f:
            meta = json.load(f)
        with open(log) as f:
            rows = list(csv.DictReader(f))
        sec[(meta['model'], meta['lambda'] > 0)] = float(rows[-1]['epoch_sec'])
    for item in extras:
        arch, lam, value = item.split(':')
        sec[(arch, float(lam) > 0)] = float(value)
    return sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--extra', nargs='*', default=[])
    args = parser.parse_args()
    sec = measured(args.pilot, args.extra)

    print("Seconds per epoch (train + validation with LPIPS logging):")
    for arch in ARCHS:
        print(f"  {arch:7s} lambda=0: {sec.get((arch, False), float('nan')):7.1f}   "
              f"lambda>0: {sec.get((arch, True), float('nan')):7.1f}")

    # (description, list of (arch, lambda>0, epoch-count multiplier))
    blocks = {
        'A  (5 archs x 3 seeds, MSE)': [(a, False, 1) for a in ARCHS for _ in range(3)],
        'B  (lambda sweep: 4 x SRNO + 4 x SRCNN)': [(a, True, 1) for a in ("srno", "srcnn") for _ in range(4)],
        'C  (best lambda, 5 archs x 3 seeds, minus SRNO/SRCNN seed 0 from B)':
            [(a, True, 1) for a in ARCHS for s in range(3) if not (a in ("srno", "srcnn") and s == 0)],
        'D  (residual=false, 5 archs, MSE, seed 0)': [(a, False, 1) for a in ARCHS],
        'N  (regime B, sigma=1e-2: 5 MSE + 5 best lambda)': [(a, lam, 1) for lam in (False, True) for a in ARCHS],
    }
    stage_sum = sum(1 + STAGE_FACTOR * (n - 1) for n in range(1, 5))  # 4 stages
    rows = []
    for name, runs in blocks.items():
        hours = [sum(sec[(a, lam)] * m * e for a, lam, m in runs) / 3600 for e in SCENARIOS]
        rows.append((name, len(runs), hours))
    for arch in ARCHS:  # block E depends on the best architecture, still unknown
        per_epoch = sec[(arch, False)]
        hours = [per_epoch * e * (stage_sum + 4) / 3600 for e in SCENARIOS]
        rows.append((f'E  if best arch = {arch} (4 stages + control)', 5, hours))

    print(f"\n{'block':70s} {'runs':>4s} " + " ".join(f"{e:>4d} ep" for e in SCENARIOS) + "   (GPU-hours)")
    for name, n, hours in rows:
        print(f"{name:70s} {n:4d} " + " ".join(f"{h:7.1f}" for h in hours))
    fixed = [sum(h[i] for name, _, h in rows if not name.startswith('E')) for i in range(len(SCENARIOS))]
    e_range = [(min(h[i] for name, _, h in rows if name.startswith('E')),
                max(h[i] for name, _, h in rows if name.startswith('E'))) for i in range(len(SCENARIOS))]
    print(f"\n{'Total A+B+C+D+N':70s} {'':4s} " + " ".join(f"{h:7.1f}" for h in fixed))
    print(f"{'Total with E (cheapest .. most expensive best arch)':70s} {'':4s} "
          + " ".join(f"{fixed[i] + e_range[i][0]:.0f}-{fixed[i] + e_range[i][1]:.0f}" for i in range(len(SCENARIOS))))


if __name__ == "__main__":
    main()
