"""GPU-hours per block and per queue, from measured seconds per epoch
(training + validation with LPIPS logging, as train.py runs it) and
epoch-count scenarios.

    python scripts/gpu_hours.py [--pilot results_pilot] [--extra arch:lambda:sec ...]
                                [--measured results] [--e-arch <name>]

Seconds per epoch come from the pilot's train_log.csv (the last logged epoch,
i.e. after warm-up), keyed by (architecture, lambda > 0); --extra adds
measurements taken elsewhere. Epochs per run: 50 (the minimum), 150 (central;
the new rule replayed on the old LPIPS logs gives 139-202 for the CNNs) and
300 (the maximum).

--measured <results dir> (e.g. after block A seed 0): finished runs there
replace the pilot's seconds per epoch (mean over epochs 2.., same arch and
lambda > 0 or = 0) and give a "measured" column whose epochs per run are the
mean epochs_completed of that architecture (same lambda group if measured,
otherwise the architecture's other group); architectures without a finished
run fall back to 150 epochs, marked with *.

Multistage (block E) stage n also runs the n-1 frozen earlier stages
forward: cost factor 1 + 0.38 (n - 1) per epoch, from the measured PRUSR
stage-2 epoch (368 s) against its stage 1 (266 s). Block E trains stages 2-4
(stage 1 is the block-A run, already counted in A) and the
continuous-training control, which runs the epochs of stage 1 + stage 2.
--e-arch fixes the architecture of E; without it every choice is listed.

Composition as decided by the user (2026-10-01): A 5 x 3 seeds; B lambda
sweep 4 x SRNO + 4 x SRCNN; C best lambda, 5 x 3 seeds minus SRNO/SRCNN seed
0 (from B); D skip ablation on SRCNN, DSC/MS, SRNO (PRUSR and SoFoNO in
configs/D_optional, not counted); regime B (sigma = 1e-2) 5 MSE + 5 best
lambda + the lambda sweep on SRCNN (3 more runs; the best lambda is shared),
13 runs. Queues as in queues/ (each launched with --max-hours 70).
"""

import argparse
import csv
import glob
import json
import math
import os

SCENARIOS = (50, 150, 300)
CENTRAL = 150
ARCHS = ("srcnn", "dscms", "prusr", "srno", "sofono")
STAGE_FACTOR = 0.38
QUEUE_HOURS = 70


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


def campaign_measurements(results_dir):
    """Finished single-stage runs: {(arch, lambda > 0): ([epochs], [s/epoch])}."""
    out = {}
    for summary_path in glob.glob(os.path.join(results_dir, '*', 'summary.json')):
        run = os.path.dirname(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        if summary.get('status') not in ('completed_early_stop', 'completed_max_epochs'):
            continue
        with open(os.path.join(run, 'config.json')) as f:
            meta = json.load(f)
        if 'model' not in meta or meta.get('multistage') or (meta.get('config') or {}).get('FIXED_EPOCHS_FROM'):
            continue
        with open(os.path.join(run, 'train_log.csv')) as f:
            secs = [float(r['epoch_sec']) for r in csv.DictReader(f)]
        secs = secs[1:] or secs  # epoch 1 includes the warm-up
        epochs, per_epoch = out.setdefault((meta['model'], meta['lambda'] > 0), ([], []))
        epochs.append(int(summary['epochs_completed']))
        per_epoch.append(sum(secs) / len(secs))
    return out


def blocks_and_queues():
    """(arch, lambda > 0, epoch multiplier) per block and per queue."""
    A = {s: [(a, False, 1) for a in ARCHS] for s in range(3)}
    B = [(a, True, 1) for a in ("srcnn", "srno") for _ in range(4)]
    C = {s: [(a, True, 1) for a in ARCHS if not (s == 0 and a in ("srno", "srcnn"))] for s in range(3)}
    D = [(a, False, 1) for a in ("srcnn", "dscms", "srno")]
    N = [(a, False, 1) for a in ARCHS] + [(a, True, 1) for a in ARCHS] + [("srcnn", True, 1)] * 3
    blocks = {
        'A  (5 archs x 3 seeds, MSE)': A[0] + A[1] + A[2],
        'B  (lambda sweep: 4 x SRNO + 4 x SRCNN)': B,
        'C  (best lambda, 5 archs x 3 seeds, minus SRNO/SRCNN seed 0 from B)': C[0] + C[1] + C[2],
        'D  (residual=false: SRCNN, DSC/MS, SRNO; MSE, seed 0)': D,
        'N  (regime B, sigma=1e-2: 5 MSE + 5 best lambda + 3 SRCNN sweep)': N,
    }
    queues = {
        'fase1: A seed 0, then B': A[0] + B,
        'fase2: C seed 0': C[0],
        'fase3: D (reduced)': D,
        'fase4: A seeds 1-2': A[1] + A[2],
        'fase5: C seeds 1-2': C[1] + C[2],
        'fase6: E': None,  # filled with the E runs
        'fase7: regime B (after the advisor)': N,
    }
    optional = {'D_optional (PRUSR, SoFoNO; not counted)': [(a, False, 1) for a in ("prusr", "sofono")]}
    return blocks, queues, optional


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', default='results_pilot')
    parser.add_argument('--extra', nargs='*', default=[])
    parser.add_argument('--measured', help="results dir with finished campaign runs (e.g. results)")
    parser.add_argument('--e-arch', choices=ARCHS, help="architecture of block E, once chosen")
    args = parser.parse_args()
    sec = measured(args.pilot, args.extra)

    epochs_measured, star = {}, set()
    if args.measured:
        camp = campaign_measurements(args.measured)
        for key, (_, secs) in camp.items():
            sec[key] = sum(secs) / len(secs)
        for arch in ARCHS:
            for lam in (False, True):
                same, other = camp.get((arch, lam)), camp.get((arch, not lam))
                source = same or other
                if source:
                    epochs_measured[(arch, lam)] = sum(source[0]) / len(source[0])
                else:
                    epochs_measured[(arch, lam)] = CENTRAL
                    star.add(arch)
        print(f"Measured runs in {args.measured}:")
        for (arch, lam), (eps, secs) in sorted(camp.items()):
            print(f"  {arch:7s} lambda{'>0' if lam else '=0'}: {len(eps)} run(s), epochs {eps}, "
                  f"{sum(secs) / len(secs):.1f} s/epoch")
        if star:
            print(f"  no finished run (150 epochs assumed, marked *): {', '.join(sorted(star))}")
        print()

    scenarios = list(SCENARIOS) + (['measured'] if args.measured else [])

    def epochs(arch, lam, scenario):
        return epochs_measured[(arch, lam)] if scenario == 'measured' else scenario

    def hours(runs):
        return [sum(sec[(a, lam)] * m * epochs(a, lam, s) for a, lam, m in runs) / 3600 for s in scenarios]

    def e_hours(arch):
        stage_sum = sum(1 + STAGE_FACTOR * (n - 1) for n in range(2, 5))  # stages 2-4
        return [sec[(arch, False)] * epochs(arch, False, s) * (stage_sum + 2) / 3600 for s in scenarios]

    print("Seconds per epoch (train + validation with LPIPS logging):")
    for arch in ARCHS:
        print(f"  {arch:7s} lambda=0: {sec.get((arch, False), float('nan')):7.1f}   "
              f"lambda>0: {sec.get((arch, True), float('nan')):7.1f}")
    print("\nGPU-hours per run (" + " / ".join(f"{s} ep" if s != 'measured' else 'measured' for s in scenarios) + "):")
    for arch in ARCHS:
        for lam in (False, True):
            if (arch, lam) in sec:
                print(f"  {arch:7s} lambda{'>0' if lam else '=0'}: "
                      + " / ".join(f"{h:6.2f}" for h in hours([(arch, lam, 1)]))
                      + (" *" if arch in star else ""))

    blocks, queues, optional = blocks_and_queues()
    head = " ".join(f"{s:>7}" if s == 'measured' else f"{s:>4d} ep" for s in scenarios)
    print(f"\n{'block':72s} {'runs':>4s} {head}   (GPU-hours)")
    fixed = [0.0] * len(scenarios)
    for name, runs in blocks.items():
        h = hours(runs)
        fixed = [f + x for f, x in zip(fixed, h)]
        print(f"{name:72s} {len(runs):4d} " + " ".join(f"{x:7.1f}" for x in h))
    e_archs = [args.e_arch] if args.e_arch else list(ARCHS)
    e_rows = {arch: e_hours(arch) for arch in e_archs}
    for arch, h in e_rows.items():
        print(f"{'E  (stages 2-4 + control), arch = ' + arch:72s} {4:4d} " + " ".join(f"{x:7.1f}" for x in h))
    print(f"\n{'Total A+B+C+D+N':72s} {'':4s} " + " ".join(f"{x:7.1f}" for x in fixed))
    lo = [fixed[i] + min(h[i] for h in e_rows.values()) for i in range(len(scenarios))]
    hi = [fixed[i] + max(h[i] for h in e_rows.values()) for i in range(len(scenarios))]
    label = f"Total with E ({args.e_arch})" if args.e_arch else "Total with E (cheapest .. most expensive arch)"
    print(f"{label:72s} {'':4s} " + " ".join(f"{a:7.1f}" if args.e_arch else f"{a:.0f}-{b:.0f}"
                                              for a, b in zip(lo, hi)))
    for name, runs in optional.items():
        print(f"{name:72s} {len(runs):4d} " + " ".join(f"{x:7.1f}" for x in hours(runs)))

    col = scenarios.index('measured') if args.measured else scenarios.index(CENTRAL)
    print(f"\nQueues (queues/*.txt, each launched with --max-hours {QUEUE_HOURS}), "
          f"{'measured' if args.measured else f'{CENTRAL} epochs'}:")
    for name, runs in queues.items():
        if runs is None:
            h = max(e_rows.values(), key=lambda v: v[col])[col] if not args.e_arch else e_rows[args.e_arch][col]
            n, note = 4, "" if args.e_arch else " (most expensive arch)"
        else:
            h, n, note = hours(runs)[col], len(runs), ""
        print(f"  {name:44s} {n:3d} runs {h:7.1f} h  -> {math.ceil(h / QUEUE_HOURS)} x {QUEUE_HOURS} h{note}")


if __name__ == "__main__":
    main()
