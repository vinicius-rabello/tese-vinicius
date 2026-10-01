"""Checks for the block bootstrap and stats.py (stage 4.6). CPU, synthetic data:

    CUDA_VISIBLE_DEVICES="" python tests/test_stats.py
"""

import csv
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation.bootstrap import block_ids, mean_ci  # noqa: E402
from stats import pooled_stats  # noqa: E402

with open(os.path.join(ROOT, 'evaluation', 'test_dates.csv')) as f:
    DATES = [r['date'] for r in csv.DictReader(f)]
N, K = len(DATES), 63


def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
    return ok


def ar1(rng, n, phi=0.8, scale=1.0):
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0, scale)
    return x


def fake_run(folder, name, meta, mse, eps_level):
    """A run folder with the files stats.py reads; eps(k) = eps_level for k <= 20, 0.9 above."""
    run = os.path.join(folder, name)
    os.makedirs(run)
    with open(os.path.join(run, 'config.json'), 'w') as f:
        json.dump(meta, f)
    with open(os.path.join(run, 'metrics_per_sample.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'date', 'mse', 'psnr'])
        for i, d in enumerate(DATES):
            writer.writerow([i, d, mse[i], 10 * np.log10(5.3268 ** 2 / mse[i])])
    target = np.ones((N, K))
    error = np.where(np.arange(1, K + 1) <= 20, eps_level, 0.9) * target
    np.savez_compressed(os.path.join(run, 'spectra.npz'), pred=np.exp(-2 * np.log(np.arange(1, K + 1)))[None].repeat(N, 0),
                        pred_power=target * 0.8, target_power=target, error_power=error)


def main():
    results = []
    rng = np.random.default_rng(1)

    blocks = block_ids(DATES, 7)
    sizes = np.bincount(blocks)
    results.append(check("7-day blocks: 974 test days -> 140 blocks of at most 7 days",
                         len(sizes) == 140 and sizes.max() == 7, f"(sizes {sizes.min()}..{sizes.max()})"))

    _, lo, hi = mean_ci(np.full(N, 3.0), DATES, n_resamples=200)
    results.append(check("constant series: interval collapses to the value", lo == hi == 3.0))
    same = rng.normal(size=N)
    diff, lo, hi = mean_ci(same - same, DATES, n_resamples=200)
    results.append(check("identical models: paired difference 0 with interval [0, 0]", diff == lo == hi == 0.0))

    true = 0.5
    series = true + ar1(rng, N)
    m7, lo7, hi7 = mean_ci(series, DATES, 7)
    m30, lo30, hi30 = mean_ci(series, DATES, 30)
    results.append(check("autocorrelated series: 95% interval contains the true mean",
                         lo7 <= true <= hi7 and lo30 <= true <= hi30,
                         f"(mean {m7:.3f}, 7-day [{lo7:.3f}, {hi7:.3f}], 30-day [{lo30:.3f}, {hi30:.3f}])"))
    results.append(check("same seed -> same interval", mean_ci(series, DATES, 7) == (m7, lo7, hi7)))

    p, t = np.ones((N, K)) * 0.8, np.ones((N, K))
    e = np.where(np.arange(1, K + 1) <= 20, 0.1, 0.9) * t
    results.append(check("pooled k_c from shell sums: eps <= 1/2 up to k = 20 -> k_c = 20",
                         pooled_stats(p, t, e)['k_c'] == 20.0))

    with tempfile.TemporaryDirectory() as folder:
        res = os.path.join(folder, 'results')
        mses = {}
        for seed in range(3):
            mses[seed] = 1e-3 + 1e-4 * seed + 1e-5 * np.abs(rng.normal(size=N))
            fake_run(res, f"A_srcnn_res_lam0_seed{seed}",
                     {'model': 'srcnn', 'lambda': 0.0, 'residual': True, 'seed': seed}, mses[seed], 0.1)
        fake_run(res, "B_srcnn_res_lam0.01_seed0", {'model': 'srcnn', 'lambda': 0.01, 'residual': True, 'seed': 0},
                 mses[0] * 0.9, 0.1)
        fake_run(res, "baseline_wiener", {'baseline': 'wiener', 'noise_std': 0.0}, mses[0] * 0.5, 0.3)
        out = os.path.join(folder, 'tables')
        proc = subprocess.run([sys.executable, 'stats.py', '--results-dir', res, '--out', out, '--n-resamples', '200'],
                              cwd=ROOT, capture_output=True, text=True)
        ok = proc.returncode == 0
        files = sorted(os.listdir(out)) if ok else []
        expected = {f"{b}.{e}" for b in ('main_results', 'comparisons', 'bands', 'lambda_sweep', 'ablation_skip')
                    for e in ('csv', 'md', 'tex')}
        results.append(check("stats.py writes every table as CSV, Markdown and LaTeX", ok and expected <= set(files),
                             proc.stderr[-300:] if not ok else ""))
        if ok:
            rows = list(csv.DictReader(open(os.path.join(out, 'main_results.csv'))))
            row = next(r for r in rows if r['config'] == 'srcnn, lambda 0, res' and r['metric'] == 'mse')
            per_seed = [mses[s].mean() for s in range(3)]
            results.append(check("mean +- std across the 3 seeds matches a direct computation",
                                 int(row['n_seeds']) == 3 and abs(float(row['mean_seeds']) - np.mean(per_seed)) < 1e-12
                                 and abs(float(row['std_seeds']) - np.std(per_seed, ddof=1)) < 1e-12,
                                 f"({float(row['mean_seeds']):.6g} +- {float(row['std_seeds']):.3g})"))
            kc = next(r for r in rows if r['config'] == 'srcnn, lambda 0, res' and r['metric'] == 'k_c')
            results.append(check("pooled k_c in main_results", float(kc['value']) == 20.0))
            comps = list(csv.DictReader(open(os.path.join(out, 'comparisons.csv'))))
            kinds = {c['comparison'] for c in comps}
            lam = next(c for c in comps if c['comparison'] == 'lambda_vs_mse' and c['metric'] == 'mse')
            results.append(check("comparisons: vs Wiener and lambda vs MSE present, with 7- and 30-day intervals",
                                 {'vs_wiener', 'lambda_vs_mse'} <= kinds and lam['ci30_lo'] != ''))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
