"""Final tables with uncertainty (stage 4.6). Run from the repo root:

    python stats.py [--results-dir results] [--out tables] [--n-resamples 2000]

Reads every evaluated run (`metrics_per_sample.csv`, `spectra.npz`,
`config.json`) and the baselines in results/, groups runs into configurations
(architecture, lambda, residual, noise, stage) whose seeds are replicates, and
writes, as CSV, Markdown and LaTeX:

    main_results   per configuration and metric: mean +- std across seeds (of
                   the test-set value of each seed) and the 95% block-bootstrap
                   interval of the configuration (7-day blocks)
    comparisons    paired differences B - A with 95% intervals for 7-day blocks
                   and, as a robustness check, 30-day blocks:
                     every configuration against Wiener at the same noise,
                     lambda > 0 against lambda = 0 (same arch, residual, noise),
                     residual = false against residual = true (same arch, lambda, noise)
    bands, lambda_sweep, ablation_skip   subsets of the two tables above
    multistage     (4.7) every configuration of a multistage chain (stage 1,
                   the later stages, the control), and multistage_comparisons:
                   stage n against stage n-1, the control against stages 1 and 2

Per-sample metrics are averaged over seeds day by day before the bootstrap;
pooled metrics (k_c, eps and T over k >= D0) are recomputed from the per-day
shell sums of spectra.npz on every resample.
"""

import argparse
import csv
import json
import os

import numpy as np

from evaluation.bootstrap import N_RESAMPLES, mean_ci, pooled_ci
from evaluation.spectral_metrics import BAND_NAMES, D0

PER_SAMPLE = ['mse', 'psnr', 'ssim', 'ke', 'wke', *[f'band_{b}' for b in BAND_NAMES], 'band_k_ge_D0',
              'k_c_fase', 'lpips_vgg', 'lpips_alex']
POOLED = ['k_c', 'eps_k_ge_D0', 'T_k_ge_D0']
SLOPE = 'inclinacao_espectral'
LOWER_IS_BETTER = {'mse', 'ke', 'wke', 'band_k_ge_D0', 'eps_k_ge_D0', 'lpips_vgg', 'lpips_alex',
                   *[f'band_{b}' for b in BAND_NAMES]}


# ------------------------------------------------------------------- loading --
def load_run(run_dir):
    with open(os.path.join(run_dir, 'config.json')) as f:
        meta = json.load(f)
    with open(os.path.join(run_dir, 'metrics_per_sample.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    spectra = np.load(os.path.join(run_dir, 'spectra.npz'))
    run = {
        'dates': [r['date'] for r in rows],
        'per_sample': {m: np.array([float(r[m]) for r in rows]) for m in PER_SAMPLE if m in rows[0]},
        'powers': {n: spectra[f'{n}_power'] for n in ('pred', 'target', 'error') if f'{n}_power' in spectra},
        'spectrum': spectra['pred'],
        'meta': meta,
    }
    if 'baseline' in meta:
        key = (meta['baseline'], None, None, float(meta.get('noise_std', 0.0)), '')
        seed = 0
    else:
        stage = (meta.get('config') or {}).get('STAGE', '')
        key = (meta['model'], float(meta['lambda']), bool(meta['residual']), float(meta.get('noise_std', 0.0)), stage)
        seed = meta.get('seed', 0)
    return key, seed, run


def load_all(results_dir):
    configs = {}
    for name in sorted(os.listdir(results_dir)):
        run_dir = os.path.join(results_dir, name)
        if not all(os.path.exists(os.path.join(run_dir, f)) for f in ('config.json', 'metrics_per_sample.csv', 'spectra.npz')):
            continue
        key, seed, run = load_run(run_dir)
        run['run_id'] = name
        configs.setdefault(key, {})[seed] = run
    return configs


def multistage_pairs(configs):
    """(kind, key_a, runs_a, key_b, runs_b) for the multistage chains, from
    config.json: stage n (multistage.previous_run) against stage n-1, and the
    control (FIXED_EPOCHS_FROM = [stage 1, stage 2]) against stage 1 and
    stage 2. Each side holds only the run the chain points at, so a stage 1
    taken from a 3-seed block is compared through its own seed, not the mean."""
    index = {run['run_id']: (key, seed) for key, runs in configs.items() for seed, run in runs.items()}
    pairs = []
    for key, runs in configs.items():
        for seed, run in runs.items():
            previous = (run['meta'].get('multistage') or {}).get('previous_run')
            sources = [('multistage_step', previous)] if previous else []
            fixed = (run['meta'].get('config') or {}).get('FIXED_EPOCHS_FROM') or []
            sources += [(f'control_vs_stage{i}', s) for i, s in enumerate(fixed[:2], start=1)]
            for kind, source in sources:
                if os.path.basename(source) in index:
                    key_a, seed_a = index[os.path.basename(source)]
                    pairs.append((kind, key_a, {seed_a: configs[key_a][seed_a]}, key, {seed: run}))
    return pairs


def label(key):
    arch, lam, residual, noise, stage = key
    if lam is None:
        return f"{arch} (baseline)" + (f", noise {noise:g}" if noise else "")
    parts = [arch, f"lambda {lam:g}", 'res' if residual else 'nores']
    if noise:
        parts.append(f"noise {noise:g}")
    if stage != '':
        parts.append(f"stage {stage}")
    return ", ".join(parts)


# ------------------------------------------------------------------- metrics --
def pooled_stats(pred, target, error):
    """Test-set k_c, eps over k >= D0 and mean T over k >= D0 from shell sums."""
    p, t, e = pred.sum(0), target.sum(0), error.sum(0)
    eps = e / t
    above = np.flatnonzero(eps > 0.5)
    k_c = float(above[0]) if len(above) else float(len(eps))
    high = np.arange(1, len(t) + 1) >= D0
    return {'k_c': k_c, 'eps_k_ge_D0': float(e[high].sum() / t[high].sum()), 'T_k_ge_D0': float((p / t)[high].mean())}


def slope_per_sample(spectrum, k_range):
    k = np.arange(1, spectrum.shape[1] + 1)
    sel = (k >= k_range[0]) & (k <= k_range[1])
    x = np.log(k[sel])
    xc = x - x.mean()
    y = np.log(spectrum[:, sel])
    return ((y - y.mean(1, keepdims=True)) * xc).sum(1) / (xc ** 2).sum()


def day_series(runs, metric, slope_range):
    """Per-day values averaged over seeds (seeds share the same test days)."""
    if metric == SLOPE:
        return np.mean([slope_per_sample(r['spectrum'], slope_range) for r in runs.values()], axis=0)
    return np.mean([r['per_sample'][metric] for r in runs.values()], axis=0)


def day_powers(runs):
    return [np.mean([r['powers'][n] for r in runs.values()], axis=0) for n in ('pred', 'target', 'error')]


def available(runs, metric):
    if metric in POOLED:
        return all(r['powers'] for r in runs.values())
    if metric == SLOPE:
        return True
    return all(metric in r['per_sample'] for r in runs.values())


def config_row(key, runs, metric, n_resamples, slope_range):
    """Mean +- std across seeds and the bootstrap interval of the configuration."""
    dates = next(iter(runs.values()))['dates']
    if metric in POOLED:
        per_seed = [pooled_stats(*(r['powers'][n] for n in ('pred', 'target', 'error')))[metric] for r in runs.values()]
        value, lo, hi = pooled_ci(lambda p, t, e: pooled_stats(p, t, e)[metric], day_powers(runs), dates,
                                  7, n_resamples)
    else:
        per_seed = [float(np.mean(day_series({0: r}, metric, slope_range))) for r in runs.values()]
        value, lo, hi = mean_ci(day_series(runs, metric, slope_range), dates, 7, n_resamples)
    return {'config': label(key), 'metric': metric, 'n_seeds': len(runs), 'mean_seeds': float(np.mean(per_seed)),
            'std_seeds': float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else '',
            'value': value, 'ci_lo': lo, 'ci_hi': hi}


def compare(kind, key_a, runs_a, key_b, runs_b, metric, n_resamples, slope_range):
    """Paired difference B - A over the same test days, 7- and 30-day blocks."""
    dates = next(iter(runs_a.values()))['dates']
    if dates != next(iter(runs_b.values()))['dates']:
        raise ValueError(f"{label(key_a)} and {label(key_b)} were evaluated on different test days")
    out = {'comparison': kind, 'A': label(key_a), 'B': label(key_b), 'metric': metric,
           'better_if': 'lower' if metric in LOWER_IS_BETTER else 'higher'}
    for days in (7, 30):
        if metric in POOLED:
            stat = lambda pa, ta, ea, pb, tb, eb: (pooled_stats(pb, tb, eb)[metric]  # noqa: E731
                                                   - pooled_stats(pa, ta, ea)[metric])
            diff, lo, hi = pooled_ci(stat, day_powers(runs_a) + day_powers(runs_b), dates, days, n_resamples)
        else:
            diff, lo, hi = mean_ci(day_series(runs_b, metric, slope_range) - day_series(runs_a, metric, slope_range),
                                   dates, days, n_resamples)
        out.update({'diff': diff, f'ci{days}_lo': lo, f'ci{days}_hi': hi})
    out['significant_7d'] = not (out['ci7_lo'] <= 0 <= out['ci7_hi'])
    return out


def comparison_pairs(configs):
    """(kind, key_a, key_b): vs Wiener, lambda > 0 vs 0, residual ablation."""
    pairs = []
    models = [k for k in configs if k[1] is not None]
    for key in models:
        arch, lam, residual, noise, stage = key
        wiener = ('wiener', None, None, noise, '')
        if wiener in configs:
            pairs.append(('vs_wiener', wiener, key))
        if lam > 0 and (arch, 0.0, residual, noise, stage) in configs:
            pairs.append(('lambda_vs_mse', (arch, 0.0, residual, noise, stage), key))
        if residual is False and (arch, lam, True, noise, stage) in configs:
            pairs.append(('ablation_skip', (arch, lam, True, noise, stage), key))
    return pairs


# -------------------------------------------------------------------- output --
def fmt(v, digits=4):
    return '' if v == '' or v is None else (f"{v:.{digits}g}" if isinstance(v, float) else str(v))


def write_tables(rows, cols, base):
    with open(base + '.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    with open(base + '.md', 'w') as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(fmt(r[c]) for c in cols) + " |\n")
    tex = lambda s: str(s).replace('_', r'\_').replace('%', r'\%')  # noqa: E731
    with open(base + '.tex', 'w') as f:
        f.write("\\begin{tabular}{" + "l" * len(cols) + "}\n\\hline\n")
        f.write(" & ".join(tex(c) for c in cols) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(tex(fmt(r[c])) for c in cols) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out', default='tables')
    parser.add_argument('--n-resamples', type=int, default=N_RESAMPLES)
    args = parser.parse_args()

    from collect_results import SLOPE_RANGE
    configs = load_all(args.results_dir)
    if not configs:
        raise SystemExit(f"No evaluated runs in {args.results_dir}")
    metrics = [*PER_SAMPLE, *POOLED, SLOPE]

    main_rows = [config_row(key, runs, m, args.n_resamples, SLOPE_RANGE)
                 for key, runs in sorted(configs.items(), key=lambda kv: label(kv[0]))
                 for m in metrics if available(runs, m)]
    comp_rows = [compare(kind, a, configs[a], b, configs[b], m, args.n_resamples, SLOPE_RANGE)
                 for kind, a, b in comparison_pairs(configs)
                 for m in metrics if available(configs[a], m) and available(configs[b], m)]
    chains = multistage_pairs(configs)
    comp_rows += [compare(kind, a, runs_a, b, runs_b, m, args.n_resamples, SLOPE_RANGE)
                  for kind, a, runs_a, b, runs_b in chains
                  for m in metrics if available(runs_a, m) and available(runs_b, m)]
    # Per-stage rows: each chain member with only the run the chain uses.
    members = {}
    for _, a, runs_a, b, runs_b in chains:
        members[(a, *runs_a)] = (a, runs_a)
        members[(b, *runs_b)] = (b, runs_b)
    chain_rows = sorted((config_row(key, runs, m, args.n_resamples, SLOPE_RANGE)
                         for key, runs in members.values() for m in metrics if available(runs, m)),
                        key=lambda r: r['config'])

    os.makedirs(args.out, exist_ok=True)
    main_cols = ['config', 'metric', 'n_seeds', 'mean_seeds', 'std_seeds', 'value', 'ci_lo', 'ci_hi']
    comp_cols = ['comparison', 'A', 'B', 'metric', 'better_if', 'diff', 'ci7_lo', 'ci7_hi', 'ci30_lo', 'ci30_hi',
                 'significant_7d']
    write_tables(main_rows, main_cols, os.path.join(args.out, 'main_results'))
    write_tables(comp_rows, comp_cols, os.path.join(args.out, 'comparisons'))
    write_tables([r for r in main_rows if r['metric'].startswith('band_')], main_cols, os.path.join(args.out, 'bands'))
    write_tables([r for r in comp_rows if r['comparison'] == 'lambda_vs_mse'], comp_cols,
                 os.path.join(args.out, 'lambda_sweep'))
    write_tables([r for r in comp_rows if r['comparison'] == 'ablation_skip'], comp_cols,
                 os.path.join(args.out, 'ablation_skip'))
    write_tables(chain_rows, main_cols, os.path.join(args.out, 'multistage'))
    write_tables([r for r in comp_rows if r['comparison'] == 'multistage_step' or r['comparison'].startswith('control_vs')],
                 comp_cols, os.path.join(args.out, 'multistage_comparisons'))
    print(f"{len(configs)} configurations, {len(main_rows)} metric rows, {len(comp_rows)} comparison rows -> {args.out}/")


if __name__ == "__main__":
    main()
