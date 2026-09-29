"""Collects the results of the runs in results/ into a small report folder and
a .zip to send for analysis (no checkpoints anywhere).

    python collect_results.py [--runs "A_*" ...] [--out report/<timestamp>/] [--no-eval]

Outputs in --out: all_runs.csv, baselines.csv, figures (PNG + PDF), report.md,
and runs/<run_id>/ with each run's config.json, summary.json, train_log.csv.
A finished run without test metrics is evaluated once (test set, no_grad) and
its test means are written into summary.json, so the next call doesn't repeat
it; --no-eval leaves those columns empty instead.
"""

import argparse
import csv
import fnmatch
import json
import math
import os
import shutil
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from evaluation.spectral_metrics import BAND_EDGES, BAND_NAMES, D0  # noqa: E402

STATUS_TO_REASON = {'completed_early_stop': 'early_stopping', 'completed_max_epochs': 'max_epochs',
                    'interrupted_max_hours': 'max_hours'}
COMPLETE = ('completed_early_stop', 'completed_max_epochs')
BAND_COLS = [f"e_banda_{i + 1}" for i in range(len(BAND_NAMES))]
METRIC_COLS = ['mse', 'psnr', 'ssim', 'ke', 'wke', *BAND_COLS, 'e_banda_k_ge_D0', 'k_c',
               'inclinacao_espectral', 'lpips_apendice', 'lpips_vgg']
RUN_COLS = ['run_id', 'status', 'arquitetura', 'perda', 'lambda', 'semente', 'residual', 'estagio', 'epocas',
            'motivo_parada', 'tempo_treino_h', 'n_parametros', 'commit', 'arvore_limpa', 'R_psnr',
            'escala_lpips', *METRIC_COLS, 'best_val_mse']
# Spectral-slope fitting range (k, cycles per domain): not fixed yet -- it
# waits for the user's confirmation, so the column stays empty until then.
SLOPE_RANGE = None
FIG_META = {'png': {'Software': None}, 'pdf': {'CreationDate': None, 'ModDate': None, 'Producer': None}}


# ------------------------------------------------------------------ reading --
def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def test_means(run_dir):
    """Test-set means from metrics_per_sample.csv (and the slope from
    spectra.npz once SLOPE_RANGE is set). Empty dict if not evaluated yet."""
    rows = read_csv(os.path.join(run_dir, 'metrics_per_sample.csv'))
    if not rows:
        return {}
    col = lambda name: np.array([to_float(r[name]) for r in rows]) if name in rows[0] else None  # noqa: E731
    out = {m: float(np.mean(col(m))) for m in ('mse', 'psnr', 'ssim', 'ke', 'wke', 'k_c') if col(m) is not None}
    for i, name in enumerate(BAND_NAMES):
        if col(f'band_{name}') is not None:
            out[BAND_COLS[i]] = float(np.mean(col(f'band_{name}')))
    if col('band_k_ge_D0') is not None:
        out['e_banda_k_ge_D0'] = float(np.mean(col('band_k_ge_D0')))
    if col('lpips_alex') is not None:
        out['lpips_apendice'] = float(np.mean(col('lpips_alex')))
    if col('lpips_vgg') is not None:
        out['lpips_vgg'] = float(np.mean(col('lpips_vgg')))
    if SLOPE_RANGE is not None and os.path.exists(os.path.join(run_dir, 'spectra.npz')):
        import torch
        from evaluation.spectral_metrics import spectral_slope
        spec = torch.from_numpy(np.load(os.path.join(run_dir, 'spectra.npz'))['pred'])
        out['inclinacao_espectral'] = float(spectral_slope(spec, *SLOPE_RANGE).mean())
    return out


def evaluate_missing(run_dir, device, with_lpips):
    """Evaluates a finished run that has no test metrics yet and records the
    test means in summary.json, so this happens once per run."""
    import evaluate_run
    model, meta = evaluate_run.load_model(run_dir, device)
    constants = {k: meta[k] for k in evaluate_run.CONSTANT_KEYS}
    evaluate_run.evaluate(model, constants, run_dir, evaluate_run.test_loader(16), device, with_lpips)
    summary_path = os.path.join(run_dir, 'summary.json')
    summary = read_json(summary_path)
    summary['test_metrics'] = test_means(run_dir)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def run_row(run_id, run_dir):
    config = read_json(os.path.join(run_dir, 'config.json')) or {}
    summary = read_json(os.path.join(run_dir, 'summary.json')) or {}
    status = summary.get('status')
    lam = config.get('lambda')
    git = config.get('git') or {}
    row = {
        'run_id': run_id,
        'status': 'completa' if status in COMPLETE else 'incompleta',
        'arquitetura': config.get('model', ''),
        'perda': '' if lam is None else ('mse' if lam == 0 else 'mse+lpips'),
        'lambda': '' if lam is None else lam,
        'semente': config.get('seed', ''),
        'residual': config.get('residual', ''),
        'estagio': (config.get('config') or {}).get('STAGE', ''),
        'epocas': summary.get('epochs_completed', ''),
        'motivo_parada': STATUS_TO_REASON.get(status, ''),
        'tempo_treino_h': summary.get('train_hours', ''),
        'n_parametros': config.get('n_params', ''),
        'commit': git.get('commit', ''),
        'arvore_limpa': '' if git.get('dirty') is None else (not git['dirty']),
        'R_psnr': config.get('psnr_ssim_R', ''),
        'escala_lpips': config.get('lpips_scale', ''),
        'best_val_mse': summary.get('best_val_mse', ''),
    }
    for m in METRIC_COLS:
        row[m] = ''
    row.update(test_means(run_dir))
    return row


# ------------------------------------------------------------------ figures --
def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=200, bbox_inches='tight', metadata=FIG_META['png'])
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'), bbox_inches='tight', metadata=FIG_META['pdf'])
    plt.close(fig)


def group_key(row):
    """Comparable runs: same loss, lambda, residual flag and stage (all
    architectures and seeds together)."""
    return f"{row['perda'] or 'na'}_lam{row['lambda']}_{'res' if row['residual'] else 'nores'}" + \
        (f"_stage{row['estagio']}" if row['estagio'] != '' else '')


def mean_spectra(run_dir):
    path = os.path.join(run_dir, 'spectra.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {'k': d['k'], 'pred': d['pred'].mean(0), 'target': d['target'].mean(0), 'rho': d['rho'].mean(0)}


def by_arch(rows, results_dir):
    """Mean over seeds per architecture, for rows that have spectra."""
    out = {}
    for row in rows:
        spec = mean_spectra(os.path.join(results_dir, row['run_id']))
        if spec is not None:
            out.setdefault(row['arquitetura'], []).append((row, spec))
    return out


def figure_spectra_rho_bands(groups, results_dir, baselines_dir, out_dir):
    identity = mean_spectra(os.path.join(baselines_dir, 'baseline_identity'))
    for key, rows in sorted(groups.items()):
        archs = by_arch(rows, results_dir)
        if not archs:
            continue
        k = next(iter(archs.values()))[0][1]['k']
        target = next(iter(archs.values()))[0][1]['target']

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.loglog(k, target, color='black', lw=2.2, label='Target (HR)')
        if identity is not None:
            ax.loglog(k, identity['pred'], color='gray', lw=1.5, ls='--', label='Input (LR)')
        for arch, items in sorted(archs.items()):
            ax.loglog(k, np.mean([s['pred'] for _, s in items], 0), lw=1.3, label=f"{arch} (n={len(items)})")
        ax.axvline(D0, color='gray', ls=':', lw=1)
        ax.set_xlabel('Wavenumber k (cycles per 128-px domain; 1 px = 0.04°)')
        ax.set_ylabel('Kinetic energy E(k) (m²/s²)')
        ax.set_title(f'Radial KE spectrum, test set mean -- {key}')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, out_dir, f'spectra_{key}')

        fig, ax = plt.subplots(figsize=(7, 5))
        if identity is not None:
            ax.plot(k, identity['rho'], color='gray', ls='--', lw=1.5, label='Input (LR)')
        for arch, items in sorted(archs.items()):
            ax.plot(k, np.mean([s['rho'] for _, s in items], 0), lw=1.3, label=f"{arch} (n={len(items)})")
        ax.axhline(0.5, color='red', ls=':', lw=1, label='ρ = 1/2')
        ax.axvline(D0, color='gray', ls=':', lw=1)
        ax.set_xlabel('Wavenumber k (cycles per 128-px domain)')
        ax.set_ylabel('Spectral correlation ρ(k) (dimensionless)')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'Spectral correlation, test set mean -- {key}')
        ax.grid(True, ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, out_dir, f'rho_k_{key}')

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"[{lo:g}, {hi:g})" if hi != math.inf else f"≥ {lo:g}"
                  for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
        arch_rows = {}
        for row in rows:
            if row[BAND_COLS[0]] != '':
                arch_rows.setdefault(row['arquitetura'], []).append(row)
        width = 0.8 / max(1, len(arch_rows))
        for i, (arch, rs) in enumerate(sorted(arch_rows.items())):
            vals = [np.mean([r[c] for r in rs]) for c in BAND_COLS]
            ax.bar(np.arange(len(BAND_COLS)) + i * width, vals, width, label=f"{arch} (n={len(rs)})")
        ax.set_xticks(np.arange(len(BAND_COLS)) + width * (len(arch_rows) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.set_xlabel(f'Radial band of k (cycles per domain); D0 = {D0:g}')
        ax.set_ylabel('Error in band (m²/s²)')
        ax.set_title(f'Test error by radial band -- {key}')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, out_dir, f'error_by_band_{key}')


def figure_train_curves(rows, results_dir, out_dir):
    logged = [(r, read_csv(os.path.join(results_dir, r['run_id'], 'train_log.csv'))) for r in rows]
    logged = [(r, log) for r, log in logged if log]
    if not logged:
        return
    cols = min(3, len(logged))
    nrows = math.ceil(len(logged) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5.5 * cols, 3.8 * nrows), squeeze=False)
    for ax, (row, log) in zip(axes.flat, logged):
        epochs = [int(r['epoch']) for r in log]
        ax.semilogy(epochs, [to_float(r['train_total']) for r in log], label='train loss')
        ax.semilogy(epochs, [to_float(r['val_mse']) for r in log], label='val MSE')
        if row['motivo_parada']:
            ax.axvline(epochs[-1], color='red', ls=':', lw=1, label=f"stop ({row['motivo_parada']})")
        ax.set_title(row['run_id'], fontsize=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / MSE (m²/s²)')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[len(logged):]:
        ax.axis('off')
    fig.tight_layout()
    save(fig, out_dir, 'train_curves')


def figure_lambda_sweep(rows, out_dir):
    """Only when one architecture has test metrics for more than one lambda."""
    per_arch = {}
    for r in rows:
        if r['mse'] != '' and r['lambda'] != '':
            per_arch.setdefault(r['arquitetura'], []).append(r)
    per_arch = {a: rs for a, rs in per_arch.items() if len({r['lambda'] for r in rs}) > 1}
    if not per_arch:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for arch, rs in sorted(per_arch.items()):
        lams = sorted({r['lambda'] for r in rs})
        mean = lambda col, lam: np.mean([r[col] for r in rs if r['lambda'] == lam])  # noqa: E731
        for ax, col, label in zip(axes, ('mse', 'ke', 'k_c'),
                                  ('Test MSE (m²/s²)', 'Test KE (log-spectral, dimensionless)',
                                   'Effective resolution k_c (cycles per domain)')):
            ax.plot(lams, [mean(col, lam) for lam in lams], marker='o', label=arch)
            ax.set_xscale('symlog', linthresh=min(l for l in lams if l > 0))
            ax.set_xlabel('λ (lambda = 0 is plain MSE)')
            ax.set_ylabel(label)
            ax.grid(True, which='both', ls='--', alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Lambda sweep: MSE, KE and ρ-based resolution against λ')
    fig.tight_layout()
    save(fig, out_dir, 'lambda_sweep')
    return True


# ------------------------------------------------------------------- report --
def fmt(v, digits=4):
    if v == '' or v is None:
        return ''
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def alerts(rows, results_dir):
    out = []
    rs = {r['R_psnr'] for r in rows if r['R_psnr'] != ''}
    for r in rows:
        rid = r['run_id']
        if r['motivo_parada'] == 'max_epochs':
            out.append(f"{rid}: stopped at max_epochs (may not have converged).")
        config = read_json(os.path.join(results_dir, rid, 'config.json')) or {}
        min_epochs = (config.get('stopping') or {}).get('min_epochs')
        if r['motivo_parada'] == 'early_stopping' and min_epochs is not None and r['epocas'] == min_epochs:
            out.append(f"{rid}: stopped exactly at min_epochs = {min_epochs} (early plateau).")
        if r['arvore_limpa'] is False:
            out.append(f"{rid}: trained on a dirty tree (uncommitted changes).")
        if len(rs) > 1 and r['R_psnr'] != '':
            out.append(f"{rid}: R_psnr = {r['R_psnr']} differs across runs ({sorted(rs)}).")
        samples = read_csv(os.path.join(results_dir, rid, 'metrics_per_sample.csv'))
        if samples and r['R_psnr'] != '':
            worst = max(abs(to_float(s['psnr']) - 10 * math.log10(r['R_psnr'] ** 2 / to_float(s['mse'])))
                        for s in samples)
            if not worst < 1e-3:
                out.append(f"{rid}: PSNR inconsistent with 10 log10(R²/MSE) (max deviation {worst:.3g} dB).")
        log = read_csv(os.path.join(results_dir, rid, 'train_log.csv'))
        if log and r['lambda'] not in ('', 0, 0.0):
            last = log[-1]
            ratio = r['lambda'] * to_float(last['train_lpips']) / to_float(last['train_mse'])
            if ratio > 10:
                out.append(f"{rid}: LPIPS dominates the loss at the end (λ·LPIPS/MSE = {ratio:.3g}).")
        values = [r[c] for c in METRIC_COLS + ['best_val_mse'] if isinstance(r[c], float)]
        values += [to_float(v) for e in log for k, v in e.items() if k not in ('train_lpips',) and k != 'es_metric']
        if any(not math.isfinite(v) for v in values):
            out.append(f"{rid}: NaN or infinite values in its metrics or training log.")
    return out


def write_report(rows, baseline_rows, alert_lines, figures, out_dir):
    lines = ["# Results report", ""]
    lines += ["## Alerts", ""] + ([f"- {a}" for a in alert_lines] or ["- none"]) + [""]
    done = sorted([r for r in rows if r['mse'] != ''], key=lambda r: r['mse'])
    pending = [r for r in rows if r['mse'] == '']
    lines += ["## Runs (test set, m/s, fixed R), sorted by MSE", ""]
    head = ['run_id', 'epocas', 'motivo_parada', 'mse', 'psnr', 'ssim', 'ke', 'wke', 'e_banda_k_ge_D0', 'k_c',
            'best_val_mse']
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in head) + " |" for r in done]
    if baseline_rows:
        lines += ["", "Baselines:", "", "| baseline | mse | psnr | ssim | ke | wke | e_banda_k_ge_D0 | k_c |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {b['baseline']} | " + " | ".join(fmt(b[c]) for c in
                                                         ('mse', 'psnr', 'ssim', 'ke', 'wke', 'e_banda_k_ge_D0', 'k_c'))
                  + " |" for b in baseline_rows]
    if pending:
        lines += ["", "Not evaluated yet / incomplete: " + ", ".join(f"{r['run_id']} ({r['status']})" for r in pending)]

    lines += ["", "## Difference to plain MSE (same architecture and residual flag, mean over seeds)", ""]
    diffs = []
    for arch in sorted({r['arquitetura'] for r in done}):
        for residual in (True, False):
            same = [r for r in done if r['arquitetura'] == arch and r['residual'] is residual]
            base = [r for r in same if r['lambda'] == 0]
            if not base:
                continue
            for lam in sorted({r['lambda'] for r in same if r['lambda'] != 0}):
                other = [r for r in same if r['lambda'] == lam]
                d = {c: np.mean([r[c] for r in other]) - np.mean([r[c] for r in base])
                     for c in ('mse', 'psnr', 'ssim', 'ke', 'wke', 'k_c')}
                diffs.append(f"| {arch} | {residual} | {lam:g} | " + " | ".join(f"{d[c]:+.4g}" for c in d) + " |")
    if diffs:
        lines += ["| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |",
                  "|---|---|---|---|---|---|---|---|---|"] + diffs
    else:
        lines += ["(no architecture has both an MSE and an MSE+LPIPS run yet)"]
    lines += ["", "## Figures", ""] + [f"- {f}" for f in figures]
    with open(os.path.join(out_dir, 'report.md'), 'w') as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path, rows, cols):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, '') for c in cols})


def folder_size(path):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, files in os.walk(path) for f in files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--runs', nargs='*', default=['*'], help='run_id patterns, e.g. "A_*"')
    parser.add_argument('--out', default=None, help='default: report/<date-time>/')
    parser.add_argument('--no-eval', action='store_true', help="don't evaluate finished runs lacking test metrics")
    parser.add_argument('--no-lpips', action='store_true', help='skip LPIPS when evaluating missing runs')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    out_dir = args.out or os.path.join('report', stamp)
    os.makedirs(out_dir, exist_ok=True)
    patterns = [p for item in args.runs for p in item.replace(',', ' ').split()]

    run_ids = sorted(d for d in os.listdir(args.results_dir)
                     if os.path.isdir(os.path.join(args.results_dir, d)) and not d.startswith('baseline_')
                     and any(fnmatch.fnmatch(d, p) for p in patterns))
    rows = []
    for run_id in run_ids:
        run_dir = os.path.join(args.results_dir, run_id)
        summary = read_json(os.path.join(run_dir, 'summary.json')) or {}
        evaluated = os.path.exists(os.path.join(run_dir, 'metrics_per_sample.csv'))
        if summary.get('status') in COMPLETE and not evaluated and not args.no_eval:
            import torch
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Evaluating {run_id} on the test set ({device}) ...")
            evaluate_missing(run_dir, device, not args.no_lpips)
        rows.append(run_row(run_id, run_dir))
        target = os.path.join(out_dir, 'runs', run_id)
        os.makedirs(target, exist_ok=True)
        for name in ('config.json', 'summary.json', 'train_log.csv', 'eval_summary.json'):
            if os.path.exists(os.path.join(run_dir, name)):
                shutil.copyfile(os.path.join(run_dir, name), os.path.join(target, name))

    baseline_rows = []
    for name in ('identity', 'wiener'):
        means = test_means(os.path.join(args.results_dir, f'baseline_{name}'))
        if means:
            config = read_json(os.path.join(args.results_dir, f'baseline_{name}', 'config.json')) or {}
            row = {'baseline': name, 'gamma': config.get('gamma', ''), **{c: '' for c in METRIC_COLS}, **means}
            baseline_rows.append(row)

    write_csv(os.path.join(out_dir, 'all_runs.csv'), rows, RUN_COLS)
    write_csv(os.path.join(out_dir, 'baselines.csv'), baseline_rows, ['baseline', 'gamma', *METRIC_COLS])

    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    figure_spectra_rho_bands(groups, args.results_dir, args.results_dir, out_dir)
    figure_train_curves(rows, args.results_dir, out_dir)
    figure_lambda_sweep(rows, out_dir)
    figures = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))

    write_report(rows, baseline_rows, alerts(rows, args.results_dir), figures, out_dir)

    zip_path = os.path.join(os.path.dirname(os.path.normpath(out_dir)) or '.',
                            f"report_{os.path.basename(os.path.normpath(out_dir))}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for d, _, files in os.walk(out_dir):
            for f in sorted(files):
                if not f.endswith('.pt'):
                    full = os.path.join(d, f)
                    z.write(full, os.path.relpath(full, os.path.dirname(os.path.normpath(out_dir))))
    print(f"Report: {out_dir} ({folder_size(out_dir) / 1e6:.2f} MB)")
    print(f"Zip:    {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
