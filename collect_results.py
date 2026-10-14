"""Collects the results of the runs in results/ into a small report folder and
a .zip to send for analysis (no checkpoints anywhere).

    python collect_results.py [--runs "A_*" ...] [--out report/<timestamp>/] [--no-eval]

Outputs in --out: all_runs.csv, baselines.csv, figures (PNG + PDF), report.md,
and runs/<run_id>/ with each run's config.json, summary.json, train_log.csv.
A finished run without test metrics is evaluated once (test set, no_grad) and
its test means are written into summary.json, so the next call doesn't repeat
it; --no-eval leaves those columns empty instead.

    python collect_results.py --split val [--out report/<timestamp>_val/]

Validation metrics (for model selection; never the test set): reads and, if
missing, writes the *_val files of evaluate_run.py --split val, and every
output gets the suffix _val (all_runs_val.csv, baselines_val.csv,
report_val.md, figures). It also applies the a priori lambda rule
(evaluation/selection.py, DECISIONS.md) to the SRCNN and SRNO sweeps and
writes selecao_lambda_srcnn.csv / selecao_lambda_srno.csv.
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
# k_c: largest k with eps(k') <= 1/2 for all k' <= k (test set pooled);
# k_c_fase: the phase-coherence version (rho), which a blur can't lower;
# T_k / eps_k: T(k) mean and eps(k) pooled over k >= D0.
METRIC_COLS = ['mse', 'psnr', 'ssim', 'ke', 'wke', *BAND_COLS, 'e_banda_k_ge_D0', 'k_c', 'k_c_fase', 'T_k', 'eps_k',
               'inclinacao_espectral', 'lpips_apendice', 'lpips_vgg']
RUN_COLS = ['run_id', 'status', 'arquitetura', 'perda', 'lambda', 'semente', 'residual', 'ruido', 'estagio', 'epocas',
            'motivo_parada', 'tempo_treino_h', 'n_parametros', 'commit', 'arvore_limpa', 'R_psnr',
            'escala_lpips', *METRIC_COLS, 'best_val_mse']
# Added at the end so the existing columns keep their positions: k_c with the
# stricter threshold eps <= 0.1 (test set pooled), and censoring markers
# (True = k_c reached the last shell, so the value is only a lower bound).
K_C_EXTRA = ['k_c_01', 'k_c_censurado', 'k_c_01_censurado']
# eps_hf: mean of the pooled eps(k) over the shells k >= D0 (informative since the
# 2026-10-07 amendment); eps_w_10_40: eps weighted by target power over
# 10 <= k < 40 (the lambda selection metric).
POOLED_EXTRA = K_C_EXTRA + ['eps_hf', 'eps_w_10_40']
RUN_COLS += POOLED_EXTRA
# Split read and reported: 'test' (default, files without suffix) or 'val'
# (--split val: *_val files written by evaluate_run.py --split val, and every
# output file of this script gets _val too). Set once in main().
SPLIT = 'test'
# Window of the FFT-based metrics read and reported: 'none' (default, the
# original files) or 'hann' (--window hann: *_hann files written by
# evaluate_run.py --window hann; every output of this script gets _hann too).
WINDOW = 'none'


def sfx():
    return ('' if SPLIT == 'test' else f'_{SPLIT}') + ('' if WINDOW == 'none' else f'_{WINDOW}')


def split_name():
    return ('test set' if SPLIT == 'test' else 'validation set') + ('' if WINDOW == 'none' else f', {WINDOW} window')
# Spectral-slope fitting range (k, cycles per domain), confirmed by the user
# (proposal: scripts/propose_slope_range.py, target slope -2.11 over it).
SLOPE_RANGE = (11, 46)
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
    rows = read_csv(os.path.join(run_dir, f'metrics_per_sample{sfx()}.csv'))
    if not rows:
        return {}
    col = lambda name: np.array([to_float(r[name]) for r in rows]) if name in rows[0] else None  # noqa: E731
    out = {m: float(np.mean(col(m))) for m in ('mse', 'psnr', 'ssim', 'ke', 'wke') if col(m) is not None}
    if col('k_c_fase') is not None:
        out['k_c_fase'] = float(np.mean(col('k_c_fase')))
    elif col('k_c') is not None:
        # Evaluated before k_c was redefined: that column is the phase version.
        out['k_c_fase'] = float(np.mean(col('k_c')))
    pooled = pooled_from_npz(run_dir)
    if pooled:
        out.update({'k_c': pooled['k_c'], 'T_k': pooled['T_k_ge_D0'], 'eps_k': pooled['eps_k_ge_D0']})
        out.update({c: pooled[c] for c in POOLED_EXTRA})
    for i, name in enumerate(BAND_NAMES):
        if col(f'band_{name}') is not None:
            out[BAND_COLS[i]] = float(np.mean(col(f'band_{name}')))
    if col('band_k_ge_D0') is not None:
        out['e_banda_k_ge_D0'] = float(np.mean(col('band_k_ge_D0')))
    if col('lpips_alex') is not None:
        out['lpips_apendice'] = float(np.mean(col('lpips_alex')))
    if col('lpips_vgg') is not None:
        out['lpips_vgg'] = float(np.mean(col('lpips_vgg')))
    if SLOPE_RANGE is not None and os.path.exists(os.path.join(run_dir, f'spectra{sfx()}.npz')):
        import torch
        from evaluation.spectral_metrics import spectral_slope
        spec = torch.from_numpy(np.load(os.path.join(run_dir, f'spectra{sfx()}.npz'))['pred'])
        out['inclinacao_espectral'] = float(spectral_slope(spec, *SLOPE_RANGE).mean())
    return out


def pooled_from_npz(run_dir):
    """Test-set T(k), eps(k), k_c from the shell sums in spectra.npz (None for
    evaluations made before those sums were stored)."""
    path = os.path.join(run_dir, f'spectra{sfx()}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    if 'error_power' not in d:
        return None
    import torch
    from evaluate_run import pooled_spectral
    return pooled_spectral(*(torch.from_numpy(d[f'{n}_power']) for n in ('pred', 'target', 'error')))


def num_mean(values):
    """Mean of numeric values; NaN if any is missing (e.g. k_c for runs
    evaluated before it was redefined), so figures and tables skip it."""
    if not values or any(v == '' or v is None for v in values):
        return float('nan')
    return float(np.mean(values))


def evaluate_missing(run_dir, device, with_lpips):
    """Evaluates a finished run that has no metrics of SPLIT yet and records
    the means in summary.json (test_metrics / val_metrics), once per run."""
    import evaluate_run
    model, meta = evaluate_run.load_model(run_dir, device)
    constants = {k: meta[k] for k in evaluate_run.CONSTANT_KEYS}
    make_loader = evaluate_run.test_loader if SPLIT == 'test' else evaluate_run.val_loader
    evaluate_run.evaluate(model, constants, run_dir, make_loader(16, evaluate_run.run_noise_std(meta)),
                          device, with_lpips, split=SPLIT, window=WINDOW)
    if WINDOW != 'none':
        return  # windowed metrics stay in their own *_hann files; summary.json is not touched
    summary_path = os.path.join(run_dir, 'summary.json')
    summary = read_json(summary_path)
    summary[f'{SPLIT}_metrics'] = test_means(run_dir)
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
        'ruido': config.get('noise_std', 0.0) if config else '',
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
    for m in METRIC_COLS + POOLED_EXTRA:
        row[m] = ''
    row.update(test_means(run_dir))
    return row


# ------------------------------------------------------------------ figures --
def save(fig, out_dir, name):
    name += sfx()
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=200, bbox_inches='tight', metadata=FIG_META['png'])
    fig.savefig(os.path.join(out_dir, f'{name}.pdf'), bbox_inches='tight', metadata=FIG_META['pdf'])
    plt.close(fig)


def group_key(row):
    """Comparable runs: same loss, lambda, residual flag and stage (all
    architectures and seeds together)."""
    noise = row.get('ruido') or 0
    return f"{row['perda'] or 'na'}_lam{row['lambda']}_{'res' if row['residual'] else 'nores'}" + \
        (f"_noise{noise:g}" if noise else '') + (f"_stage{row['estagio']}" if row['estagio'] != '' else '')


def mean_spectra(run_dir):
    path = os.path.join(run_dir, f'spectra{sfx()}.npz')
    if not os.path.exists(path):
        return None
    d = np.load(path)
    out = {'k': d['k'], 'pred': d['pred'].mean(0), 'target': d['target'].mean(0), 'rho': d['rho'].mean(0)}
    if 'error_power' in d:
        t = d['target_power'].sum(0)
        out['T'] = d['pred_power'].sum(0) / t
        out['eps'] = d['error_power'].sum(0) / t
    return out


def by_arch(rows, results_dir):
    """Mean over seeds per architecture, for rows that have spectra."""
    out = {}
    for row in rows:
        spec = mean_spectra(os.path.join(results_dir, row['run_id']))
        if spec is not None:
            out.setdefault(row['arquitetura'], []).append((row, spec))
    return out


def figure_spectra_rho_bands(groups, results_dir, baselines_dir, out_dir):
    for key, rows in sorted(groups.items()):
        archs = by_arch(rows, results_dir)
        if not archs:
            continue
        noise = rows[0].get('ruido') or 0
        tag = f"_noise{noise:g}" if noise else ""
        identity = mean_spectra(os.path.join(baselines_dir, f'baseline_identity{tag}'))
        wiener = mean_spectra(os.path.join(baselines_dir, f'baseline_wiener{tag}'))
        figure_amplitude(key, archs, identity, wiener, out_dir)
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
        ax.set_title(f'Radial KE spectrum, {split_name()} mean -- {key}')
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
        ax.set_ylabel('Phase coherence ρ(k) (dimensionless)')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'Phase coherence ρ(k) (blind to blur), {split_name()} mean -- {key}')
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
            vals = [num_mean([r[c] for r in rs]) for c in BAND_COLS]
            ax.bar(np.arange(len(BAND_COLS)) + i * width, vals, width, label=f"{arch} (n={len(rs)})")
        ax.set_xticks(np.arange(len(BAND_COLS)) + width * (len(arch_rows) - 1) / 2)
        ax.set_xticklabels(labels)
        ax.set_yscale('log')
        ax.set_xlabel(f'Radial band of k (cycles per domain); D0 = {D0:g}')
        ax.set_ylabel('Error in band (m²/s²)')
        ax.set_title(f'Error by radial band, {split_name()} -- {key}')
        ax.grid(True, axis='y', ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, out_dir, f'error_by_band_{key}')


def figure_amplitude(key, archs, identity, wiener, out_dir):
    """transfer_k (T(k) = E_yhat/E_y) and eps_k (relative spectral error), test
    set pooled, with the input (LR) and Wiener curves for reference."""
    curves = {arch: [s for _, s in items if 'T' in s] for arch, items in archs.items()}
    curves = {a: c for a, c in curves.items() if c}
    if not curves:
        return
    k = next(iter(curves.values()))[0]['k']
    for name, field, ylabel, ref_line in (
            ('transfer_k', 'T', 'Transfer ratio T(k) = E_ŷ(k) / E_y(k)', 1.0),
            ('eps_k', 'eps', 'Relative spectral error ε(k) = E|Ŷ−Y|² / E|Y|²', 0.5)):
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, ref, style in (('Input (LR)', identity, '--'), ('Wiener', wiener, ':')):
            if ref is not None and field in ref:
                ax.semilogy(k, ref[field], color='gray', ls=style, lw=1.5, label=label)
        for arch, specs in sorted(curves.items()):
            ax.semilogy(k, np.mean([s[field] for s in specs], 0), lw=1.3, label=f"{arch} (n={len(specs)})")
        ax.axhline(ref_line, color='red', ls=':', lw=1,
                   label='T = 1 (all energy kept)' if field == 'T' else 'ε = 1/2 (k_c threshold)')
        ax.axvline(D0, color='gray', ls=':', lw=1)
        ax.set_xlabel('Wavenumber k (cycles per 128-px domain)')
        ax.set_ylabel(f'{ylabel} (dimensionless)')
        ax.set_title(f'{ylabel.split(" = ")[0]}, {split_name()} -- {key}')
        ax.grid(True, which='both', ls='--', alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, out_dir, f'{name}_{key}')


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
        mean = lambda col, lam: num_mean([r[col] for r in rs if r['lambda'] == lam])  # noqa: E731
        for ax, col, label in zip(axes, ('mse', 'ke', 'k_c'),
                                  (f'MSE, {split_name()} (m²/s²)', f'KE, {split_name()} (log-spectral, dimensionless)',
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
        samples = read_csv(os.path.join(results_dir, rid, f'metrics_per_sample{sfx()}.csv'))
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


def fmt_kc(row, col):
    """fmt, with a trailing + on a censored k_c / k_c_01."""
    text = fmt(row.get(col, ''))
    return text + '+' if col in ('k_c', 'k_c_01') and row.get(f'{col}_censurado') is True else text


def write_report(rows, baseline_rows, alert_lines, figures, out_dir, selection_lines=()):
    lines = ["# Results report" + ("" if SPLIT == 'test' else f" ({split_name()})"), ""]
    selection_lines = list(selection_lines)
    lines += ["## Alerts", ""] + ([f"- {a}" for a in alert_lines] or ["- none"]) + [""]
    done = sorted([r for r in rows if r['mse'] != ''], key=lambda r: r['mse'])
    pending = [r for r in rows if r['mse'] == '']
    lines += [f"## Runs ({split_name()}, m/s, fixed R), sorted by MSE", ""]
    head = ['run_id', 'epocas', 'motivo_parada', 'mse', 'psnr', 'ssim', 'ke', 'wke', 'e_banda_k_ge_D0', 'k_c', 'k_c_fase',
            'eps_k',
            'best_val_mse', 'k_c_01']
    lines += ["k_c: eps <= 1/2; k_c_01: eps <= 0.1; a trailing + marks a censored value (eps never exceeded the "
              "threshold up to the last shell, so k_c is only a lower bound).", ""]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(fmt_kc(r, c) for c in head) + " |" for r in done]
    if baseline_rows:
        cols = ('ruido', 'mse', 'psnr', 'ssim', 'ke', 'wke', 'e_banda_k_ge_D0', 'k_c', 'k_c_fase', 'eps_k', 'k_c_01')
        lines += ["", "Baselines:", "", "| baseline | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        lines += [f"| {b['baseline']} | " + " | ".join(fmt_kc(b, c) for c in cols) + " |" for b in baseline_rows]
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
                d = {c: num_mean([r[c] for r in other]) - num_mean([r[c] for r in base])
                     for c in ('mse', 'psnr', 'ssim', 'ke', 'wke', 'k_c')}
                diffs.append(f"| {arch} | {residual} | {lam:g} | " + " | ".join(f"{d[c]:+.4g}" for c in d) + " |")
    if diffs:
        lines += ["| arch | residual | λ | ΔMSE | ΔPSNR | ΔSSIM | ΔKE | ΔWKE | Δk_c |",
                  "|---|---|---|---|---|---|---|---|---|"] + diffs
    else:
        lines += ["(no architecture has both an MSE and an MSE+LPIPS run yet)"]
    lines += selection_lines
    lines += ["", "## Figures", ""] + [f"- {f}" for f in figures]
    with open(os.path.join(out_dir, f'report{sfx()}.md'), 'w') as f:
        f.write("\n".join(lines) + "\n")


def lambda_selection(rows, out_dir):
    """--split val: the a priori lambda rule (evaluation/selection.py) on the
    SRCNN and SRNO sweeps (lambda = 0: block A seed 0; lambda > 0: block B),
    one selecao_lambda_<arch>.csv per sweep; returns the report lines."""
    from evaluation.selection import TABLE_COLS, family_lambdas, select_lambda
    lines, chosen = ["", "## Lambda selection (a priori rule, validation set)", ""], {}
    for arch in ('srcnn', 'srno'):
        runs = [r for r in rows if r['arquitetura'] == arch and r['residual'] is True and not r['ruido']
                and r['estagio'] == '' and r['semente'] == 0 and r['mse'] != '' and r['eps_w_10_40'] != '']
        lams = [r['lambda'] for r in runs]
        if 0 not in lams or len(lams) < 2:
            lines.append(f"- {arch}: not enough evaluated runs (lambda = 0 and at least one lambda > 0 on validation).")
            continue
        if len(set(lams)) != len(lams):
            raise SystemExit(f"{arch}: more than one seed-0 run per lambda: {sorted(r['run_id'] for r in runs)}")
        table, chosen[arch], why = select_lambda(arch, [{'run_id': r['run_id'], 'lambda': r['lambda'],
                                                          'mse_val': r['mse'], 'eps_w_val': r['eps_w_10_40'],
                                                          'eps_hf_val': r['eps_hf']} for r in runs])
        write_csv(os.path.join(out_dir, f'selecao_lambda_{arch}.csv'), table, TABLE_COLS)
        lines.append(f"- {arch}: {why} (`selecao_lambda_{arch}.csv`)")
    per_arch, args = family_lambdas(chosen)
    if args:
        lines.append(f"- Decision: `python scripts/make_block_c.py {args} --sweep-arch srno srcnn`; "
                     + ", ".join(f"{a} {lam:g}" for a, lam in per_arch.items()))
    for line in lines[3:]:
        print(line)
    return lines


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
    parser.add_argument('--split', choices=('test', 'val'), default='test',
                        help="val: validation metrics (model selection), all outputs with the suffix _val")
    parser.add_argument('--window', choices=('none', 'hann'), default='none',
                        help="hann: FFT-based metrics with a Hann window (*_hann files); outputs get the suffix _hann")
    args = parser.parse_args()
    global SPLIT, WINDOW
    SPLIT = args.split
    WINDOW = args.window

    stamp = datetime.now().strftime('%Y%m%d-%H%M')
    out_dir = args.out or os.path.join('report', stamp + sfx())
    os.makedirs(out_dir, exist_ok=True)
    patterns = [p for item in args.runs for p in item.replace(',', ' ').split()]

    run_ids = sorted(d for d in os.listdir(args.results_dir)
                     if os.path.isdir(os.path.join(args.results_dir, d)) and not d.startswith('baseline_')
                     and any(fnmatch.fnmatch(d, p) for p in patterns))
    rows = []
    for run_id in run_ids:
        run_dir = os.path.join(args.results_dir, run_id)
        summary = read_json(os.path.join(run_dir, 'summary.json')) or {}
        evaluated = os.path.exists(os.path.join(run_dir, f'metrics_per_sample{sfx()}.csv'))
        if summary.get('status') in COMPLETE and not evaluated and not args.no_eval:
            import torch
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Evaluating {run_id} on the {split_name()} ({device}) ...")
            evaluate_missing(run_dir, device, not args.no_lpips)
        rows.append(run_row(run_id, run_dir))
        target = os.path.join(out_dir, 'runs', run_id)
        os.makedirs(target, exist_ok=True)
        for name in ('config.json', 'summary.json', 'train_log.csv', f'eval_summary{sfx()}.json'):
            if os.path.exists(os.path.join(run_dir, name)):
                shutil.copyfile(os.path.join(run_dir, name), os.path.join(target, name))

    baseline_rows = []
    for folder in sorted(d for d in os.listdir(args.results_dir) if d.startswith('baseline_')):
        means = test_means(os.path.join(args.results_dir, folder))
        if means:
            config = read_json(os.path.join(args.results_dir, folder, 'config.json')) or {}
            row = {'baseline': folder[len('baseline_'):], 'ruido': config.get('noise_std', 0.0),
                   'gamma': config.get('gamma', ''), **{c: '' for c in METRIC_COLS + POOLED_EXTRA}, **means}
            baseline_rows.append(row)

    write_csv(os.path.join(out_dir, f'all_runs{sfx()}.csv'), rows, RUN_COLS)
    write_csv(os.path.join(out_dir, f'baselines{sfx()}.csv'), baseline_rows,
              ['baseline', 'ruido', 'gamma', *METRIC_COLS, *POOLED_EXTRA])

    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)
    figure_spectra_rho_bands(groups, args.results_dir, args.results_dir, out_dir)
    figure_train_curves(rows, args.results_dir, out_dir)
    figure_lambda_sweep(rows, out_dir)
    figures = sorted(f for f in os.listdir(out_dir) if f.endswith('.png'))

    # The lambda rule is defined on the unwindowed metrics (DECISIONS.md), so it only runs with --window none.
    selection = lambda_selection(rows, out_dir) if SPLIT == 'val' and WINDOW == 'none' else ()
    write_report(rows, baseline_rows, alerts(rows, args.results_dir), figures, out_dir, selection)

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
