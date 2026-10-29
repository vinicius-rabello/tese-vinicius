"""Old (no window) vs Hann spectral metrics for every run/baseline that has both
evaluations. Reads only saved files (metrics_per_sample*.csv, spectra*.npz).

    venv/bin/python analises/hann/comparar.py

Writes tables/metricas_espectrais_hann.csv (one row per run and window) and
analises/hann/comparacao.md (old vs new, rankings, pointwise check).
"""

import csv
import os
import sys

import numpy as np
import torch

REPO = '/home/lcadame/Vinicius_Thesis/Vinicius'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
from evaluate_run import pooled_spectral  # noqa: E402

K = np.arange(1, 64)
SEL = (K >= 11) & (K <= 46)
HIGH = K > 46
POINTWISE = ('mse', 'psnr', 'ssim', 'lpips_vgg', 'lpips_alex')
PER_SAMPLE = ('ke', 'wke', 'k_c_fase')
# (name, higher is better?)
SPECTRAL = [('k_c', True), ('k_c_01', True), ('k_c_fase', True), ('eps_w_10_40', False), ('eps_hf', False),
            ('eps_k_ge_D0', False), ('T_11_46_geo', None), ('T_gt46_geo', None), ('slope_11_46', None),
            ('ke', False), ('wke', False)]


def load(run_dir, sfx):
    rows = list(csv.DictReader(open(os.path.join(run_dir, f'metrics_per_sample{sfx}.csv'))))
    sp = np.load(os.path.join(run_dir, f'spectra{sfx}.npz'))
    p, t, e = (torch.from_numpy(sp[f'{n}_power']) for n in ('pred', 'target', 'error'))
    pooled = pooled_spectral(p, t, e)
    T = (p.sum(0) / t.sum(0)).numpy()
    spec = sp['pred'].mean(0)
    x = np.log(K[SEL]); y = np.log(spec[SEL])
    out = {m: float(np.mean([float(r[m]) for r in rows])) for m in POINTWISE + PER_SAMPLE if m in rows[0]}
    out.update({k: pooled[k] for k in ('k_c', 'k_c_01', 'eps_w_10_40', 'eps_hf', 'eps_k_ge_D0')})
    out['T_11_46_geo'] = float(np.exp(np.log(T[SEL]).mean()))
    out['T_gt46_geo'] = float(np.exp(np.log(np.maximum(T[HIGH], 1e-30)).mean()))
    out['slope_11_46'] = float(np.polyfit(x, y, 1)[0])
    out['_rows'] = rows
    return out


def main():
    items = []
    for name in sorted(os.listdir('results')):
        d = os.path.join('results', name)
        if all(os.path.exists(os.path.join(d, f)) for f in ('metrics_per_sample.csv', 'metrics_per_sample_hann.csv',
                                                             'spectra.npz', 'spectra_hann.npz')):
            items.append((name, load(d, ''), load(d, '_hann')))
    table = []
    for name, old, new in items:
        for win, m in (('none', old), ('hann', new)):
            table.append({'run': name, 'window': win, **{k: v for k, v in m.items() if not k.startswith('_')}})
    os.makedirs('tables', exist_ok=True)
    cols = ['run', 'window', *POINTWISE, *[s for s, _ in SPECTRAL]]
    with open('tables/metricas_espectrais_hann.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore'); w.writeheader(); w.writerows(table)

    lines = ['# Métricas espectrais: sem janela × Hann (teste)', '',
             f'{len(items)} runs/baselines com as duas avaliações. Faixa principal k = 11–46; k > 46 à parte.', '']
    # Pointwise check, per sample.
    lines += ['## Métricas pontuais (devem ser idênticas)', '', '| run | máx. dif. relativa por amostra (MSE, PSNR, SSIM, LPIPS) |', '|---|---|']
    for name, old, new in items:
        diffs = []
        for m in POINTWISE:
            if m in old['_rows'][0] and m in new['_rows'][0]:
                diffs.append(max(abs(float(a[m]) - float(b[m])) / (abs(float(a[m])) + 1e-30)
                                 for a, b in zip(old['_rows'], new['_rows'])))
        lines.append(f"| {name} | {max(diffs):.2g} |")
    # Old vs new.
    show = ['k_c', 'k_c_01', 'eps_w_10_40', 'eps_hf', 'T_11_46_geo', 'T_gt46_geo', 'slope_11_46', 'ke']
    lines += ['', '## Antigo → Hann', '', '| run | ' + ' | '.join(show) + ' |', '|---|' + '---|' * len(show)]
    for name, old, new in items:
        lines.append(f"| {name} | " + ' | '.join(f"{old[k]:.3g} → {new[k]:.3g}" for k in show) + ' |')
    # Rankings among trained runs (baselines excluded), per metric.
    runs = [(n, o, h) for n, o, h in items if not n.startswith('baseline_')]
    lines += ['', '## Ranking entre runs treinadas (baselines fora)', '',
              '| métrica | ordem sem janela | ordem com Hann | muda? | Spearman |', '|---|---|---|---|---|']
    for metric, higher in SPECTRAL:
        if higher is None and metric != 'T_11_46_geo':
            continue
        better_high = True if metric == 'T_11_46_geo' else higher
        key = lambda trio, i: -trio[i][metric] if better_high else trio[i][metric]  # noqa: E731
        o = [n for n, *_ in sorted(runs, key=lambda r: key(r, 1))]
        h = [n for n, *_ in sorted(runs, key=lambda r: key(r, 2))]
        ro = {n: i for i, n in enumerate(o)}; rh = {n: i for i, n in enumerate(h)}
        nn = len(o); rho = 1 - 6 * sum((ro[n] - rh[n]) ** 2 for n in o) / (nn * (nn ** 2 - 1)) if nn > 1 else 1
        short = lambda lst: ', '.join(s.replace('_res', '').replace('_seed0', '') for s in lst)  # noqa: E731
        lines.append(f"| {metric} | {short(o)} | {short(h)} | {'sim' if o != h else 'não'} | {rho:.2f} |")
    open(os.path.join(OUT, 'comparacao.md'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
