"""Partial analysis 01 (regime A, seed 0: SRCNN and DSC/MS + identity and
Wiener), read only, CPU. Run from the repository root:

    CUDA_VISIBLE_DEVICES="" ../Vinicius/venv/bin/python ../analises/parcial_01/analise.py

Reads results/<run>/{config.json, summary.json, metrics_per_sample.csv,
spectra.npz, eval_summary.json} and the report's all_runs.csv / baselines.csv;
writes tabela_parcial_01.csv and consistencia.txt next to this script.
"""

import csv
import json
import math
import os
import sys

import numpy as np

REPO = '/home/lcadame/Vinicius_Thesis/Vinicius'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from collect_results import SLOPE_RANGE  # noqa: E402

R_EXPECTED, OFFSET_EXPECTED, LPIPS_EXPECTED = 5.326794, -2.457645, 2.869149
STEPS = {'srcnn': 217, 'dscms': 217, 'prusr': 217, 'srno': 433}
BANDS = [('k<5', 1, 4), ('5-10', 5, 9), ('10-20', 10, 19), ('20-40', 20, 39), ('>=40', 40, 63)]
D0 = 10
ITEMS = [('SRCNN', 'A_srcnn_res_lam0_seed0'), ('DSC/MS', 'A_dscms_res_lam0_seed0'), ('PRUSR', 'A_prusr_res_lam0_seed0'), ('SRNO', 'A_srno_res_lam0_seed0'),
         ('identidade', 'baseline_identity'), ('Wiener', 'baseline_wiener')]


def load(name):
    d = os.path.join(REPO, 'results', name)
    j = lambda f: json.load(open(os.path.join(d, f))) if os.path.exists(os.path.join(d, f)) else {}  # noqa: E731
    rows = list(csv.DictReader(open(os.path.join(d, 'metrics_per_sample.csv'))))
    return j('config.json'), j('summary.json'), rows, np.load(os.path.join(d, 'spectra.npz'))


def slope(spec):
    k = np.arange(1, spec.shape[1] + 1)
    sel = (k >= SLOPE_RANGE[0]) & (k <= SLOPE_RANGE[1])
    x = np.log(k[sel]); xc = x - x.mean()
    y = np.log(spec[:, sel])
    return float((((y - y.mean(1, keepdims=True)) * xc).sum(1) / (xc ** 2).sum()).mean())


def main():
    lines, table = [], []
    consts = {}
    for label, name in ITEMS:
        cfg, summ, rows, sp = load(name)
        mse = np.array([float(r['mse']) for r in rows]); psnr = np.array([float(r['psnr']) for r in rows])
        R = cfg.get('psnr_ssim_R')
        dev = float(np.max(np.abs(psnr - 10 * np.log10(R ** 2 / mse))))
        consts[label] = (cfg.get('psnr_ssim_R'), cfg.get('hr_train_min'), cfg.get('lpips_scale'))
        lines.append(f"{label:10s} R={R:.6f} offset={cfg.get('hr_train_min'):.6f} lpips_scale={cfg.get('lpips_scale'):.6f} "
                     f"max|PSNR-10log10(R^2/MSE)|={dev:.2e} dB  n={len(rows)}")
        p, t, e = sp['pred_power'].sum(0), sp['target_power'].sum(0), sp['error_power'].sum(0)
        eps, T = e / t, p / t
        k = np.arange(1, len(t) + 1)
        above = np.flatnonzero(eps > 0.5); kc = int(above[0]) if len(above) else len(eps)
        above01 = np.flatnonzero(eps > 0.1); kc01 = int(above01[0]) if len(above01) else len(eps)
        row = {
            'modelo': label,
            'MSE (m2/s2)': float(mse.mean()), 'PSNR (dB)': float(psnr.mean()),
            'SSIM': float(np.mean([float(r['ssim']) for r in rows])),
            'KE': float(np.mean([float(r['ke']) for r in rows])), 'WKE': float(np.mean([float(r['wke']) for r in rows])),
            'eps_hf (media eps(k), k>=10)': float(eps[k >= D0].mean()),
            'eps_k>=D0 (ponderado)': float(e[k >= D0].sum() / t[k >= D0].sum()),
            **{f'eps banda {b}': float(e[(k >= lo) & (k <= hi)].sum() / t[(k >= lo) & (k <= hi)].sum()) for b, lo, hi in BANDS},
            **{f'T banda {b}': float(p[(k >= lo) & (k <= hi)].sum() / t[(k >= lo) & (k <= hi)].sum()) for b, lo, hi in BANDS},
            **{f'MSE banda {b}': float(np.mean([float(r[f'band_{n}']) for r in rows]))
               for (b, _, _), n in zip(BANDS, ('k0_5', 'k5_10', 'k10_20', 'k20_40', 'k40_max'))},
            'k_c': f"{kc}+" if kc == len(eps) else str(kc), 'k_c_01': f"{kc01}+" if kc01 == len(eps) else str(kc01),
            'k_c_fase': float(np.mean([float(r['k_c_fase']) for r in rows])),
            'inclinacao (k=11-46)': slope(sp['pred']), 'inclinacao alvo': slope(sp['target']),
            'lpips_vgg': float(np.mean([float(r['lpips_vgg']) for r in rows])) if 'lpips_vgg' in rows[0] else '',
            'parametros': cfg.get('n_params', ''), 'epocas': summ.get('epochs_completed', ''),
            'passos': summ.get('epochs_completed', 0) * STEPS[cfg['model']] if summ else '',
            'horas treino': round(summ['train_hours'], 2) if 'train_hours' in summ else '',
        }
        table.append(row)
    same = len({v for v in consts.values()}) == 1
    lines.append(f"constants identical across all four: {same}; expected R={R_EXPECTED}, offset={OFFSET_EXPECTED}, "
                 f"lpips_scale={LPIPS_EXPECTED}: "
                 + str(all(abs(a - R_EXPECTED) < 1e-6 and abs(b - OFFSET_EXPECTED) < 1e-6 and abs(c - LPIPS_EXPECTED) < 1e-6
                           for a, b, c in consts.values())))
    open(os.path.join(OUT, 'consistencia.txt'), 'w').write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT, 'tabela_parcial_03.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    print('\n'.join(lines))
    for key in table[0]:
        print(f"{key:30s} " + " ".join(f"{(f'{r[key]:.4g}' if isinstance(r[key], float) else str(r[key])):>12s}" for r in table))


if __name__ == '__main__':
    main()
