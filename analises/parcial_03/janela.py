"""Partial 03: windowed (Hann) / interior spectral metrics on the TEST set for the
finished block-A seed-0 runs + identity + Wiener. CPU only. Reuses parcial_02."""
import csv, json, os, sys, importlib.util
import numpy as np, torch
OUT = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('diag', os.path.join(OUT, '..', 'parcial_02', 'diagnostico.py'))
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
torch.set_num_threads(12)
D.RUNS = {'SRCNN': 'results/A_srcnn_res_lam0_seed0', 'DSC/MS': 'results/A_dscms_res_lam0_seed0',
          'PRUSR': 'results/A_prusr_res_lam0_seed0', 'SRNO': 'results/A_srno_res_lam0_seed0'}
_, hr, preds = D.predictions('test')
K = D.K; rows = []
edge = np.zeros((128, 128), bool); edge[:8] = edge[-8:] = True; edge[:, :8] = edge[:, -8:] = True
ht = D.band_component(hr.numpy(), 40, 64)
for name, p in preds.items():
    row = {'modelo': name, 'MSE': float(((p - hr) ** 2).mean())}
    for var in ('a_como_esta', 'b_hann', 'd_sem_eixos'):
        T, eps, coh, pp, tt, ee = D.pooled(p, hr, var)
        tag = {'a_como_esta': '', 'b_hann': ' Hann', 'd_sem_eixos': ' sem eixos'}[var]
        row[f'eps_hf{tag}'] = eps[K >= 10].mean()
        row[f'eps pond k>=10{tag}'] = ee[K >= 10].sum() / tt[K >= 10].sum()
        row[f'eps pond 10-40{tag}'] = ee[(K >= 10) & (K < 40)].sum() / tt[(K >= 10) & (K < 40)].sum()
        above = np.flatnonzero(eps > 0.5); row[f'k_c{tag}'] = int(K[above[0]]) - 1 if len(above) else 63
        for k in (25, 31, 40, 60):
            row[f'eps k{k}{tag}'] = eps[k - 1]; row[f'T k{k}{tag}'] = T[k - 1]
    hp = D.band_component(p.numpy(), 40, 64)
    for lab, m in (('R2 k>=40 todos', np.ones((128, 128), bool)), ('R2 k>=40 interior', ~edge)):
        row[lab] = float(np.mean([1 - ((hp[:, c][:, m] - ht[:, c][:, m]) ** 2).sum() / ((ht[:, c][:, m] - ht[:, c][:, m].mean()) ** 2).sum() for c in (0, 1)]))
    rows.append(row); print(name, 'ok', flush=True)
with open(os.path.join(OUT, 'janela_teste.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('done')
