"""Thesis figures with the current pipeline (read only, CPU):

    CUDA_VISIBLE_DEVICES="" venv/bin/python analises/dissertacao/figuras_espectro.py

1. campos_hr_lr_espectro.(png|pdf): HR and LR (Gaussian low-pass, D0 = 10,
   N = 128) u and v of one test day, with their radial KE spectra. Replaces
   notebooks/ek_vis.ipynb, which used an obsolete dataset module and sigma = 5.
2. espectro_medio_hr_teste.(png|pdf) and espectro_medio_hr_teste.json: mean
   radial spectrum of 50 evenly spaced HR test fields, log-log slope fitted
   over k = 11..46 (collect_results.SLOPE_RANGE), also with a Hann window.
"""

import json
import os
import sys

import numpy as np
import torch

REPO = '/home/lcadame/Vinicius_Thesis/Vinicius'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import evaluate_run  # noqa: E402
from collect_results import SLOPE_RANGE  # noqa: E402
from datasets.super_res_dataset import SuperResDataset  # noqa: E402
from evaluation import spectral_metrics as sm  # noqa: E402

META = {'png': {'Software': None}, 'pdf': {'CreationDate': None, 'ModDate': None, 'Producer': None}}
L_KM = 515.0          # domain side: 128 px x ~4.03 km (0.04 deg at ~25 N)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), dpi=200, bbox_inches='tight', metadata=META[ext])
    plt.close(fig)


def fit(k, e, lo, hi):
    sel = (k >= lo) & (k <= hi)
    slope, intercept = np.polyfit(np.log(k[sel]), np.log(e[sel]), 1)
    return float(slope), float(intercept)


def main():
    torch.set_grad_enabled(False)
    test = SuperResDataset(evaluate_run.TEST_FILES, downsample_factor=1, freq_threshold=10)
    dates = evaluate_run.read_dates()
    k = np.arange(1, 64)

    # ---- 1. fields + spectra of one test day
    day = dates.index('2023-01-15')
    lr, hr = test[day]
    spec_hr = sm.radial_spectrum(hr[None])[0].numpy()
    spec_lr = sm.radial_spectrum(lr[None])[0].numpy()
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], wspace=0.45, hspace=0.3)
    for row, comp in enumerate(('u (leste)', 'v (norte)')):
        lim = float(max(lr[row].abs().max(), hr[row].abs().max()))
        for col, (field, title) in enumerate(((hr[row], 'HR (alvo)'), (lr[row], 'LR (entrada, D0 = 10)'))):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(field.numpy(), origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim)
            ax.set_title(f'{title}: {comp}, {dates[day]}', fontsize=10)
            ax.set_xlabel('x (px, 0,04°)'); ax.set_ylabel('y (px, 0,04°)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label('m/s')
    ax = fig.add_subplot(gs[:, 2])
    ax.loglog(k, spec_hr, color='black', lw=2, label='HR')
    ax.loglog(k, spec_lr, color='tab:blue', ls='--', lw=2, label='LR')
    ax.loglog(k, spec_hr * np.exp(-k ** 2 / (2 * 10 ** 2)) ** 2, color='tab:orange', ls=':', lw=2,
              label='HR × H(k)², H(k) = exp(−k²/(2·10²))')
    ax.axvline(10, color='gray', ls=':', lw=1)
    ax.set_ylim(spec_hr.min() * 1e-6, spec_hr.max() * 3)
    ax.set_xlabel('k (ciclos por domínio de 128 px; λ ≈ 515/k km)')
    ax.set_ylabel('E(k) (m²/s²)')
    ax.set_title(f'Espectro radial de energia cinética (u e v), {dates[day]}', fontsize=10)
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=9)
    save(fig, 'campos_hr_lr_espectro')

    # ---- 2. mean spectrum of 50 HR test fields + slope
    idx = np.linspace(0, len(test) - 1, 50).round().astype(int)
    hr50 = torch.from_numpy(test.hr_data[idx]).permute(0, 3, 1, 2).contiguous()
    mean_spec = sm.radial_spectrum(hr50).mean(0).numpy()
    w1 = torch.hann_window(128, periodic=False, dtype=torch.float32)
    w2 = (w1[:, None] * w1[None, :])
    mean_hann = sm.radial_spectrum(hr50 * w2).mean(0).numpy() / float((w2 ** 2).mean())
    lo, hi = SLOPE_RANGE
    s, b = fit(k, mean_spec, lo, hi)
    sh, bh = fit(k, mean_hann, lo, hi)
    extra = {f'{a}-{c}': fit(k, mean_spec, a, c)[0] for a, c in ((11, 30), (20, 46), (11, 40))}
    extra_hann = {f'{a}-{c}': fit(k, mean_hann, a, c)[0] for a, c in ((11, 30), (20, 46), (11, 40))}
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(k, mean_spec, color='black', lw=2, label='HR, média de 50 dias do teste')
    ax.loglog(k, mean_hann, color='tab:green', lw=1.3, label='idem, com janela Hann (sem vazamento de borda)')
    sel = (k >= lo) & (k <= hi)
    ax.loglog(k[sel], np.exp(b) * k[sel] ** s * 1.8, color='tab:red', ls='--', label=f'ajuste k = {lo}–{hi}: inclinação {s:.2f}')
    ax.loglog(k[sel], np.exp(bh) * k[sel] ** sh * 0.55, color='tab:green', ls='--', label=f'ajuste com Hann: inclinação {sh:.2f}')
    ax.axvspan(lo, hi, color='gray', alpha=0.08)
    ax.axvline(10, color='gray', ls=':', lw=1)
    ax.set_xlabel('k (ciclos por domínio de 128 px; λ ≈ 515/k km)')
    ax.set_ylabel('E(k) (m²/s²)')
    ax.set_title('Espectro radial médio de energia cinética, HR, conjunto de teste')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, 'espectro_medio_hr_teste')
    out = {'n_campos': int(len(idx)), 'indices': idx.tolist(), 'datas': [dates[i] for i in idx],
           'faixa_k': [lo, hi], 'inclinacao': s, 'inclinacao_hann': sh,
           'inclinacao_outras_faixas': extra, 'inclinacao_outras_faixas_hann': extra_hann,
           'dia_figura_campos': dates[day], 'k': k.tolist(), 'E_medio': mean_spec.tolist(), 'E_medio_hann': mean_hann.tolist()}
    with open(os.path.join(OUT, 'espectro_medio_hr_teste.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k_: v for k_, v in out.items() if k_ not in ('k', 'E_medio', 'E_medio_hann', 'indices', 'datas')},
                     indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
